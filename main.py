import math
import sys
import time
from collections import deque
from copy import deepcopy
from typing import List, Set, Union

import networkx as nx
import numpy as np
import torch
from qgym.envs import Routing
from qgym.envs.routing import RoutingState
import matplotlib.pyplot as plt
import random

from torch import nn, Tensor

import graphviz
from torch.amp import GradScaler, autocast
from torch.types import Device
from torchviz import make_dot

from custom_interaction_generator import CustomInteractionGenerator
from mcts import MCTS, Node
from plotting import moving_average, plot
from manhattan_rewarder import ManhattanRewarder

from gymnasium.vector import SyncVectorEnv

from transformer import Transformer
from utils import NormalizeInfoWrapper, embed_nodes, compute_returns, compute_returns_batched

EMBEDDING_DIM = 20      # AR paper embedding dimension
LOOKAHEAD_LENGTH = 48   # AR paper lookahead length
NUM_HEADS = 6           # AR paper number of heads in multihead attention
NUM_LAYERS = 4          # AR paper number of transformer layers
FFN_DIM = 4 * 21            # Default hidden layer dimensionality of ffn in transformer
N_ENVS: int = 64

REPLAY_BUFFER_MIN_SIZE_MCTS: int = 320

def render_rgb(env):
    """Convenience method that we will use later on to display our results. Copied from notebooks"""
    plt.figure(figsize=(40, 20))
    plt.title(f"Step {env._state.steps_done}", fontsize=40)
    plt.imshow(env.render())
    plt.axis("off")
    plt.show()


def make_env(connection_graph, rewarder=None):
    def _thunk():
        env = Routing(
            max_observation_reach=LOOKAHEAD_LENGTH,
            observe_legal_surpasses=True,
            observe_connection_graph=True,
            connection_graph=connection_graph,
            render_mode="rgb_array",
            rewarder=rewarder,
            interaction_generator=CustomInteractionGenerator()
        )
        # env.seed(seed)
        env = NormalizeInfoWrapper(env)
        return env
    return _thunk


class ReplayBuffer:
    """
    A simple replay buffer for standard RL:
      - stores transitions (state, action, reward, next_state, done)
      - samples minibatches of transitions as tensors
    """
    def __init__(self, capacity: int, device: torch.device):
        """
        :param capacity: maximum number of transitions to store
        :param device:   torch.device to which sampled batches will be moved
        """
        self.capacity = capacity
        self.device   = device
        self.buffer   = deque(maxlen=capacity)

    def store(self,
              state: RoutingState,
              action: Union[int, torch.Tensor],
              reward: float,
              next_state: RoutingState,
              done: bool):
        """
        Stores an experience.
        We detach and move everything to CPU to avoid pinning GPU memory.
        """
        # ensure everything is on CPU and detached
        stored_action = (action.detach() if isinstance(action, torch.Tensor)
                         else action)
        self.buffer.append((state, stored_action, reward, deepcopy(next_state), done))

    def sample(self, batch_size: int):
        """
        Samples a batch of transitions, returning:
            states:     float32 tensor (B, *state_shape)
            actions:    int    tensor (B,)
            rewards:    float (B,)
            next_states float32 tensor (B, *state_shape)
            dones:      bool (B,)
        all moved to self.device.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = deepcopy(states[0])
        actions = actions[0].to(self.device)
        rewards = rewards[0]
        next_states = deepcopy(next_states[0])
        dones = dones[0]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class AlphaRouterAgent(nn.Module):
    """The AlphaRouter routing agent"""
    def __init__(self, env: Routing, device: Device, embedding_dim: int = 20, num_heads: int = 6, ffn_dim: int = 64,
                 num_layers: int = 4, lookahead_length: int = 48, buffer_size: int = int(1e6), *args, **kwargs):
        """
        :param env: Particular routing environment we want to apply the agent to
        :param device: Device to run the agent on (cuda for Nvidia graphics card, cpu for a standard cpu)
        :param embedding_dim: Embedding dimension of the node embeddings (not including depth embedding)
        :param num_heads: Number of heads in a multihead attention layer
        :param ffn_dim: Dimensionality of the hidden layer in the feed forward layer
        :param num_layers: Number of transformer layers
        :param lookahead_length: The maximal number of the gates ahead that we will analyze
        :param buffer_size: Size of the replay buffer (its capacity)
        """
        super().__init__(*args, **kwargs)
        self.device = device
        self.env = env
        self.env.rewarder = ManhattanRewarder()
        # self.env.rewarder = GridManhattanRewarder(5)
        self.reward = 0
        self.episode_limit = 150  # The maximal number of steps in a single episode
        self.success_count = 0  # How many times did we achieve a successful routing during the learning process
        self.loss_sum = 0
        self.lookahead_length = lookahead_length
        self.embedding_dim = embedding_dim
        state: RoutingState = self.env._state
        connection_graph = state.connection_graph
        self.qubit_embedding = nn.Embedding(num_embeddings=connection_graph.number_of_nodes(),
                                            embedding_dim=embedding_dim).to(self.device)

        embedding_dim = embedding_dim + 1  # Embedding dim + 1 for depth

        if (embedding_dim) % num_heads != 0:
            # Increase the dimension so that it is a multiple of the number of heads
            padded_embedding_dim = (embedding_dim // num_heads + 1) * num_heads
            self.padding_dim = padded_embedding_dim - embedding_dim
            embedding_dim = padded_embedding_dim
        else:
            self.padding_dim = 0

        self.transformer = Transformer(embedding_dim, num_heads, ffn_dim, num_layers, lookahead_length, swap_options=len(connection_graph.edges) + 1).to(self.device)
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.device
        )
        self.mcts = MCTS(make_env(connection_graph, ManhattanRewarder())(), self.transformer,
                         self.qubit_embedding, self.device, swap_options=len(connection_graph.edges) + 1,
                         lookahead_length=self.lookahead_length, embedding_dim=self.embedding_dim,
                         padding_dim=self.padding_dim, rollouts=15, max_depth=100)

        self.to(self.device)

        self.scaler = GradScaler(self.device.type)
        self.optimizer = torch.optim.Adam(list(self.transformer.parameters()) + list(self.qubit_embedding.parameters()), lr=0.001, weight_decay=0.1)

        env_fns = [make_env(connection_graph, ManhattanRewarder()) for _ in range(N_ENVS)]
        self.vec_env = SyncVectorEnv(env_fns)

    def get_action(self, node_embedding) -> (Tensor, Tensor):
        """
        Based on the current environment state, get an action suggested by the agent
        :param node_embedding: Node embedding of the current state, should be an output of self.embed_nodes() right before this function
        :return:
            Action - tensor of the index of the swap option (last option is no swap)
            log_prob - tensor of log probability of the action being optimal as predicted by the model
        """
        with autocast(self.device.type):
            transformer_output = self.transformer(node_embedding)
            # make_dot(transformer_output, params=dict(self.transformer.named_parameters())).render("graph", format='png')

            # Transform the transformer output shape from (number_of_logical_gates x number_of_swap_options) to (number of swap_options)
            # as we want to output the optimal swap action
            swap_scores = transformer_output.sum(dim=1)
            softmax_output = torch.softmax(swap_scores, dim=-1)
            dist = torch.distributions.Categorical(probs=softmax_output, validate_args=False)

            action = dist.sample()
            log_prob: Tensor = dist.log_prob(action)

        return action, log_prob

    def episode(self) -> (Tensor, int, int):
        """
        Run an episode, i.e. try to solve one particular circuit

        Returns
        -------
        A tuple of 3 elements:
        - A tensor of losses of shape (#steps)
        - A list of swap numbers per each environment
        - A tensor of undiscounted rewards per each environment (for plotting and analysis)
        """
        # Generate a new interaction circuit
        self.env.reset()
        self.optimizer.zero_grad()

        log_probs = []
        rewards = []

        swaps = 0

        for _ in range(self.episode_limit):
            old_state = deepcopy(self.env._state)

            state_embedding = embed_nodes(self.env._state, self.lookahead_length, self.device,
                                          self.qubit_embedding, self.embedding_dim, self.padding_dim)
            action, log_prob = self.get_action(state_embedding.unsqueeze(0))
            observation, reward, done, truncated, info = self.env.step(action)

            self.replay_buffer.store(old_state, action, reward, self.env._state, done)

            log_probs.append(log_prob)
            rewards.append(reward)

            swaps += 1
            if done:
                self.success_count += 1
                break

        returns = compute_returns(rewards)
        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns):
            policy_loss.append(-log_prob * G_t)
        policy_loss = torch.stack(policy_loss).sum()

        # backprop / gradient step
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # MCTS
        self.optimizer.zero_grad()
        # if not len(self.replay_buffer.buffer) > REPLAY_BUFFER_MIN_SIZE_MCTS:
        #     continue
        mcts_input = self.replay_buffer.sample(1)
        # mcts_input = self.env._state
        # self.mcts.run(Node(mcts_input[0]))
        mcts_reward = self.mcts.run(mcts_input[0])
        # if mcts_reward is not None:

            # MCTS backprop / gradient step
            # self.scaler.scale(mcts_reward).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()

        return policy_loss, swaps, sum(rewards)

    def batch(self) -> (Tensor, list[int], Tensor):
        """
        Process one batch of environments, one episode each.
        One episode means trying to solve one particular circuit.

        This function serves the role of the episode() function called for each environment.
        The loss is averaged over all environments to reduce variance and computation.

        Returns
        -------
        A tuple of 3 elements:
        - A tensor of losses of shape (N_ENVS, limit of steps)
        - A list of swap numbers per each environment
        - A tensor of undiscounted rewards per each environment (for plotting and analysis)
        """
        self.vec_env.reset()
        self.optimizer.zero_grad()

        state_embeddings = torch.stack(
            [embed_nodes(cur_env._state, self.lookahead_length, self.device, self.qubit_embedding,
                              self.embedding_dim, self.padding_dim) for cur_env in self.vec_env.envs],
            dim=0
        ).to(self.device)

        log_prob_list = [[] for _ in range(N_ENVS)]
        reward_list = [[] for _ in range(N_ENVS)]
        done_list = [False] * N_ENVS
        swaps = [self.episode_limit] * N_ENVS

        for step_idx in range(self.episode_limit):
            old_states = [deepcopy(cur_env._state) for cur_env in self.vec_env.envs]

            actions, log_probs = self.get_action(state_embeddings)

            observations, rewards, terminated, truncated, info = self.vec_env.step(actions)
            done = np.logical_or(terminated, truncated)

            for i in range(N_ENVS):
                if done_list[i]:
                    continue
                if done[i]:
                    done_list[i] = True
                    swaps[i] = step_idx
                    self.success_count += 1
                log_prob_list[i].append(log_probs[i])
                reward_list[i].append(rewards[i])

            new_embeddings = torch.stack(
                [embed_nodes(env._state, self.lookahead_length, self.device, self.qubit_embedding,
                                  self.embedding_dim, self.padding_dim) for env in self.vec_env.envs],
                dim=0
            ).to(self.device)

            for i in range(N_ENVS):
                self.replay_buffer.store(old_states[i], actions[i], reward_list[i][-1],
                                         self.vec_env.envs[i]._state, done_list[i])
                # self.replay_buffer.store(state_embeddings[i], actions[i], reward_list[i][-1], new_embeddings[i], done_list[i])
            state_embeddings = new_embeddings

        returns, mask = compute_returns_batched(reward_list, device=self.device)

        # Convert log_prob_list to tensor for fast on-device computation
        for i in range(N_ENVS):
            log_prob_list[i] = torch.stack(log_prob_list[i])
        log_prob_tensor = torch.nn.utils.rnn.pad_sequence(log_prob_list, batch_first=True, padding_value=0.0).to(
            self.device)  # (B, T)

        # Policy gradient loss:
        #  per-episode sum_t (log_prob * G_t), then mean across batch
        # note: multiply by mask to ignore padding
        per_episode_loss = - (log_prob_tensor * returns * mask).sum(dim=1)  # (B,)
        policy_loss = per_episode_loss.mean()

        # backprop / gradient step
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Return undiscounted returns per env for plotting and analysis
        undiscounted_returns = torch.tensor([sum(r) for r in reward_list], dtype=torch.float32, device=returns.device)

        return policy_loss, swaps, undiscounted_returns



    def learn(self, total_timesteps: int, ma_window: int = 50, batch: bool = True) -> None:
        """
        Conduct the learning process of the agent
        :param total_timesteps: Number of episodes during the learning process
        :param ma_window: The size of the moving average window used for filtering out noise in the graphs
        :batch: If true - computes in batches with B=N_ENV
        """
        swaps = []
        returns = []

        if not batch:
            for i in range(int(total_timesteps)):
                print(f"Episode: {i}...")
                loss, swap, ep_return = self.episode()
                swaps.append(swap)
                returns.append(ep_return)
        else:
            for i in range(int(total_timesteps // N_ENVS)):
                print(f"Episode: {i * N_ENVS}...")
                loss, ep_swaps, ep_return = self.batch()
                swaps.extend(ep_swaps)
                returns.extend(ep_return.detach().cpu())

        ma_swaps = moving_average(swaps, window=ma_window)
        ma_returns = moving_average(returns, window=ma_window)

        print(f"Success count: {self.success_count}")
        print(f"Success %: {self.success_count / total_timesteps * 100}")

        episodes = range(1, total_timesteps + 1)

        plot(episodes, swaps, label='Swaps per Episode', xlabel='Episode', ylabel='Swaps', title='Swaps per Episode')
        plot(episodes, returns, label=f'Returns per episode', xlabel='Episode', ylabel='Return', title='Returns per episode')
        plot(episodes, ma_swaps, label=f'{ma_window}-Episode Moving Avg', xlabel='Episode', ylabel='Swaps', title='Swaps per episode 50-Episode MA')
        plot(episodes, ma_returns, label=f'{ma_window}-Episode Moving Avg', xlabel='Episode', ylabel='Return', title='Return per episode 50-Episode MA')


if __name__ == '__main__':
    connection_graph = nx.Graph()
    # connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,0)])
    connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,6), (6,7)])
    # connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (5,6), (6,7), (7,8), (8,9), (10,11),(11,12),(12,13), (13,14), (0,5), (5, 10), (1, 6), (6,11), (2,7), (7,12), (3,8), (8,13), (4,9), (9,14)])
    # width = 5
    nx.draw(connection_graph, with_labels=True)
    plt.show()

    # env = InitialMapping(connection_graph=connection_graph, graph_generator=BasicGraphGenerator(0.6), render_mode="human")
    env = Routing(
        max_observation_reach=LOOKAHEAD_LENGTH,
        observe_legal_surpasses=True,
        observe_connection_graph=True,
        connection_graph=connection_graph,
        render_mode="rgb_array",
        rewarder=None,
        interaction_generator=CustomInteractionGenerator()
    )

    print(f"cuda version: {torch.version.cuda}")  # e.g. "11.8"
    print(f"cuda availability: {torch.cuda.is_available()}")
    print(f"device count: {torch.cuda.device_count()}")  # number of GPUs
    print(f"devices: {torch.cuda.get_device_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"device: {device}")
    model = AlphaRouterAgent(env, device).to(device)

    start = time.time()
    model.learn(64, batch=True)
    # model.learn(1024, batch=True)
    end = time.time()
    print(f'It took: {end - start} seconds')

    has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.transformer.parameters())
    print("Gradients flowing:", has_grad)

    print('Successful execution')
    # B = 32
    # 165.20124554634094
    # 104.16623425483704

    # Before vectorization (B = 64)
    # 162.52327227592468
    # 96.18174529075623

    # After vectorization (B = 64)
    # 147.20666241645813
    # 98.53906488418579

    # B = 128
    # 131.8683156967163
    # 90.94579315185547