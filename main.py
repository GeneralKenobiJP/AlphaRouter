import math
import multiprocessing
import time
from collections import deque
from typing import List, Set, Union

import networkx as nx
import numpy as np
import torch
from qgym.envs import Routing
from qgym.envs.routing import BasicRewarder, RoutingState
import matplotlib.pyplot as plt
import random
import cudnn

from torch import nn, Tensor

import graphviz
from torch.amp import GradScaler, autocast
from torch.types import Device
from torchviz import make_dot

from custom_interaction_generator import CustomInteractionGenerator
from plotting import moving_average, plot

from gymnasium.vector import SyncVectorEnv

EMBEDDING_DIM = 20      # AR paper embedding dimension
LOOKAHEAD_LENGTH = 48   # AR paper lookahead length
NUM_HEADS = 6           # AR paper number of heads in multihead attention
NUM_LAYERS = 4          # AR paper number of transformer layers
FFN_DIM = 4 * 21            # Default hidden layer dimensionality of ffn in transformer
N_ENVS: int = 32
THREADS = max(multiprocessing.cpu_count() - 2, 1)

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
        return env
    return _thunk


class PositionalEncoding(nn.Module):
    """
    A positional encoding module.
    Takes in node embedding and adds positional information.
    """
    def __init__(self, embedding_dim=24, lookahead_length=48):
        """
        :param embedding_dim: Embedding dimension of the node embeddings (not including depth embedding)
        :param lookahead_length: The maximal number of the gates ahead that we will analyze
        """
        super().__init__()

        # final_embedding_dim = embedding_dim + 1  # Embedding dim + 1 for depth

        position = torch.arange(lookahead_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(1e5) / embedding_dim))
        pe = torch.zeros(lookahead_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Rounded up to the nearest even number so that odd indices slicing works
        final_embedding_dim_even = embedding_dim if embedding_dim % 2 == 0 else embedding_dim - 1
        pe[:, 1::2] = torch.cos(position * div_term[::final_embedding_dim_even])
        print(f'Final_embedding_dim: {embedding_dim}')
        print(f'Final_embedding_dim_even: {final_embedding_dim_even}')
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the positional encoding
        :param x: input tensor
        """
        # print(f'shape x: {x.shape}, shape pe: {self.pe.shape}')
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return x


class TransformerLayer(nn.Module):
    """
    A single instance of Transformer layer of structure:
    - Multihead attention
    - Add & norm layer
    - Feed-forward network
    - Add & norm layer
    """
    def __init__(self, embedding_dim, num_heads, ffn_dim):
        """
        :param embedding_dim: Embedding dimension of the node embeddings
        :param num_heads: Number of heads in a multihead attention layer
        :param ffn_dim: Dimensionality of the hidden layer in the feed forward layer
        """
        super().__init__()

        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        # self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Applies the transformer layer
        :param x: input tensor
        """
        attention_output, _ = self.multihead_attention(x, x, x)
        x = self.norm1(x + attention_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class Transformer(nn.Module):
    """
    Complete transformer encoder used in the agent with linear decoder
    """
    def __init__(self, embedding_dim, num_heads, ffn_dim, num_layers, lookahead_length, swap_options):
        """
        :param embedding_dim: Embedding dimension of the node embeddings (not including depth embedding)
        :param num_heads: Number of heads in a multihead attention layer
        :param ffn_dim: Dimensionality of the hidden layer in the feed forward layer
        :param num_layers: Number of transformer layers
        :param lookahead_length: The maximal number of the gates ahead that we will analyze
        :param swap_options: Number of possible swaps actions + 1 (to include no swap)
        """
        super().__init__()

        self.positional_encoding = PositionalEncoding(embedding_dim, lookahead_length)

        # embedding_dim = embedding_dim + 1  # Embedding dim + 1 for depth

        # if embedding_dim % num_heads != 0:
        #     # Increase the dimension so that it is a multiple of the number of heads
        #     padded_embedding_dim = (embedding_dim // num_heads + 1) * num_heads
        #     self.padding_dim = padded_embedding_dim - embedding_dim
        #     embedding_dim = padded_embedding_dim
        # else:
        #     self.padding_dim = 0

        self.layers = nn.ModuleList([
            TransformerLayer(embedding_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])

        # Linear decoder that outputs the matrix of shape (number_of_logical_gates x number_of_swap_options)
        self.decoder = nn.Linear(embedding_dim, swap_options)

    def forward(self, x):
        """
        Pads the input accordingly, applies the transformer encoder and linear decoder
        :param x: input tensor
        """
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x)
        return self.decoder(x)


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
              state: torch.Tensor,
              action: Union[int, torch.Tensor],
              reward: float,
              next_state: torch.Tensor,
              done: bool):
        """
        Stores an experience.
        We detach and move everything to CPU to avoid pinning GPU memory.
        """
        # ensure everything is on CPU and detached
        stored_state = state.detach().to(self.device)
        stored_action = (action.detach() if isinstance(action, torch.Tensor)
              else torch.tensor(action, dtype=torch.int, device=self.device))
        stored_reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        stored_next_state = next_state.detach().to(self.device)
        stored_done = torch.tensor(done, dtype=torch.float32, device=self.device)
        self.buffer.append((stored_state, stored_action, stored_reward, stored_next_state, stored_done))

    def sample(self, batch_size: int):
        """
        Samples a batch of transitions, returning:
            states:     float32 tensor (B, *state_shape)
            actions:    int    tensor (B,)
            rewards:    float32 tensor (B,)
            next_states float32 tensor (B, *state_shape)
            dones:      float32 tensor (B,)
        all moved to self.device.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        next_states = torch.stack(next_states, dim=0)
        dones = torch.stack(dones, dim=0)

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
        self.qubit_embedding = nn.Embedding(num_embeddings=connection_graph.number_of_nodes(), embedding_dim=embedding_dim).to(self.device)

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

        self.to(self.device)

        self.scaler = GradScaler(self.device.type)
        self.optimizer = torch.optim.Adam(list(self.transformer.parameters()) + list(self.qubit_embedding.parameters()), lr=0.001, weight_decay=0.1)

        # self.pool = multiprocessing.Pool(THREADS)

        env_fns = [make_env(connection_graph, ManhattanRewarder()) for _ in range(N_ENVS)]
        self.vec_env = SyncVectorEnv(env_fns)
        self.vec_env.reset()

    def compute_returns(self, rewards, gamma=0.99):
        """
        Given a list of rewards [r0, r1, r2, …, r_{T-1}],
        returns a list of discounted returns:
          G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + …
        """
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # (Optionally) normalize returns to reduce variance:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def _pad_and_mask_rewards(self, reward_list: list[list[float]]):
        size = len(reward_list)
        max_length = max(len(rewards) for rewards in reward_list) if size > 0 else 0

        padded_reward_list = torch.zeros((size, max_length), dtype=torch.float32, device=self.device)
        mask = torch.zeros((size, max_length), dtype=torch.float32, device=self.device)

        for i, rewards in enumerate(reward_list):
            length = len(rewards)
            if length == 0:
                continue
            padded_reward_list[i, :length] = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            mask[i, :length] = 1
        return padded_reward_list, mask

    def compute_returns_batched(self, reward_list: list[list[float]], gamma=0.99):
        """
        Given a list of rewards [r0, r1, r2, …, r_{T-1}],
        returns a list of discounted returns:
          G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + …
        """
        padded_reward_list, mask = self._pad_and_mask_rewards(reward_list)

        batch_size, length = padded_reward_list.shape
        returns = torch.zeros((batch_size, length), dtype=padded_reward_list.dtype, device=self.device)

        R = torch.zeros(batch_size, dtype=padded_reward_list.dtype, device=self.device)

        for i in range(length-1, -1, -1):
            R = padded_reward_list[:, i] + gamma * R
            R = R * mask[:, i]
            returns[:, i] = R

        valid = mask.bool()
        vals = returns[valid]
        mean = vals.mean()
        std = vals.std(unbiased=False)
        returns = (returns - mean) / (std + 1e-8)

        return returns, mask

    def learn(self, total_timesteps: int):
        """
        Conduct the learning process of the agent
        :param total_timesteps: Number of episodes during the learning process
        """
        # losses = []
        # swaps = []
        returns = []
        # Compute 50-episode moving average of loss
        ma_window = 50

        if self.device.type == 'cpu':
            for i in range(int(total_timesteps)):
                print(f"Episode: {i}...")
                loss, swap, ep_return = self.episode()
                # losses.append(loss.detach().cpu())
                # swaps.append(swap)
                returns.append(ep_return)
        else:
            for i in range(int(total_timesteps // N_ENVS)):
                print(f"Episode: {i * N_ENVS}...")
                loss, ep_return = self.batch()
                # losses.append(loss.detach().cpu())
                # swaps.append(swap)
                returns.extend(ep_return.detach().cpu())

        # ma_losses = moving_average(losses, window=ma_window)
        # ma_swaps = moving_average(swaps, window=ma_window)
        ma_returns = moving_average(returns, window=ma_window)

        # print(f"Success count: {self.success_count}")
        # print(f"Success %: {self.success_count / total_timesteps * 100}")
        # print(f"Average loss: {self.loss_sum / total_timesteps}")

        episodes = range(1, total_timesteps + 1)

        # plot(episodes, losses, label='Loss per Episode', xlabel='Episode', ylabel='Loss', title='Training Loss per Episode')
        # plot(episodes, swaps, label='Swaps per Episode', xlabel='Episode', ylabel='Swaps', title='Swaps per Episode')
        plot(episodes, returns, label=f'Returns per episode', xlabel='Episode', ylabel='Return', title='Returns per episode')
        # plot(episodes, ma_losses, label=f'{ma_window}-Episode Moving Avg', xlabel='Episode', ylabel='Loss', title='Training Loss 50-Episode MA')
        # plot(episodes, ma_swaps, label=f'{ma_window}-Episode Moving Avg', xlabel='Episode', ylabel='Swaps', title='Swaps per episode 50-Episode MA')
        plot(episodes, ma_returns, label=f'{ma_window}-Episode Moving Avg', xlabel='Episode', ylabel='Return', title='Return per episode 50-Episode MA')

    def compute_embedding(self, idx) -> Tensor:
        return self.embed_nodes(self.vec_env.envs[idx]._state)

    def batch(self):
        self.vec_env.reset()
        self.optimizer.zero_grad()

        # env_states = [this_env._state for this_env in self.vec_env.envs]
        # states = self.pool.map(self.embed_nodes, env_states)
        # states = zip(*self.pool.map(self.compute_embedding, list(range(THREADS))))
        # print(states)

        state_embeddings = torch.stack(
            [self.embed_nodes(env._state) for env in self.vec_env.envs],
            dim=0
        ).to(self.device)
        # print(state_embeddings)

        log_prob_list = [[] for _ in range(N_ENVS)]
        reward_list = [[] for _ in range(N_ENVS)]
        done_list = [False] * N_ENVS

        for _ in range(self.episode_limit):

            with autocast(self.device.type):
                # transformer_output = self.transformer(x)
                # transformer_output = transformer_output.squeeze(0)
                transformer_output = self.transformer(state_embeddings)
                # print(transformer_output)
                # make_dot(transformer_output, params=dict(self.transformer.named_parameters())).render("graph", format='png')

                # Transform the transformer output shape from (number_of_logical_gates x number_of_swap_options) to (number of swap_options)
                # as we want to output the optimal swap action
                swap_scores = transformer_output.sum(dim=1)
                softmax_output = torch.softmax(swap_scores, dim=-1)
                dist = torch.distributions.Categorical(probs=softmax_output, validate_args=False)

                actions = dist.sample()
                log_probs: Tensor = dist.log_prob(actions)
            # return action.item(), log_prob

            # rewards = []
            # observations, rewards, done, truncated, info = self.vec_env.step(actions) #todo: add a condition on done status
            for i in range(N_ENVS):
                if done_list[i] is True:
                    continue
                observations, reward, terminated, truncated, info = self.vec_env.envs[i].step(actions[i])
                done = terminated or truncated
                # rewards.append(reward)
                reward_list[i].append(reward)
                done_list[i] = done
                log_prob_list[i].append(log_probs[i])

            new_embeddings = torch.stack(
                [self.embed_nodes(env._state) for env in self.vec_env.envs],
                dim=0
            ).to(self.device)

            for i in range(N_ENVS):
                self.replay_buffer.store(state_embeddings[i], actions[i], reward_list[i][-1], new_embeddings[i], done_list[i])
            state_embeddings = new_embeddings

            # log_prob_list[i].append(log_probs)
            # reward_list.append(rewards)

        # print(reward_list)

        returns, mask = self.compute_returns_batched(reward_list)

        # tensor_log_prob_list = torch.zeros((N_ENVS, self.episode_limit), dtype=torch.float32, device=self.device)
        for i in range(N_ENVS):
            # length = len(log_prob_list[i])
            # tensor_log_prob_list[i, :length] = torch.tensor(log_prob_list[i], dtype=torch.float32, device=self.device)
            log_prob_list[i] = torch.stack(log_prob_list[i])
            # log_prob_list[i] = nn.functional.pad(log_prob_list[i], , value=0)

        log_prob_tensor = torch.nn.utils.rnn.pad_sequence(log_prob_list, batch_first=True, padding_value=0.0).to(
            self.device)  # (B, T)


        # Policy gradient loss:
        #  per-episode sum_t (log_prob * G_t), then mean across batch
        # note: multiply by mask to ignore padding
        # log_prob_list = torch.stack(log_prob_list, dim=0)
        per_episode_loss = - (log_prob_tensor * returns * mask).sum(dim=1)  # (B,)
        policy_loss = per_episode_loss.mean()

        # policy_loss = []
        # for log_prob, G_t in zip(log_probs, returns):
        #     policy_loss.append(-log_prob * G_t)
        # policy_loss = torch.stack(policy_loss).sum()

        # backprop / gradient step
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # also return useful diagnostics
        # total undisc returns per env for logging:
        # compute undiscounted returns per env as sum of original lists
        # print(f'Reward list: {reward_list}')
        # print(f'Reward list shape: {np.array(reward_list).shape}')
        undiscounted_returns = torch.tensor([sum(r) for r in reward_list], dtype=torch.float32, device=returns.device)
        # print(f'undiscounted returns {undiscounted_returns}')

        return policy_loss, undiscounted_returns
        # return policy_loss, swaps, sum(rewards)


    def episode(self) -> (Tensor, int, int):
        """
        Run an episode, i.e. try to solve one particular circuit
        """
        # Generate a new interaction circuit
        self.env.reset()
        self.optimizer.zero_grad()

        log_probs = []
        rewards = []
        # try:
        #     render_rgb(self.env)
        # except:
        #     pass

        # interaction_circuit = state.interaction_circuit[:self.lookahead_length]
        # print(f'interaction_circuit: {interaction_circuit}')

        swaps = 0
        state_embedding = self.embed_nodes(self.env._state)

        for _ in range(self.episode_limit):
            # print(state.mapping)
            action, log_prob = self.get_action(state_embedding)
            observation, reward, done, truncated, info = self.env.step(action)

            new_embedding = self.embed_nodes(self.env._state)
            self.replay_buffer.store(state_embedding, action, reward, new_embedding, done)
            state_embedding = new_embedding

            log_probs.append(log_prob)
            rewards.append(reward)

            swaps += 1
            if done:
                self.success_count += 1
                break

        print(f'swaps: {swaps}')

        returns = self.compute_returns(rewards)
        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns):
            policy_loss.append(-log_prob * G_t)
        policy_loss = torch.stack(policy_loss).sum()

        # backprop / gradient step
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        print(f"Loss: {policy_loss}")

        return policy_loss, swaps, sum(rewards)

    def get_action(self, node_embedding) -> (int, int):
        """
        Based on the current environment state, get an action suggested by the agent
        :param node_embedding: Node embedding of the current state, should be an output of self.embed_nodes() right before this function
        :return: Action - index of the swap option (last option is no swap)
        """
        with autocast(self.device.type):
            x = node_embedding.unsqueeze(0)
            # transformer_output = self.transformer(x)
            # transformer_output = transformer_output.squeeze(0)
            transformer_output = self.transformer(x)
            transformer_output = transformer_output.squeeze(0)
            # make_dot(transformer_output, params=dict(self.transformer.named_parameters())).render("graph", format='png')

            # Transform the transformer output shape from (number_of_logical_gates x number_of_swap_options) to (number of swap_options)
            # as we want to output the optimal swap action
            swap_scores = transformer_output.sum(dim=0)
            softmax_output = torch.softmax(swap_scores, dim=-1)
            dist = torch.distributions.Categorical(probs=softmax_output, validate_args=False)

            # action = torch.multinomial(softmax_output, 1)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # print(f'Node embedding: {node_embedding}')
        # print(f'Transformer output: {transformer_output}')
        # print(f'Transformer output shape: {transformer_output.shape}')
        # print(f'Swap scores: {swap_scores}')
        # print(f'Swap scores shape: {swap_scores.shape}')
        # print(f'Softmax output: {softmax_output}')
        # print(f'Softmax output shape: {softmax_output.shape}')
        # print(f'Action: {action}')
        # print(f'Intified action: {int(action)}')
        # print(f'Logical q0: {state.mapping[0]}')
        # print(f'Logical q7: {state.mapping[7]}')

        return action.item(), log_prob

    def embed_nodes(self, state: RoutingState):
        """
        Return the node embedding based on the current environment.

        We observe a maximum of self.lookahead_length gates and for each gate we produce a gate embedding in the following way:
        - We retrieve logical qubits involved in the gate
        - We retrieve physical qubits mapped to the logical qubits
        - For each physical qubit we produce a qubit embedding with nn.Embedding
        - We take a mean of these two qubit embeddings
        - We concatenate this embedding with the depth of the gate, where depth is equal to the number of another gates
            this gate depends on + 1

        This procedure yields a node embedding matrix of shape (Lookahead Length x (Embedding Dimension + 1))

        :return: Node embedding
        """
        interaction_circuit = state.interaction_circuit[:self.lookahead_length]  # Limit lookahead depth

        feature_matrix = []
        qubit_gate_dependencies: List[Set[int]] = []
        for _ in range(state.n_qubits):
            qubit_gate_dependencies.append(set())

        for idx, (logical_q1, logical_q2) in enumerate(interaction_circuit):
            physical_q1 = state.mapping[logical_q1]
            physical_q2 = state.mapping[logical_q2]

            embedding_q1 = self.qubit_embedding(torch.tensor(physical_q1, device=self.device))
            embedding_q2 = self.qubit_embedding(torch.tensor(physical_q2, device=self.device))

            gate_dependencies = qubit_gate_dependencies[logical_q1].union(qubit_gate_dependencies[logical_q2])
            depth = len(gate_dependencies) + 1

            gate_vector = (embedding_q1 + embedding_q2) / 2  # Mean of qubit encodings
            gate_vector = nn.functional.pad(gate_vector, (0, 1), value=depth)  # Append depth
            # Pad the input with 0s so that the dimensionality is a multiple of the number of heads
            gate_vector = nn.functional.pad(gate_vector, (0, self.padding_dim), value=0)  # Append padding dimension

            qubit_gate_dependencies[logical_q1].add(idx)
            qubit_gate_dependencies[logical_q2].add(idx)

            feature_matrix.append(gate_vector)

        if len(feature_matrix) == 0:
            return torch.zeros((self.lookahead_length, self.embedding_dim + 1), device=self.device)

        return torch.stack(feature_matrix)


class ManhattanRewarder(BasicRewarder):

    def __init__(self):
        super().__init__()

    def compute_reward(
        self,
        *,
        old_state: RoutingState,
        action: int,
        new_state: RoutingState,
    ) -> float:
        interaction_circuit = new_state.interaction_circuit
        if len(interaction_circuit) < 1:
            return 0
        return -np.abs(new_state.mapping[interaction_circuit[0][0]] - new_state.mapping[interaction_circuit[0][1]])

class GridManhattanRewarder(BasicRewarder):

    def __init__(self, width: int):
        super().__init__()
        self.n = width

    def compute_reward(
        self,
        *,
        old_state: RoutingState,
        action: int,
        new_state: RoutingState,
    ) -> float:
        interaction_circuit = new_state.interaction_circuit
        if len(interaction_circuit) < 1:
            return 0
        q = new_state.mapping[interaction_circuit[0][0]]
        p = new_state.mapping[interaction_circuit[0][1]]
        n1 = q % self.n
        n2 = p % self.n
        m1 = q / self.n
        m2 = p / self.n
        return - (np.abs(n2-n1) + np.abs(m2-m1))

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
    # action_space = env.action_space
    # observation_space = env.observation_space
    # print(action_space)
    # print(observation_space)

    print(f"cuda version: {torch.version.cuda}")  # e.g. "11.8"
    print(f"cuda availability: {torch.cuda.is_available()}")
    print(f"device count: {torch.cuda.device_count()}")  # number of GPUs
    print(f"devices: {torch.cuda.get_device_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"device: {device}")
    model = AlphaRouterAgent(env, device).to(device)

    # state: RoutingState = model.env._state
    # print(state.interaction_circuit.shape)

    # model.learn(6e2)
    # model.learn(500)
    # model.batch()
    # model.learn(5)
    start = time.time()
    model.learn(1024)
    end = time.time()
    print(f'It took: {end - start} seconds')

    has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.transformer.parameters())
    print("Gradients flowing:", has_grad)

    print('Successful execution')