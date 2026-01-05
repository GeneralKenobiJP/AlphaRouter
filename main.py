import math
import pickle
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import List, Set, Union, Hashable, Any, Optional

import networkx as nx
import numpy as np
import torch
from networkx.algorithms.coloring.greedy_coloring import _Node
from qgym.envs import Routing
from qgym.envs.routing import RoutingState
import matplotlib.pyplot as plt
import random

from qgym.generators.interaction import InteractionGenerator
from torch import nn, Tensor

import graphviz
from torch.amp import GradScaler, autocast
from torch.types import Device
from torchviz import make_dot

from custom_interaction_generator import CustomInteractionGenerator
from mcts import ParallelMCTS, MCTS
from plotting import moving_average, plot, plot_mapping_and_circuit
from manhattan_rewarder import ManhattanRewarder

from gymnasium.vector import SyncVectorEnv

from transformer import Transformer
from utils import NormalizeInfoWrapper, embed_nodes, compute_returns, compute_returns_batched, boost_process_priority, \
    open_binary, get_timestamp

EMBEDDING_DIM = 20      # AR paper embedding dimension
LOOKAHEAD_LENGTH = 48   # AR paper lookahead length
NUM_HEADS = 6           # AR paper number of heads in multihead attention
NUM_LAYERS = 4          # AR paper number of transformer layers
# FFN_DIM = 4 * 21            # Default hidden layer dimensionality of ffn in transformer
D_MODEL = 216
FFN_DIM = 4 * D_MODEL            # Default hidden layer dimensionality of ffn in transformer
N_ENVS: int = 64
MCTS_PROCESSES: int = 1

REPLAY_BUFFER_MIN_SIZE_MCTS: int = 320
# REPLAY_BUFFER_MIN_SIZE_MCTS: int = 65

# LEARNING_RATE = 2e-5
# LEARNING_RATE = 1e-2
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0
LR_DECAY_STEP = N_ENVS * 2
LR_DECAY = 0.8
MAX_GRAD_NORM = 5
LOSS_COEFFICIENT = 0.1

MODEL_DIR = Path(__file__).resolve().parent.joinpath('models/')
TEMP_MODEL_DIR = MODEL_DIR.joinpath('temp/')
TEMP_BUFFER_DIR = TEMP_MODEL_DIR.joinpath('buffers/')
TEMP_TRANSFORMER_DIR = TEMP_MODEL_DIR.joinpath('temp_transformer.pt')
TEMP_EMBEDDING_DIR = TEMP_MODEL_DIR.joinpath('temp_embedding.pt')
LOG_DIR = Path(__file__).resolve().parent.joinpath('logs/')

def render_rgb(env):
    """Convenience method that we will use later on to display our results. Copied from notebooks"""
    plt.figure(figsize=(40, 20))
    plt.title(f"Step {env._state.steps_done}", fontsize=40)
    plt.imshow(env.render())
    plt.axis("off")
    plt.show()


def make_env(connection_graph, interaction_generator: InteractionGenerator = None, rewarder=None):
    def _thunk():
        env = Routing(
            max_observation_reach=LOOKAHEAD_LENGTH,
            observe_legal_surpasses=True,
            observe_connection_graph=True,
            connection_graph=connection_graph,
            render_mode="rgb_array",
            rewarder=rewarder,
            interaction_generator=interaction_generator
            # interaction_generator=CustomInteractionGenerator()
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
        # stored_action = (action.detach() if isinstance(action, torch.Tensor)
        #                  else action)
        # self.buffer.append((state, stored_action, reward, deepcopy(next_state), done))
        self.buffer.append(state)

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
        # states, actions, rewards, next_states, dones = zip(*batch)
        #
        # states = deepcopy(states)
        # next_states = deepcopy(next_states)
        #
        # return states, actions, rewards, next_states, dones

        return batch

    def save_to_disk(self, path: str | Path, compress: Optional[str] = None) -> None:
        path = path if isinstance(path, Path) else Path(path)
        with open_binary(path, 'wb', compress=compress) as f:
            pickle.dump((self.buffer.maxlen, list(self.buffer)), f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Buffer snapshot saved at {path}")

    def load_from_disk(self, path: str | Path, compress: Optional[str] = None) -> None:
        path = path if isinstance(path, Path) else Path(path)
        with open_binary(path, 'rb', compress) as f:
            maxlen, data = pickle.load(f)
        self.buffer = deque(data, maxlen=maxlen)

        print(f"Buffer snapshot loaded from {path}")

    def __len__(self):
        return len(self.buffer)


class AlphaRouterAgent(nn.Module):
    """The AlphaRouter routing agent"""
    def __init__(self, connection_graph: nx.Graph, interaction_generator: InteractionGenerator | None = None,
                 device: Device = torch.device("cpu"), embedding_dim: int = 20, num_heads: int = 6, d_model: int = D_MODEL, ffn_dim: int = FFN_DIM,
                 num_layers: int = 4, lookahead_length: int = 48, buffer_size: int = int(1e6), episode_limit: int = 150,
                 loss_coefficient: float = LOSS_COEFFICIENT, *args, **kwargs):
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
        print('Init')
        super().__init__(*args, **kwargs)
        self.device = device
        self.env = make_env(connection_graph, interaction_generator)
        # self.env.rewarder = ManhattanRewarder()
        # self.env.rewarder = GridManhattanRewarder(5)
        self.reward = 0
        self.episode_limit = episode_limit  # The maximal number of steps in a single episode
        self.success_count = 0  # How many times did we achieve a successful routing during the learning process
        self.loss_sum = 0
        self.lookahead_length = lookahead_length
        self.embedding_dim = embedding_dim
        # state: RoutingState = self.env._state
        # connection_graph = state.connection_graph
        self.qubit_embedding = nn.Embedding(num_embeddings=connection_graph.number_of_nodes(),
                                            embedding_dim=embedding_dim)
        # nn.init.xavier_uniform_(self.qubit_embedding.weight)
        self.qubit_embedding.weight.share_memory_()
        print(self.qubit_embedding.weight)
        self.loss_coefficient = loss_coefficient
        self.action_space_size = len(connection_graph.edges)

        embedding_dim = embedding_dim + 1  # Embedding dim + 1 for depth

        if (embedding_dim) % num_heads != 0:
            # Increase the dimension so that it is a multiple of the number of heads
            padded_embedding_dim = (embedding_dim // num_heads + 1) * num_heads
            self.padding_dim = padded_embedding_dim - embedding_dim
            embedding_dim = padded_embedding_dim
        else:
            self.padding_dim = 0

        self.transformer = Transformer(embedding_dim, num_heads, d_model, ffn_dim, num_layers, lookahead_length,
                                       swap_options=self.action_space_size, device=self.device).to(self.device)
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.device
        )
        self.mcts: ParallelMCTS | MCTS | None = None

        self.to(self.device)

        self.scaler = GradScaler(device=self.device.type)
        self.optimizer = torch.optim.Adam(list(self.transformer.parameters()) + list(self.qubit_embedding.parameters()),
                                          lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=LR_DECAY_STEP,
                                                         gamma=LR_DECAY)

        env_fns = [make_env(connection_graph, interaction_generator) for _ in range(N_ENVS)]
        self.vec_env = SyncVectorEnv(env_fns)

        print('Init successful')

    def get_action(self, node_embedding) -> (Tensor, Tensor, Tensor):
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

        return action, log_prob, swap_scores

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
            # action, log_prob, _ = self.get_action(state_embedding.unsqueeze(0))
            action, log_prob, _, _ = self.transformer(state_embedding.unsqueeze(0))
            observation, reward, done, truncated, info = self.env.step(action)

            self.replay_buffer.store(old_state, action, reward, deepcopy(self.env._state), done)

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
        # mcts_input = self.replay_buffer.sample(1)
        # mcts_input = self.env._state
        # self.mcts.run(Node(mcts_input[0]))
        # mcts_reward = self.mcts.run_parallel(mcts_input[0])
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
        self.transformer.train()
        self.vec_env.reset()
        self.optimizer.zero_grad()

        state_embeddings = torch.stack(
            [embed_nodes(cur_env._state, self.lookahead_length, self.device, self.qubit_embedding,
                              self.embedding_dim, self.padding_dim) for cur_env in self.vec_env.envs],
            dim=0
        ).to(self.device)

        # print(f'Loop embeddings: {state_embeddings}')

        # assert state_embeddings.shape[1] is LOOKAHEAD_LENGTH

        done_list = [False] * N_ENVS
        swaps = [self.episode_limit] * N_ENVS

        for step_idx in range(self.episode_limit):
            old_states = [deepcopy(cur_env._state) for cur_env in self.vec_env.envs]

            actions, _, _, _, _ = self.transformer(state_embeddings)

            if isinstance(actions, Tensor):
                actions = actions.cpu().numpy()

            observations, rewards, terminated, truncated, info = self.vec_env.step(actions)
            done = np.logical_or(terminated, truncated)

            # print(f'env 0 state mapping: {self.vec_env.envs[0]._state.mapping}')

            # print(actions)
            # print(old_states[0].mapping)
            # print(self.vec_env.envs[0]._state.mapping)

            for i in range(N_ENVS):
                if done_list[i] or done[i]:
                    continue
                env = self.vec_env.envs[i]
                if env._state.is_legal_surpass(*env._state.interaction_circuit[env._state.position]):
                    obs, rew, ter, trun, inf = env.step(self.action_space_size)
                    done[i] = ter or trun or done[i]

            # observations, rewards, terminated, truncated, info = self.vec_env.step(np.full((N_ENVS,), self.action_space_size))
            # done = np.logical_or(terminated, truncated, done)

            for i in range(N_ENVS):
                # print(self.vec_env.envs[0]._state)
                if done_list[i]:
                    continue
                if done[i]:
                    done_list[i] = True
                    swaps[i] = step_idx
                    self.success_count += 1

            new_embeddings = torch.stack(
                [embed_nodes(env._state, self.lookahead_length, self.device, self.qubit_embedding,
                                  self.embedding_dim, self.padding_dim) for env in self.vec_env.envs],
                dim=0
            ).to(self.device)

            for i in range(N_ENVS):
                if done_list[i]:
                    # We do NOT want to run MCTS on terminal states, as it is a waste of time
                    continue
                self.replay_buffer.store(old_states[i], actions[i], None,
                                         self.vec_env.envs[i]._state, done_list[i])
            state_embeddings = new_embeddings
            # print(f'Loop embeddings: {state_embeddings}')
            # print(f'state: {self.vec_env.envs[0]._state}')

        # returns, mask = compute_returns_batched(reward_list, device=self.device)

        # Convert log_prob_list to tensor for fast on-device computation
        # for i in range(N_ENVS):
        #     log_prob_list[i] = torch.stack(log_prob_list[i])
        # log_prob_tensor = torch.nn.utils.rnn.pad_sequence(log_prob_list, batch_first=True, padding_value=0.0).to(
        #     self.device)  # (B, T)

        # Policy gradient loss:
        #  per-episode sum_t (log_prob * G_t), then mean across batch
        # note: multiply by mask to ignore padding
        # per_episode_loss = - (log_prob_tensor * returns * mask).sum(dim=1)  # (B,)
        # policy_loss = per_episode_loss.mean()

        # backprop / gradient step
        # self.scaler.scale(policy_loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()


        # print(len(self.replay_buffer))
        loss = []
        l1 = []
        l2 = []
        transformer_all_log_probs = torch.tensor([])

        if len(self.replay_buffer.buffer) > REPLAY_BUFFER_MIN_SIZE_MCTS:
            mcts_input = self.replay_buffer.sample(MCTS_PROCESSES)
            # print(f'mcts_input: {mcts_input}')

            # if self.device != torch.device('cpu'):
            # torch.save(self.transformer.state_dict(), TEMP_TRANSFORMER_DIR)
            torch.save(self.qubit_embedding.state_dict(), TEMP_EMBEDDING_DIR)

            if MCTS_PROCESSES > 1:
                self.transformer.cpu()

            mcts_actions, mcts_state_evals = self.mcts.run(mcts_input)

            if MCTS_PROCESSES > 1:
                self.transformer.to(self.device)

            # print(f'mcts_action: {mcts_actions}')
            # print(f'mcts_state_evals: {mcts_state_evals}')

            embeddings = torch.stack(
                [embed_nodes(state, self.lookahead_length, self.device, self.qubit_embedding,
                             self.embedding_dim, self.padding_dim) for state in mcts_input],
                dim=0
            ).to(self.device)

            # device = swap_scores.device
            # teacher_actions = torch.tensor(mcts_actions, dtype=torch.long, device=device)  # (B,)
            # mcts_vals = torch.tensor(mcts_state_evals, dtype=torch.float32, device=device)  # (B,)
            #
            # # Cross-entropy: higher stability and correct handling:
            # ce_loss = torch.nn.functional.cross_entropy(swap_scores, teacher_actions, reduction='none')  # (B,)
            #
            # # MSE between transformer state_values and QMCTS
            # mse_loss = torch.nn.functional.mse_loss(state_values, mcts_vals, reduction='none')  # (B,)

            # transformer_actions, transformer_log_probs, transformer_state_evals = self.get_action(embeddings)
            transformer_actions, _, transformer_state_evals, transformer_all_log_probs, state_values = self.transformer(embeddings)
            # print(f'transformer_log_probs: {transformer_all_log_probs}')
            if isinstance(mcts_actions, list):
                l1 = torch.stack([-batch_log_probs[idx] for batch_log_probs, idx in zip(transformer_all_log_probs, mcts_actions)])
            else:
                l1 = -transformer_all_log_probs[0]
            print(f'### l1: {l1}')
            mcts_state_evals = torch.tensor(mcts_state_evals, dtype=transformer_all_log_probs.dtype, device=transformer_all_log_probs.device)
            l2 = (state_values - mcts_state_evals).pow(2)
            # l2 = torch.nn.functional.mse_loss(state_values, mcts_state_evals, reduction='none')
            print(f'### l2: {l2}')
            loss = (l1 + self.loss_coefficient * l2) / self.action_space_size
            print(f'### loss: {loss}')
            mean_loss = loss.mean()
            print(f'### mean_loss: {mean_loss}')

            if isinstance(mcts_actions, list):
                for state, transformer_action, mcts_action in zip(mcts_input, transformer_actions, mcts_actions):
                    print(f'### state: {state}')
                    print(f'### transformer_action: {transformer_action}')
                    print(f'### mcts_action: {mcts_action}')
            else:
                print(f'### state: {mcts_input[0]}')
                print(f'### transformer_action: {transformer_actions[0]}')
                print(f'### mcts_action: {mcts_actions}')

            # self.scaler.scale(mean_loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

        # Return undiscounted returns per env for plotting and analysis
        # undiscounted_returns = torch.tensor([sum(r) for r in reward_list], dtype=torch.float32, device=self.device)

        # return policy_loss, swaps, undiscounted_returns
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu()
        if isinstance(l1, torch.Tensor):
            l1 = l1.detach().cpu()
        if isinstance(l2, torch.Tensor):
            l2 = l2.detach().cpu()
        return loss, swaps, transformer_all_log_probs, l1, l2

    def save_snapshot(self, idx):
        path = TEMP_MODEL_DIR.joinpath(f"model_temp_{idx}.pt")
        torch.save(self.state_dict(), path)

        print(f"Model snapshot saved at {path}")

        # buffer_path = TEMP_BUFFER_DIR.joinpath(f"buffer_{idx}.pt")
        # self.replay_buffer.save_to_disk(buffer_path)

    # def _has_passed_recurring_threshold(total_iterations, iteration, number_of_thresholds):
    #     return (iteration * number_of_thresholds / total_iterations)

    def learn(self, total_timesteps: int, ma_window: int = 50, batch: bool = True) -> None:
        """
        Conduct the learning process of the agent
        :param total_timesteps: Number of episodes during the learning process
        :param ma_window: The size of the moving average window used for filtering out noise in the graphs
        :batch: If true - computes in batches with B=N_ENV
        """
        if MCTS_PROCESSES > 1:
            print("Parallel MCTS Init")
            print(self.qubit_embedding.weight)
            self.qubit_embedding.cpu()
            self.mcts = ParallelMCTS(self.transformer, self.qubit_embedding, self.device,
                                     swap_options=self.action_space_size, lookahead_length=self.lookahead_length,
                                     embedding_dim=self.embedding_dim, padding_dim=self.padding_dim, rollouts=200,
                                     processes=MCTS_PROCESSES, merge_worker_trees=False)
            print("Parallel MCTS Init successful")
            self.qubit_embedding.to(self.device)
            print(self.qubit_embedding.weight)
        else:
            print("MCTS Init")
            self.mcts = MCTS(self.transformer, self.qubit_embedding, self.device,
                                     swap_options=self.action_space_size, lookahead_length=self.lookahead_length,
                                     embedding_dim=self.embedding_dim, padding_dim=self.padding_dim, rollouts=200)
            print("MCTS Init successful")

        if total_timesteps % N_ENVS != 0:
            total_timesteps -= (total_timesteps % N_ENVS)
        losses = []
        swaps = []
        returns = []
        losses_l1 = []
        losses_l2 = []
        snapshot = 0

        start = time.time()

        if not batch:
            for i in range(int(total_timesteps)):
                print(f"Episode: {i}...")
                loss, swap, ep_return = self.episode()
                swaps.append(swap)
                returns.append(ep_return)

                if i != 0 and i % 512 == 0:
                    self.save_snapshot(snapshot)
                    snapshot += 1
        else:
            for i in range(int(total_timesteps // N_ENVS)):
                print(f"Episode: {i * N_ENVS}...")
                loss, ep_swaps, ep_return, l1, l2 = self.batch()
                swaps.extend(ep_swaps)
                losses.extend(loss)
                losses_l1.extend(l1)
                losses_l2.extend(l2)
                returns.extend(ep_return.detach().cpu())

                self.scheduler.step()

                if i != 0 and (i * N_ENVS) % 512 == 0:
                    self.save_snapshot(snapshot)
                    snapshot += 1

        if MCTS_PROCESSES > 1:
            self.mcts.close_workers()

        end = time.time()

        ma_swaps = moving_average(swaps, window=ma_window)
        ma_loss = moving_average(losses, window=ma_window)
        ma_loss_l1 = moving_average(losses_l1, window=ma_window)
        ma_loss_l2 = moving_average(losses_l2, window=ma_window)
        # ma_returns = moving_average(returns, window=ma_window)

        print(f"Success count: {self.success_count}")
        print(f"Success %: {self.success_count / total_timesteps * 100}")

        path = Path(LOG_DIR.joinpath(get_timestamp() + '.txt'))
        with path.open('w', encoding='utf-8') as f:
            f.write(f'episodes: {total_timesteps}\n')
            f.write(f'learning rate: {LEARNING_RATE}\n')
            f.write(f'weight decay: {WEIGHT_DECAY}\n')
            f.write(f'lr_decay: {LR_DECAY}\n')
            f.write(f'lr_decay_step: {LR_DECAY_STEP}\n')
            f.write(f'loss_coefficient: {LOSS_COEFFICIENT}\n')
            f.write(f'max_grad_norm: {MAX_GRAD_NORM}\n')
            f.write(f'N_ENVS: {N_ENVS}\n')
            f.write(f'N_MCTS_WORKERS: {len(self.mcts.workers)}\n')
            f.write(f'ma_swaps: {ma_swaps}\n')
            f.write(f'ma_loss: {ma_loss}\n')
            f.write(f'ma_loss_l1: {ma_loss_l1}\n')
            f.write(f'ma_loss_l2: {ma_loss_l2}\n')
            f.write(f"Success count: {self.success_count}\n")
            f.write(f"Success %: {self.success_count / total_timesteps * 100}\n")
            f.write(f'The training took: {end - start} seconds\n')

        episodes = range(1, total_timesteps + 1)
        mcts_games = range(1, len(losses) + 1)

        plot(episodes, swaps, label='Swaps per Episode', xlabel='Episode', ylabel='Swaps', title='Swaps per Episode')
        plot(mcts_games, losses, label='Loss per MCTS games', xlabel='Game', ylabel='Loss', title='Loss per MCTS Games')
        plot(mcts_games, losses_l1, label='Loss per MCTS games', xlabel='Game', ylabel='Loss', title='Loss L1 per MCTS Games')
        plot(mcts_games, losses_l2, label='Loss per MCTS games', xlabel='Game', ylabel='Loss', title='Loss L2 per MCTS Games')
        # plot(episodes, returns, label=f'Returns per episode', xlabel='Episode', ylabel='Return', title='Returns per episode')
        plot(episodes, ma_swaps, label=f'{ma_window}-Episode Moving Avg', xlabel='Episode', ylabel='Swaps', title='Swaps per episode 50-Episode MA')
        plot(mcts_games, ma_loss, label=f'{ma_window}-Episode Moving Avg', xlabel='Game', ylabel='Loss', title='Loss per MCTS games 50-Episode MA')
        plot(mcts_games, ma_loss_l1, label=f'{ma_window}-Episode Moving Avg', xlabel='Game', ylabel='Loss', title='Loss L1 per MCTS games 50-Episode MA')
        plot(mcts_games, ma_loss_l2, label=f'{ma_window}-Episode Moving Avg', xlabel='Game', ylabel='Loss', title='Loss L2 per MCTS games 50-Episode MA')
        # plot(episodes, ma_returns, label=f'{ma_window}-Episode Moving Avg', xlabel='Episode', ylabel='Return', title='Return per episode 50-Episode MA')

    def forward(self, env: Routing, plot: bool = False):
        state: RoutingState = env._state
        assert len(state.connection_graph.edges) == self.action_space_size

        self.qubit_embedding.to(torch.device('cpu'))

        actions = []

        # print(f'mappping: {env._state.mapping}')
        # print(f'interaction circuit: {env._state.interaction_circuit}')

        for i in range(self.episode_limit):
            if plot:
                plot_mapping_and_circuit(env._state.connection_graph, env._state.mapping, env._state.interaction_circuit)
                time.sleep(0.1)
            node_embedding = embed_nodes(env._state, self.lookahead_length, self.device, self.qubit_embedding,
                        self.embedding_dim, self.padding_dim)
            node_embedding = node_embedding.unsqueeze(0)  # now shape (1, T, E)
            transformer_action, _, _, _, _ = self.transformer(node_embedding)

            if isinstance(transformer_action, Tensor):
                transformer_action = transformer_action.cpu().item()

            observation, reward, done, truncated, info = env.step(transformer_action)
            actions.append(transformer_action)

            # print(f'action: {transformer_action}')
            # print(f'mapping: {env._state.mapping}')
            # print(f'env._state.interaction_circuit[env._state.position]: {env._state.interaction_circuit[env._state.position]}')
            # print(f'is_legal_surpass: {env._state.is_legal_surpass(*env._state.interaction_circuit[env._state.position])}')

            if env._state.is_legal_surpass(*env._state.interaction_circuit[env._state.position]):
                observation, reward, done, truncated, info = env.step(self.action_space_size)
                if plot:
                    plot_mapping_and_circuit(env._state.connection_graph, env._state.mapping,
                                         env._state.interaction_circuit)
                # print('gate ahead, cap\'n!')

            if done:
                return actions

        print(f'Failed to schedule a correct circuit within the action limit.', file=sys.stderr)

        return actions


if __name__ == '__main__':
    print("Hello world")

    boost_process_priority()

    connection_graph = nx.Graph()
    # connection_graph.add_edges_from([(0,1), (1,2), (2,3)])
    # connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,0)])

    connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,6), (6,7)])

    # connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (5,6), (6,7), (7,8), (8,9), (10,11),(11,12),(12,13), (13,14), (0,5), (5, 10), (1, 6), (6,11), (2,7), (7,12), (3,8), (8,13), (4,9), (9,14)])
    # width = 5
    nx.draw(connection_graph, with_labels=True)
    plt.show()

    print(f"cuda version: {torch.version.cuda}")  # e.g. "11.8"
    print(f"cuda availability: {torch.cuda.is_available()}")
    print(f"device count: {torch.cuda.device_count()}")  # number of GPUs
    print(f"devices: {torch.cuda.get_device_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"device: {device}")
    path = MODEL_DIR.joinpath(f"model_{get_timestamp()}.pt")
    print(path)

    model = AlphaRouterAgent(connection_graph=connection_graph, interaction_generator=None,
                             device=device, episode_limit=150, loss_coefficient=LOSS_COEFFICIENT).to(device)
    # model = AlphaRouterAgent(connection_graph=connection_graph, interaction_generator=CustomInteractionGenerator(),
    #                          device=device, episode_limit=60, loss_coefficient=LOSS_COEFFICIENT,
    #                          num_heads=2, num_layers=2, d_model=32, ffn_dim=8).to(device)
    # model.load_state_dict(torch.load(path, weights_only=True, map_location=device))

    print(f'Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    start = time.time()
    model.learn(1024, batch=True)
    end = time.time()
    print(f'It took: {end - start} seconds')

    has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.transformer.parameters())
    print("Gradients flowing:", has_grad)

    print('Successful execution')

    # model.eval()
    #
    # for i in range(100):
    #     env.reset()
    #     actions = model(env)
    #     print('### ### ###')
    #     print(f'len(actions): {len(actions)}')
    #     print(f'actions: {actions}')
    #
    # print('Successful inference')

    torch.save(model.state_dict(), path)
    # print(f"Model saved: {model.state_dict()}")
    print('Model saved successfully')

    env = Routing(
        max_observation_reach=LOOKAHEAD_LENGTH,
        observe_legal_surpasses=True,
        observe_connection_graph=True,
        connection_graph=connection_graph,
        render_mode="rgb_array",
        rewarder=None,
        interaction_generator=CustomInteractionGenerator()
    )

    ### LOAD A SPECIFIC MODEL INSTEAD OF THE LAST ONE???
    # path = MODEL_DIR.joinpath(f"model_1028_far_gate_patched_qgym_paper_hyper_2-2025-12-04T11-27-24-601885.pt")

    inference_model = AlphaRouterAgent(connection_graph=connection_graph, device=device, episode_limit=90).to(device)
    # inference_model = AlphaRouterAgent(connection_graph=connection_graph, device=device, episode_limit=90,
    #                                    num_heads=2, num_layers=2, d_model=32, ffn_dim=8).to(device)
    inference_model.load_state_dict(torch.load(path, weights_only=True))
    inference_model.eval()
    # inference_model.train()
    print('Model loaded successfully')
    # print(f"Model loaded: {inference_model.state_dict()}")

    for i in range(10):
        env.reset()
        actions = inference_model(env, plot=False)
        print('### ### ###')
        print(f'len(actions): {len(actions)}')
        print(f'actions: {actions}')

    print('Successful inference')
