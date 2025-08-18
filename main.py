import math
import time
from collections import deque
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
from plotting import moving_average, plot
from manhattan_rewarder import ManhattanRewarder

from gymnasium.vector import SyncVectorEnv

from utils import NormalizeInfoWrapper

EMBEDDING_DIM = 20      # AR paper embedding dimension
LOOKAHEAD_LENGTH = 48   # AR paper lookahead length
NUM_HEADS = 6           # AR paper number of heads in multihead attention
NUM_LAYERS = 4          # AR paper number of transformer layers
FFN_DIM = 4 * 21            # Default hidden layer dimensionality of ffn in transformer
N_ENVS: int = 64

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


class PositionalEncoding(nn.Module):
    """
    A positional encoding module.
    Takes in node embedding and adds positional information.
    """
    def __init__(self, embedding_dim=24, lookahead_length=48):
        """
        :param embedding_dim: Embedding dimension of the (padded) node embeddings
        :param lookahead_length: The maximal number of the gates ahead that we will analyze
        """
        super().__init__()

        position = torch.arange(lookahead_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(1e5) / embedding_dim))
        pe = torch.zeros(lookahead_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Rounded up to the nearest even number so that odd indices slicing works
        final_embedding_dim_even = embedding_dim if embedding_dim % 2 == 0 else embedding_dim - 1
        pe[:, 1::2] = torch.cos(position * div_term[::final_embedding_dim_even])
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the positional encoding
        :param x: input tensor
        """
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
        :param embedding_dim: Embedding dimension of the node embeddings (with the padding and depth)
        :param num_heads: Number of heads in a multihead attention layer
        :param ffn_dim: Dimensionality of the hidden layer in the feed forward layer
        :param num_layers: Number of transformer layers
        :param lookahead_length: The maximal number of the gates ahead that we will analyze
        :param swap_options: Number of possible swaps actions + 1 (to include no swap)
        """
        super().__init__()

        self.positional_encoding = PositionalEncoding(embedding_dim, lookahead_length)

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
        stored_state = state.detach()
        stored_action = (action.detach() if isinstance(action, torch.Tensor)
                         else action)
        stored_next_state = next_state.detach()
        self.buffer.append((stored_state, stored_action, reward, stored_next_state, done))

        # stored_state = state.detach().to(self.device)
        # stored_action = (action.detach() if isinstance(action, torch.Tensor)
        #       else torch.tensor(action, dtype=torch.int, device=self.device))
        # stored_reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        # stored_next_state = next_state.detach().to(self.device)
        # stored_done = torch.tensor(done, dtype=torch.float32, device=self.device)
        # self.buffer.append((stored_state, stored_action, stored_reward, stored_next_state, stored_done))

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

        states = torch.stack(states, dim=0).to(device)
        actions = torch.stack(actions, dim=0).to(device)
        rewards = torch.stack(rewards, dim=0).to(device)
        next_states = torch.stack(next_states, dim=0).to(device)
        dones = torch.stack(dones, dim=0).to(device)

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

        self.to(self.device)

        self.scaler = GradScaler(self.device.type)
        self.optimizer = torch.optim.Adam(list(self.transformer.parameters()) + list(self.qubit_embedding.parameters()), lr=0.001, weight_decay=0.1)

        env_fns = [make_env(connection_graph, ManhattanRewarder()) for _ in range(N_ENVS)]
        self.vec_env = SyncVectorEnv(env_fns)

    def compute_returns(self, rewards, gamma=0.99):
        """
        Given a list of rewards [r0, r1, r2, …, r_{T-1}],
        returns a list of discounted returns:
          G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + …

        Parameters
        ----------
        rewards: list of rewards collected in a given episode
        gamma: float, discount factor.
            The higher the gamma, the more important later rewards are.

        Returns
        -------
        Discounted returns.
        """
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # (Optionally) normalize returns to reduce variance:
        # improves performance
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def _pad_and_mask_rewards(self, reward_list: list[list[float]]) -> (Tensor, Tensor):
        """
        Pads the list of rewards and masks it.
        The reward list is of shape (N_ENVS, #steps),
        where #steps varies per env.
        We want to pad it with zeros so that the shape becomes
        (N_ENVS, limit of steps).

        Additionally, we want to produce a mask that will show
        which indices are just paddings, for future reference.

        Parameters
        ----------
        reward_list: list of rewards collected in a given episode
            Shape: (N_ENVS, #steps), where #steps varies per env

        Returns
        -------
        A tuple of two elements:
            - Padded reward list of shape(N_ENVS, limit of steps),
            - Padding mask with 1 for each entry being an actual reward and 0 for a padding
        """
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

    def compute_returns_batched(self, reward_list: list[list[float]], gamma=0.99) -> (Tensor, Tensor):
        """
        Given a list of rewards [r0, r1, r2, …, r_{T-1}],
        returns a list of discounted returns:
          G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + …

        Considers a full batch of rewards for multiple envs

        Parameters
        ----------
        reward_list: list of rewards collected in a given episode
            Shape: (N_ENVS, #steps), where #steps varies per env
        gamma: float, discount factor.
            The higher the gamma, the more important later rewards are.

        Returns
        -------
        A tuple of two elements:
            - Discounted returns.
            - Padding mask with 1 for each entry being an actual reward and 0 for a padding.
        """

        padded_reward_list, mask = self._pad_and_mask_rewards(reward_list)

        batch_size, length = padded_reward_list.shape
        returns = torch.zeros((batch_size, length), dtype=padded_reward_list.dtype, device=self.device)

        R = torch.zeros(batch_size, dtype=padded_reward_list.dtype, device=self.device)

        for i in range(length-1, -1, -1):
            R = padded_reward_list[:, i] + gamma * R
            R = R * mask[:, i]
            returns[:, i] = R

        # (Optionally) normalize returns to reduce variance:
        # improves performance
        valid = mask.bool()
        vals = returns[valid]
        mean = vals.mean()
        std = vals.std(unbiased=False)
        returns = (returns - mean) / (std + 1e-8)

        return returns, mask

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
        state_embedding = self.embed_nodes(self.env._state)

        for _ in range(self.episode_limit):
            action, log_prob = self.get_action(state_embedding.unsqueeze(0))
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

        returns = self.compute_returns(rewards)
        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns):
            policy_loss.append(-log_prob * G_t)
        policy_loss = torch.stack(policy_loss).sum()

        # backprop / gradient step
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

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
            [self.embed_nodes(env._state) for env in self.vec_env.envs],
            dim=0
        ).to(self.device)

        log_prob_list = [[] for _ in range(N_ENVS)]
        reward_list = [[] for _ in range(N_ENVS)]
        done_list = [False] * N_ENVS
        swaps = [self.episode_limit] * N_ENVS

        for step_idx in range(self.episode_limit):

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
                [self.embed_nodes(env._state) for env in self.vec_env.envs],
                dim=0
            ).to(self.device)

            for i in range(N_ENVS):
                self.replay_buffer.store(state_embeddings[i], actions[i], reward_list[i][-1], new_embeddings[i], done_list[i])
            state_embeddings = new_embeddings

        returns, mask = self.compute_returns_batched(reward_list)

        # Convert log_prob_list to tensor for fast on-device computation
        for i in range(N_ENVS):
            log_prob_list[i] = torch.stack(log_prob_list[i])
        log_prob_tensor = torch.nn.utils.rnn.pad_sequence(log_prob_list, batch_first=True, padding_value=0.0).to(
            self.device)  # (B, T)

        # Policy gradient loss:
        #  per-episode sum_t (log_prob * G_t), then mean across batch
        # note: multiply by mask to ignore padding
        # log_prob_list = torch.stack(log_prob_list, dim=0)
        per_episode_loss = - (log_prob_tensor * returns * mask).sum(dim=1)  # (B,)
        policy_loss = per_episode_loss.mean()

        # backprop / gradient step
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Return undiscounted returns per env for plotting and analysis
        undiscounted_returns = torch.tensor([sum(r) for r in reward_list], dtype=torch.float32, device=returns.device)

        return policy_loss, swaps, undiscounted_returns

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
        - We pad the embedding so that the length of the embedding is divisible by the number of the attention heads

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

    def learn(self, total_timesteps: int, ma_window: int = 50, batch: bool = True) -> None:
        """
        Conduct the learning process of the agent
        :param total_timesteps: Number of episodes during the learning process
        :param ma_window: The size of the moving average window used for filtering out noise in the graphs
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
    model.learn(1024)
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