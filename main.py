import math
from typing import List, Set

import networkx as nx
import torch
from gymnasium import Env
from numpy import ndarray
# import qgym
from qgym.envs import InitialMapping, Routing
from qgym.envs.routing import BasicRewarder, RoutingState
from qgym.generators import BasicGraphGenerator
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import random
from time import sleep
from torch import nn

EMBEDDING_DIM = 20      # AR paper embedding dimension
LOOKAHEAD_LENGTH = 48   # AR paper lookahead length
NUM_HEADS = 6           # AR paper number of heads in multihead attention
NUM_LAYERS = 4          # AR paper number of transformer layers
FFN_DIM = 4 * 21            # Default hidden layer dimensionality of ffn in transformer

def render_rgb(env):
    """Convenience method that we will use later on to display our results. Copied from notebooks"""
    plt.figure(figsize=(40, 20))
    plt.title(f"Step {env._state.steps_done}", fontsize=40)
    plt.imshow(env.render())
    plt.axis("off")
    plt.show()


class PositionalEncoding(nn.Module):
    """
    A positional encoding module.
    Takes in node embedding and adds positional information.
    """
    def __init__(self, embedding_dim=20, lookahead_length=48):
        """
        :param embedding_dim: Embedding dimension of the node embeddings (not including depth embedding)
        :param lookahead_length: The maximal number of the gates ahead that we will analyze
        """
        super().__init__()

        final_embedding_dim = embedding_dim + 1  # Embedding dim + 1 for depth

        position = torch.arange(lookahead_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, final_embedding_dim, 2) * (-math.log(1e5) / embedding_dim))
        pe = torch.zeros(LOOKAHEAD_LENGTH, final_embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        final_embedding_dim_even = (final_embedding_dim // 2) * 2  # Rounded up to the nearest even number so that odd indices slicing works
        pe[:, 1::2] = torch.cos(position * div_term[::final_embedding_dim_even])
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the positional encoding
        :param x: input tensor
        """
        print(f'shape x: {x.shape}, shape pe: {self.pe.shape}')
        x = x + self.pe[:x.size(0)]
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
        # self.dropout1 = nn.Dropout()
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

        embedding_dim = embedding_dim + 1  # Embedding dim + 1 for depth

        if embedding_dim % num_heads != 0:
            # Increase the dimension so that it is a multiple of the number of heads
            padded_embedding_dim = (embedding_dim // num_heads + 1) * num_heads
            self.padding_dim = padded_embedding_dim - embedding_dim
            embedding_dim = padded_embedding_dim
        else:
            self.padding_dim = 0

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
        if self.padding_dim > 0:
            # print(f'pre-padding x shape: {x.shape}')
            # print(f'padding_dim: {self.padding_dim}')

            # Pad the input with 0s so that the dimensionality is a multiple of the number of heads
            x = nn.functional.pad(x, (0, self.padding_dim), value=0)

            # print(f'padded x shape: {x.shape}')
            # print(f'padded x: {x}')

        for layer in self.layers:
            x = layer(x)
        return self.decoder.forward(x)


class AlphaRouterAgent:
    """The AlphaRouter routing agent"""
    def __init__(self, env: Routing, embedding_dim: int = 20, num_heads: int = 6, ffn_dim: int = 64, num_layers: int = 4, lookahead_length: int = 48):
        """
        :param env: Particular routing environment we want to apply the agent to
        :param embedding_dim: Embedding dimension of the node embeddings (not including depth embedding)
        :param num_heads: Number of heads in a multihead attention layer
        :param ffn_dim: Dimensionality of the hidden layer in the feed forward layer
        :param num_layers: Number of transformer layers
        :param lookahead_length: The maximal number of the gates ahead that we will analyze
        """
        self.env = env
        self.env.rewarder = BasicRewarder()
        self.reward = 0
        self.episode_limit = 100  # The maximal number of steps in a single episode
        self.success_count = 0  # How many times did we achieved a successful routing during the learning process
        self.lookahead_length = lookahead_length
        state: RoutingState = self.env._state
        connection_graph = state.connection_graph
        self.qubit_embedding = nn.Embedding(num_embeddings=connection_graph.number_of_nodes(), embedding_dim=embedding_dim)
        self.transformer = Transformer(embedding_dim, num_heads, ffn_dim, num_layers, lookahead_length, swap_options=len(connection_graph.edges) + 1)
        self.softmax = nn.Softmax(dim=0)

    def learn(self, total_timesteps: int):
        """
        Conduct the learning process of the agent
        :param total_timesteps: Number of episodes during the learning process
        """
        for i in range(int(total_timesteps)):
            self.episode()

        print(f"Success count: {self.success_count}")
        print(f"Success %: {self.success_count / total_timesteps * 100}")

    def episode(self):
        """
        Run an episode, i.e. try to solve one particular circuit
        """
        # Generate a new interaction circuit
        self.env.reset()
        render_rgb(self.env)

        state: RoutingState = self.env._state
        interaction_circuit = state.interaction_circuit[:self.lookahead_length]
        print(f'interaction_circuit: {interaction_circuit}')

        for _ in range(self.episode_limit):
            # print(state.mapping)
            action = self.get_action(self.env)
            observation, reward, done, truncated, info = self.env.step(action)
            # render_rgb(self.env)
            # self.env.render()
            # sleep(1)
            if done:
                self.success_count += 1
                break

    def get_action(self, env: Routing) -> int:
        """
        Based on the current environment state, get an action suggested by the agent
        :param env: Routing environment
        :return: Action - index of the swap option (last option is no swap)
        """
        node_embedding = self.embed_nodes()
        transformer_output = self.transformer.forward(node_embedding)
        # Transform the transformer output shape from (number_of_logical_gates x number_of_swap_options) to (number of swap_options)
        # as we want to output the optimal swap action
        swap_scores = transformer_output.sum(dim=0)
        softmax_output = self.softmax(swap_scores)

        action = torch.multinomial(softmax_output, 1)

        print(f'Node embedding: {node_embedding}')
        print(f'Transformer output: {transformer_output}')
        print(f'Transformer output shape: {transformer_output.shape}')
        print(f'Swap scores: {swap_scores}')
        print(f'Swap scores shape: {swap_scores.shape}')
        print(f'Softmax output: {softmax_output}')
        print(f'Softmax output shape: {softmax_output.shape}')
        print(f'Action: {action}')
        print(f'Intified action: {int(action)}')

        return int(action)

    def embed_nodes(self):
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
        state: RoutingState = self.env._state
        interaction_circuit = state.interaction_circuit[:self.lookahead_length]  # Limit lookahead depth

        feature_matrix = []
        qubit_gate_dependencies: List[Set[int]] = []
        for _ in range(state.n_qubits):
            qubit_gate_dependencies.append(set())

        for idx, (logical_q1, logical_q2) in enumerate(interaction_circuit):
            physical_q1 = state.mapping[logical_q1]
            physical_q2 = state.mapping[logical_q2]

            embedding_q1 = self.qubit_embedding(torch.tensor(physical_q1))
            embedding_q2 = self.qubit_embedding(torch.tensor(physical_q2))

            gate_dependencies = qubit_gate_dependencies[logical_q1].union(qubit_gate_dependencies[logical_q2])
            depth = len(gate_dependencies) + 1

            gate_vector = (embedding_q1 + embedding_q2) / 2  # Mean of qubit encodings
            gate_vector = nn.functional.pad(gate_vector, (0, 1), value=depth)  # Append depth

            qubit_gate_dependencies[logical_q1].add(idx)
            qubit_gate_dependencies[logical_q2].add(idx)

            feature_matrix.append(gate_vector)

        if len(feature_matrix) == 0:
            return torch.zeros((LOOKAHEAD_LENGTH, EMBEDDING_DIM + 1))

        print(f'Qubit dependencies: {qubit_gate_dependencies}')

        return torch.stack(feature_matrix)


if __name__ == '__main__':
    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,0)])
    nx.draw(connection_graph, with_labels=True)
    plt.show()

    # env = InitialMapping(connection_graph=connection_graph, graph_generator=BasicGraphGenerator(0.6), render_mode="human")
    env = Routing(
        max_observation_reach=LOOKAHEAD_LENGTH,
        observe_legal_surpasses=True,
        observe_connection_graph=True,
        connection_graph=connection_graph,
        render_mode="rgb_array"
    )
    # action_space = env.action_space
    # observation_space = env.observation_space
    # print(action_space)
    # print(observation_space)

    model = AlphaRouterAgent(env)
    state: RoutingState = model.env._state
    print(state.interaction_circuit.shape)
    node_embedding = model.embed_nodes()
    print(node_embedding.shape)
    print(node_embedding)

    positional_encoding = PositionalEncoding()
    context_encoding = positional_encoding.forward(node_embedding)
    print(context_encoding)

    # transformer_layer = TransformerLayer((EMBEDDING_DIM + 1), 6, EMBEDDING_DIM + 1)
    # transformer_layer_output = transformer_layer.forward(context_encoding)
    # print(transformer_layer_output)

    # transformer = Transformer(EMBEDDING_DIM, NUM_HEADS, FFN_DIM, NUM_LAYERS, LOOKAHEAD_LENGTH)
    # transformer_output = transformer.forward(node_embedding)
    # print(transformer_output)

    model.learn(1e0)

    # model = PPO("MultiInputPolicy", env, verbose=1)
    # model.learn(int(1e3))
    #
    # print(model.get_parameters())

    print('Successful execution')