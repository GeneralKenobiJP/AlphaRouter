import math
from typing import Set

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

EMBEDDING_DIM = 20
LOOKAHEAD_LENGTH = 48
NUM_HEADS = 6
NUM_LAYERS = 4
FFN_DIM = 64

def render_rgb(env):
    """Convenience method that we will use later on to display our results. Copied from notebooks"""
    plt.figure(figsize=(40, 20))
    plt.title(f"Step {env._state.steps_done}", fontsize=40)
    plt.imshow(env.render())
    plt.axis("off")
    plt.show()


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim=20, lookahead_length=48):
        super().__init__()

        final_embedding_dim = embedding_dim + 1  # Embedding dim + 1 for depth

        position = torch.arange(lookahead_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, final_embedding_dim, 2) * (-math.log(1e5) / embedding_dim))
        pe = torch.zeros(LOOKAHEAD_LENGTH, final_embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[::(final_embedding_dim // 2) * 2])
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        print(f'shape x: {x.shape}, shape pe: {self.pe.shape}')
        x = x + self.pe[:x.size(0)]
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim):
        super().__init__()
        if embedding_dim % num_heads != 0:
            # Increase the dimension to a multiple of heads
            padded_embedding_dim = (embedding_dim // num_heads + 1) * num_heads
            self.padding_dim = padded_embedding_dim - embedding_dim
            embedding_dim = padded_embedding_dim
        else:
            self.padding_dim = 0
        self.embedding_dim = embedding_dim
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
        if self.padding_dim > 0:
            # print(f'pre-padding x shape: {x.shape}')
            # print(f'embedding_dim: {self.embedding_dim}')
            # print(f'padding_dim: {self.padding_dim}')
            x = nn.functional.pad(x, (0, self.padding_dim), value=0)
            # print(f'padded x shape: {x.shape}')
            # print(f'padded x: {x}')

        attention_output, _ = self.multihead_attention(x, x, x)
        x = self.norm1(x + attention_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, num_layers, lookahead_length):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embedding_dim, lookahead_length)

        embedding_dim = embedding_dim + 1  # Embedding dim + 1 for depth

        if embedding_dim % num_heads != 0:
            # Increase the dimension to a multiple of heads
            padded_embedding_dim = (embedding_dim // num_heads + 1) * num_heads
            self.padding_dim = padded_embedding_dim - embedding_dim
            embedding_dim = padded_embedding_dim
        else:
            self.padding_dim = 0

        self.layers = nn.ModuleList([
            TransformerLayer(embedding_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.positional_encoding(x)
        if self.padding_dim > 0:
            print(f'pre-padding x shape: {x.shape}')
            # print(f'embedding_dim: {self.embedding_dim}')
            print(f'padding_dim: {self.padding_dim}')
            x = nn.functional.pad(x, (0, self.padding_dim), value=0)
            print(f'padded x shape: {x.shape}')
            print(f'padded x: {x}')

        i=0
        for layer in self.layers:
            print(f'Iteration: {i}')
            i+=1
            x = layer(x)
        return x

class AlphaRouterAgent:
    def __init__(self, env: Routing):
        self.env = env
        # self.env.rewarder = BasicRewarder(illegal_action_penalty=0, penalty_per_swap=-1, reward_per_surpass=0)
        self.env.rewarder = BasicRewarder()
        self.reward = 0
        self.episode_limit = 100
        self.success_count = 0
        self.qubit_embedding = nn.Embedding(num_embeddings=connection_graph.number_of_nodes(), embedding_dim=EMBEDDING_DIM)

    def learn(self, total_timesteps: int):
        for i in range(int(total_timesteps)):
            self.episode()

        print(f"Success count: {self.success_count}")
        print(f"Success %: {self.success_count / total_timesteps * 100}")

    def episode(self):
        self.env.reset()
        # state: RoutingState = self.env._state
        # interaction_circuit = state.interaction_circuit
        # print(state.mapping)
        # s: Set = set()
        # for i in interaction_circuit:
        #     s.add(i[0])
        #     s.add(i[1])
        # print(s)

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
        node_embedding = self.embed_nodes()

        return env.action_space.sample()

    def embed_nodes(self):
        state: RoutingState = self.env._state
        interaction_circuit = state.interaction_circuit[:LOOKAHEAD_LENGTH]  # Limit lookahead depth

        feature_matrix = []
        for depth, (logical_q1, logical_q2) in enumerate(interaction_circuit):
            physical_q1 = state.mapping[logical_q1]
            physical_q2 = state.mapping[logical_q2]

            embedding_q1 = self.qubit_embedding(torch.tensor(physical_q1))
            embedding_q2 = self.qubit_embedding(torch.tensor(physical_q2))

            gate_vector = (embedding_q1 + embedding_q2) / 2  # Mean of qubit encodings
            gate_vector = nn.functional.pad(gate_vector, (0, 1), value=depth)  # Append depth

            feature_matrix.append(gate_vector)

        if len(feature_matrix) == 0:
            return torch.zeros((LOOKAHEAD_LENGTH, EMBEDDING_DIM + 1))

        return torch.stack(feature_matrix)

    # def predict(self, observation: ndarray | dict[str, ndarray]):
    #     return observatio

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,0)])
    nx.draw(connection_graph, with_labels=True)
    plt.show()

    # env = InitialMapping(connection_graph=connection_graph, graph_generator=BasicGraphGenerator(0.6), render_mode="human")
    env = Routing(
        max_observation_reach=48,
        observe_legal_surpasses=True,
        observe_connection_graph=True,
        connection_graph=connection_graph,
        render_mode="rgb_array"
    )
    action_space = env.action_space
    observation_space = env.observation_space
    print(action_space)
    print(observation_space)

    model = AlphaRouterAgent(env)
    state: RoutingState = model.env._state
    print(state.interaction_circuit.shape)
    node_embedding = model.embed_nodes()
    print(node_embedding.shape)
    print(node_embedding)

    positional_encoding = PositionalEncoding()
    context_encoding = positional_encoding.forward(node_embedding)
    print(context_encoding)

    transformer_layer = TransformerLayer((EMBEDDING_DIM + 1), 6, EMBEDDING_DIM + 1)
    transformer_layer_output = transformer_layer.forward(context_encoding)
    print(transformer_layer_output)

    transformer = Transformer(EMBEDDING_DIM, NUM_HEADS, FFN_DIM, NUM_LAYERS, LOOKAHEAD_LENGTH)
    transformer_output = transformer.forward(node_embedding)
    print(transformer_output)

    # model.learn(1e2)

    # model = PPO("MultiInputPolicy", env, verbose=1)
    # model.learn(int(1e3))
    #
    # print(model.get_parameters())

    print('Successful execution')
