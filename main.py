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

def render_rgb(env):
    """Convenience method that we will use later on to display our results. Copied from notebooks"""
    plt.figure(figsize=(40, 20))
    plt.title(f"Step {env._state.steps_done}", fontsize=40)
    plt.imshow(env.render())
    plt.axis("off")
    plt.show()

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
    # model.learn(1e2)

    # model = PPO("MultiInputPolicy", env, verbose=1)
    # model.learn(int(1e3))
    #
    # print(model.get_parameters())

    print('Successful execution')
