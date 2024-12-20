import networkx as nx
from gymnasium import Env
# import qgym
from qgym.envs import InitialMapping, Routing
from qgym.envs.routing import BasicRewarder
from qgym.generators import BasicGraphGenerator
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import random

class AlphaRouterAgent:
    def __init__(self, env: Routing):
        self.env = env
        self.env.rewarder = BasicRewarder(illegal_action_penalty=0, penalty_per_swap=-1, reward_per_surpass=0)
        self.reward = 0
    def learn(self, total_timesteps: int):
        for i in range(total_timesteps):
            self.env.step(self.swap())

    def swap(self):
        action = self.env.action_space.sample()
        return action

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
    )
    action_space = env.action_space
    observation_space = env.observation_space
    # print(action_space)
    # print(observation_space)

    # model = PPO("MultiInputPolicy", env, verbose=1)
    # model.learn(int(1e3))
    #
    # print(model.get_parameters())

    print('Successful execution')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
