import networkx as nx
from gymnasium import Env
from numpy import ndarray
# import qgym
from qgym.envs import InitialMapping, Routing
from qgym.envs.routing import BasicRewarder
from qgym.generators import BasicGraphGenerator
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import random
from time import sleep

def render_rgb(env):
    """Convenience method that we will use later on to display our results."""
    plt.figure(figsize=(40, 20))
    plt.title(f"Step {env._state.steps_done}", fontsize=40)
    plt.imshow(env.render())
    plt.axis("off")
    plt.show()

class AlphaRouterAgent:
    def __init__(self, env: Routing):
        self.env = env
        self.env.rewarder = BasicRewarder(illegal_action_penalty=0, penalty_per_swap=-1, reward_per_surpass=0)
        self.reward = 0
        self.episode_limit = 100
        self.success_count = 0

    def learn(self, total_timesteps: int):
        for i in range(int(total_timesteps)):
            self.episode()

        print(f"Success count: {self.success_count}")
        print(f"Success %: {self.success_count / total_timesteps * 100}")

    def episode(self):
        self.env.reset()
        for _ in range(self.episode_limit):
            action = self.get_action(self.env)
            observation, reward, done, truncated, info = self.env.step(action)
            # render_rgb(self.env)
            # self.env.render()
            # sleep(1)
            if done:
                self.success_count += 1
                break

    def get_action(self, env: Routing) -> int:
        return env.action_space.sample()

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
    model.learn(1e3)

    # model = PPO("MultiInputPolicy", env, verbose=1)
    # model.learn(int(1e3))
    #
    # print(model.get_parameters())

    print('Successful execution')
