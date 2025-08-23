import math
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from qgym.envs import Routing
from qgym.envs.routing import RoutingState
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.types import Device

from manhattan_rewarder import ManhattanRewarder
from transformer import Transformer
from utils import embed_nodes


def gen_hash(state: RoutingState):
    return hash( (state.interaction_circuit.data.tobytes(), state.mapping.data.tobytes()) )


class Node:
    def __init__(self, state: RoutingState):
        self.visits = 0
        self.value = 0
        self.children = dict()
        self.hash = 0
        self.state: RoutingState = state

    def is_leaf_node(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        return self.state.is_done()

    def is_exhausted(self, swap_options: int) -> bool:
        return len(self.children) == swap_options


class MCTS:
    def __init__(self, root: Node, env: Routing, transformer: Transformer, qubit_embedding: nn.Embedding,
                 device: Device, swap_options: int, c: float = math.sqrt(2), rollouts: int = 200, max_depth: int = 150,
                 lookahead_length: int = 48, embedding_dim: int = 24, padding_dim: int = 3):
        self.root = root
        self.transposition_table = dict()
        self.c_constant = c
        self.swap_options = swap_options
        self.env = env
        self.transformer = transformer
        self.qubit_embedding = qubit_embedding
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.device = device
        self.scaler = GradScaler(self.device.type)
        self.lookahead_length = lookahead_length
        self.embedding_dim = embedding_dim
        self.padding_dim = padding_dim
        self.rewarder = ManhattanRewarder()


    def ucb(self, root: Node, node: Node) -> float:
        return (node.value / (node.visits + 1e-6) +
                self.c_constant * math.sqrt(
                    math.log(root.visits + 1e-6) / (node.visits + 1e-6)
                ))

    def select_child(self, root: Node) -> (Node, int):
        max_child = max(root.children.keys(), key=lambda child: self.ucb(root, child))
        action = root.children[max_child]
        return max_child, action
        # max_value = -math.inf
        # max_idx = -1
        #
        # for (idx, child) in enumerate(root.children):
        #     ucb = self.ucb(root, child)
        #     if ucb > max_value:
        #         max_value = ucb
        #         max_idx = idx
        #
        # return root.children[max_idx]

    def select(self, root: Node) -> (Node, bool, list[RoutingState], list[int]):
        node = root
        path_states = []
        path_actions = []

        while not node.is_terminal():
            if node.is_exhausted(self.swap_options):
                node, action = self.select_child(node)
                path_states.append(node.state)
                path_actions.append(node.value)
            else:
                return node, False, path_states, path_actions
        return node, True, path_states, path_actions

    # def add_node(self, leaf_node: Node):
    def get_transposition(self, hash: int) -> Optional[Node]:
        if hash in self.transposition_table:
            return self.transposition_table[hash]
        return None

    def put_transposition(self, hash: int, node: Node) -> None:
        self.transposition_table[hash] = node

    def add_child(self, parent: Node, action: int | Tensor, child_state: RoutingState) -> Node:
        hash = gen_hash(child_state)
        node = self.get_transposition(hash)
        if node is None:
            node = Node(child_state)
            self.put_transposition(hash, node)

        parent.children[node] = action
        return node

    def expand(self, leaf_node: Node) -> (Node, int, Tensor):
        leaf_node.state.obtain_observation()
        dist = torch.distributions.Uniform(0, self.swap_options)
        action = Tensor.int(dist.sample())
        log_prob: Tensor = dist.log_prob(action)

        state = deepcopy(leaf_node.state)
        state.update_state(action.item())

        node = self.add_child(leaf_node, action, state)

        return node, action, log_prob

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
            swap_scores = transformer_output.sum(dim=1)
            softmax_output = torch.softmax(swap_scores, dim=-1)
            dist = torch.distributions.Categorical(probs=softmax_output, validate_args=False)

            action = dist.sample()
            log_prob: Tensor = dist.log_prob(action)

        return action, log_prob

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

    def simulate(self, node: Node) -> float:
        losses = []
        for idx in range(self.rollouts):
            print(idx)
            state = deepcopy(node.state)

            log_probs = []
            rewards = []

            for step in range(self.max_depth):
                if state.is_done():
                    break

                embedding = embed_nodes(state, self.lookahead_length, self.device, self.qubit_embedding,
                                        self.embedding_dim, self.padding_dim)
                action, log_prob = self.get_action(embedding)

                old_state = deepcopy(state)
                state.update_state(action)
                reward = self.rewarder.compute_reward(old_state=old_state, action=action, new_state=state)

                log_probs.append(log_prob)
                rewards.append(reward)

            returns = self.compute_returns(rewards)
            policy_loss = []
            for log_prob, G_t in zip(log_probs, returns):
                policy_loss.append(-log_prob * G_t)
            policy_loss = torch.stack(policy_loss).sum()

            # self.scaler.scale(policy_loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()

            losses.append(policy_loss)

        return np.average(losses)

    def backpropagate(self, path_states: list[RoutingState], path_actions: list[int], reward: float):
        pass

    def run(self, root: Node):
        print('a')
        leaf_node, is_terminal, path_states, path_actions = self.select(root)
        print('b')
        if is_terminal:
            self.backpropagate(path_states, path_actions, leaf_node.value)
            return

        print('c')

        node, action, log_prob = self.expand(leaf_node)
        print('d')
        path_states.append(node.state)
        print('e')
        path_actions.append(action)
        print('f')

        pred = self.simulate(node)
        print('g')

        # self.backpropagate(path_states, path_actions, reward)
