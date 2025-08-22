import math
from copy import deepcopy
from typing import Optional

import torch
from qgym.envs import Routing
from qgym.envs.routing import RoutingState
from torch import Tensor

from transformer import Transformer


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
    def __init__(self, root: Node, env: Routing, transformer: Transformer, swap_options: int, c: float = 0.2):
        self.root = root
        self.transposition_table = dict()
        self.c_constant = c
        self.swap_options = swap_options
        self.env = env
        self.transformer = transformer

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

    def simulate(self, node: Node):
        pass

    def backpropagate(self, path_states: list[RoutingState], path_actions: list[int], reward: float):
        pass

    def run(self, root: Node):
        leaf_node, is_terminal, path_states, path_actions = self.select(root)
        if is_terminal:
            self.backpropagate(path_states, path_actions, leaf_node.value)
            return

        node, action, log_prob = self.expand(leaf_node)
        path_states.append(node.state)
        path_actions.append(action)



        # self.backpropagate(path_states, path_actions, reward)
