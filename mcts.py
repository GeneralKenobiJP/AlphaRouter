import math
from copy import deepcopy
from typing import Optional, List

import numpy as np
import torch
from qgym.envs import Routing
from qgym.envs.routing import RoutingState
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.types import Device

from manhattan_rewarder import ManhattanRewarder
from transformer import Transformer
from utils import embed_nodes, compute_returns, compute_returns_batched


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
    def __init__(self, env: Routing, transformer: Transformer, qubit_embedding: nn.Embedding,
                 device: Device, swap_options: int, c: float = math.sqrt(2), rollouts: int = 200, max_depth: int = 100,
                 lookahead_length: int = 48, embedding_dim: int = 24, padding_dim: int = 3):
        self.root: Node | None = None
        self.roots: List[Node] | None = None
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

    def set_root(self, state: RoutingState) -> None:
        hash = gen_hash(state)
        node = self.get_transposition(hash)
        if node is None:
            self.root = Node(state)
            self.put_transposition(hash, self.root)
        else:
            self.root = node

    def set_roots(self, states: List[RoutingState]) -> None:
        for state in states:
            hash = gen_hash(state)
            node = self.get_transposition(hash)
            if node is None:
                cur = Node(state)
                self.roots.append(cur)
                self.put_transposition(hash, cur)
            else:
                self.roots.append(node)

    def ucb(self, root: Node, node: Node) -> float:
        return (node.value / (node.visits + 1e-6) +
                self.c_constant * math.sqrt(
                    math.log(root.visits + 1e-6) / (node.visits + 1e-6)
                ))

    def select_child(self, root: Node) -> (Node, int):
        max_child = max(root.children.keys(), key=lambda child: self.ucb(root, child))
        action = root.children[max_child]
        return max_child, action

    def select(self, root: Node) -> (Node, bool, list[RoutingState], list[int]):
        node = root
        path_states = [root.state]
        path_actions = [0] # dummy 0 action to make the states dim and actions dim match (root does not need any action)

        while not node.is_terminal():
            if node.is_exhausted(self.swap_options):
                node, action = self.select_child(node)
                path_states.append(node.state)
                path_actions.append(action)
            else:
                return node, False, path_states, path_actions
        return node, True, path_states, path_actions

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

    def expand(self, leaf_node: Node) -> (Node, int):
        # actions_covered = leaf_node.children.values()
        # actions_remaining = list(range(self.swap_options))
        # for action in actions_remaining:
        #     if action in actions_covered:
        #         actions_remaining.remove(action)

        # dist = torch.distributions.Uniform(0, self.swap_options - len(actions_covered))
        # action = torch.tensor(actions_rmaining[dist.sample()])
        # log_prob: Tensor = dist.log_prob(action)

        # dist = torch.distributions.Uniform(len(leaf_node.children.values()), self.swap_options)
        # action = dist.sample().int()
        action = torch.tensor(len(leaf_node.children.values()), dtype=int)
        # log_prob: Tensor = dist.log_prob(action)

        state = deepcopy(leaf_node.state)
        state.update_state(action.item())

        node = self.add_child(leaf_node, action, state)

        return node, action

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

    def simulate(self, node: Node) -> Tensor:
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

        returns = compute_returns(rewards)
        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns):
            policy_loss.append(-log_prob * G_t)
        policy_loss = torch.stack(policy_loss).sum()

        return policy_loss

    def simulate_batch(self, nodes: List[Node]) -> Tensor:
        sim_states = [deepcopy(node.state) for node in nodes]

        log_probs = []
        rewards: List[List[float]] = [[] for _ in range(len(sim_states))]

        with torch.no_grad():

            for step in range(self.max_depth):
                state_embeddings = [embed_nodes(node.state, self.lookahead_length, self.device, self.qubit_embedding,
                                        self.embedding_dim, self.padding_dim) for node in nodes]
                actions, log_prob = self.get_action(state_embeddings)

                old_states = [deepcopy(state) for state in sim_states]

                for i, state, old_state in enumerate(zip(sim_states, old_states)):
                    state.update_state(actions[i])
                    reward = self.rewarder.compute_reward(old_state=old_state, action=actions[i], new_state=state)

                log_probs[i].append(log_prob)
                rewards[i].append(reward)

            returns, mask = compute_returns_batched(rewards, device=self.device)

            # Convert log_prob_list to tensor for fast on-device computation
            for i in range(len(sim_states)):
                log_probs[i] = torch.stack(log_probs[i])
            log_prob_tensor = torch.nn.utils.rnn.pad_sequence(log_probs, batch_first=True, padding_value=0.0).to(
                self.device)  # (B, T)

            # Policy gradient loss:
            #  per-episode sum_t (log_prob * G_t), then mean across batch
            # note: multiply by mask to ignore padding
            per_episode_loss = - (log_prob_tensor * returns * mask).sum(dim=1)  # (B,)
            policy_loss = per_episode_loss.mean()

            return policy_loss

    def backpropagate(self, path_states: list[RoutingState], path_actions: list[int], reward: Tensor):
        path_states.reverse()
        path_actions.reverse()
        for (state, action) in zip(path_states, path_actions):
            node = self.get_transposition(gen_hash(state))
            node.visits += 1
            node.value += reward.item()

    def run(self, root_state: RoutingState) -> int:
        self.set_root(root_state)

        for idx in range(self.rollouts):
            leaf_node, is_terminal, path_states, path_actions = self.select(self.root)
            if is_terminal:
                self.backpropagate(path_states, path_actions, leaf_node.value)
                # returns.append(torch.tensor(leaf_node.value, device=self.device))

            node, action = self.expand(leaf_node)
            path_states.append(node.state)
            path_actions.append(action)

            reward = self.simulate(node)

            self.backpropagate(path_states, path_actions, reward)

            # returns.append(reward)

        return self.root.children[max(self.root.children.keys(), key=lambda child: child.value)]

    # def run_batch(self, batch: List[RoutingState]) -> int:
    #     self.set_roots(batch)
    #
    #     ### todo: COMPLETE THIS
    #     # todo: MULTITHREADING POOL (do 4 func variations prolly)
    #
    #     for idx in range(self.rollouts):
    #         leaf_node, is_terminal, path_states, path_actions = self.select(self.root)
    #         if is_terminal:
    #             self.backpropagate(path_states, path_actions, leaf_node.value)
    #             # returns.append(torch.tensor(leaf_node.value, device=self.device))
    #
    #         node, action = self.expand(leaf_node)
    #         path_states.append(node.state)
    #         path_actions.append(action)
    #
    #         reward = self.simulate(node)
    #
    #         self.backpropagate(path_states, path_actions, reward)
    #
    #         # returns.append(reward)
    #
    #     return self.root.children[max(self.root.children.keys(), key=lambda child: child.value)]
