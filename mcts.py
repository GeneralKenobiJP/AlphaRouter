import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from queue import Queue
from typing import Optional, List, Dict, Tuple

import torch
from qgym.envs.routing import RoutingState
from torch import Tensor, nn, detach, NoneType
from torch.amp import GradScaler, autocast
from torch.nn import Embedding
from torch.types import Device
from tqdm import tqdm

from manhattan_rewarder import ManhattanRewarder
from transformer import Transformer
from utils import embed_nodes, compute_returns, boost_process_priority

mp.set_start_method("spawn", force=True)


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
    def __init__(self, transformer: Transformer, qubit_embedding: nn.Embedding,
                 device: Device, swap_options: int, c: float = math.sqrt(2), rollouts: int = 200, max_depth: int = 100,
                 lookahead_length: int = 48, embedding_dim: int = 24, padding_dim: int = 3):
        self.root: Node | None = None
        self.transposition_table = dict()
        self.c_constant = c
        self.swap_options = swap_options
        self.transformer = transformer
        self.qubit_embedding = qubit_embedding
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.device = device
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

    def backpropagate(self, path_states: list[RoutingState], path_actions: list[int], reward: Tensor):
        path_states.reverse()
        path_actions.reverse()
        for (state, action) in zip(path_states, path_actions):
            node = self.get_transposition(gen_hash(state))
            node.visits += 1
            node.value += reward.item()

    def rollout(self):
        leaf_node, is_terminal, path_states, path_actions = self.select(self.root)
        if is_terminal:
            self.backpropagate(path_states, path_actions, leaf_node.value)
            # returns.append(torch.tensor(leaf_node.value, device=self.device))

        node, action = self.expand(leaf_node)
        path_states.append(node.state)
        path_actions.append(action)

        reward = self.simulate(node)

        self.backpropagate(path_states, path_actions, reward)

    def run(self, root_state: RoutingState) -> int:
        self.set_root(root_state)

        for _ in range(self.rollouts):
            self.rollout()

        return self.root.children[max(self.root.children.keys(), key=lambda child: child.value)]


class MCTSWorker:
    def __init__(self, idx: int, qubit_embedding: nn.Embedding,
                 device: Device, swap_options: int, c: float = math.sqrt(2), rollouts: int = 200,
                 lookahead_length: int = 48, embedding_dim: int = 24, padding_dim: int = 3):
        self.root: Node | None = None
        self.transposition_table = dict()
        self.swap_options = swap_options
        self.c_constant = c
        self.qubit_embedding = qubit_embedding
        self.rollouts = rollouts
        self.device = device
        self.lookahead_length = lookahead_length
        self.embedding_dim = embedding_dim
        self.padding_dim = padding_dim
        self.rewarder = ManhattanRewarder()
        self.inference_queue: mp.Queue | None = None
        self.output_queue: mp.Queue | None = None
        self.idx = idx

    def set_root(self, state: RoutingState) -> None:
        hash = gen_hash(state)
        node = self.get_transposition(hash)
        if node is None:
            self.root = Node(state)
            self.put_transposition(hash, self.root)
        else:
            self.root = node

    def set_queues(self, inference_queue: Queue, output_queue: Queue) -> None:
        self.inference_queue = inference_queue
        self.output_queue = output_queue

    # def add_node_if_absent(self, state: RoutingState) -> Node:
    #     hash = gen_hash(state)
    #     node = self.get_transposition(hash)
    #     if node is None:
    #         cur = Node(state)
    #         self.put_transposition(hash, cur)
    #         return cur
    #     else:
    #         return node

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
        node.value = len(node.state.interaction_circuit) - len(node.state.swap_gates_inserted) - 1
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
        action = torch.tensor(len(leaf_node.children), dtype=int)

        state = deepcopy(leaf_node.state)
        if isinstance(action, Tensor):
            try:
                state.update_state(action.item())
            except Exception as e:
                print(f'Exception: {e} caught!')
        else:
            state.update_state(action)

        node = self.add_child(leaf_node, action, state)

        return node, action

    def simulate(self, node: Node) -> Tensor:
        state = node.state
        embedding = detach(embed_nodes(state, self.lookahead_length, self.device, self.qubit_embedding,
                                        self.embedding_dim, self.padding_dim).cpu()).share_memory_()

        self.inference_queue.put((self.idx, embedding))
        try:
            action, state_eval = self.output_queue.get()
        except:
            state_eval = None

        return state_eval


    # def simulate(self, node: Node) -> Tensor:
    #     state = deepcopy(node.state)
    #
    #     # log_probs = []
    #     rewards = []
    #
    #     for step in range(self.max_depth):
    #         if state.is_done():
    #             break
    #
    #         embedding = detach(embed_nodes(state, self.lookahead_length, self.device, self.qubit_embedding,
    #                                 self.embedding_dim, self.padding_dim).cpu()).share_memory_()
    #
    #         self.inference_queue.put((self.idx, embedding))
    #
    #         try:
    #             action, log_prob = self.output_queue.get()
    #         except:
    #             break
    #
    #         old_state = deepcopy(state)
    #         state.update_state(action)
    #         reward = self.rewarder.compute_reward(old_state=old_state, action=action, new_state=state)
    #
    #         # log_probs.append(log_prob)
    #         rewards.append(reward)
    #
    #     # returns = compute_returns(rewards)
    #     # policy_loss = []
    #     # for log_prob, G_t in zip(log_probs, returns):
    #     #     policy_loss.append(-log_prob * G_t)
    #     # policy_loss = torch.stack(policy_loss).sum()
    #
    #     reward =
    #
    #     return policy_loss

    def backpropagate(self, path_states: list[RoutingState], path_actions: list[int], reward: Tensor):
        path_states.reverse()
        path_actions.reverse()
        for (state, action) in zip(path_states, path_actions):
            node = self.get_transposition(gen_hash(state))
            node.visits += 1
            if isinstance(reward, Tensor):
                node.value += reward.item()
            else:
                node.value += reward

    def rollout(self, root: Node):
        leaf_node, is_terminal, path_states, path_actions = self.select(root)
        if is_terminal:
            self.backpropagate(path_states, path_actions, leaf_node.value)
            return
            # returns.append(torch.tensor(leaf_node.value, device=self.device))

        node, action = self.expand(leaf_node)
        path_states.append(node.state)
        path_actions.append(action)

        reward = self.simulate(node)
        self.backpropagate(path_states, path_actions, reward)

    def run(self, root_state: RoutingState, clear_tree: bool = True) -> (int, float, dict):
        if clear_tree:
            self.transposition_table = dict()

        self.set_root(root_state)

        with torch.no_grad():
            for i in range(self.rollouts):
                print(f'worker{self.idx}: rollout {i}')
                self.rollout(self.root)

        self.inference_queue.put((self.idx, None))

        # Should NOT happen
        if len(self.root.children) == 0:
            return 0, 1, self.transposition_table

        best_action = self.root.children[max(self.root.children.keys(), key=lambda child: child.value)]
        if isinstance(best_action, Tensor):
            best_action = best_action.item()

        node_list = list(self.root.children.keys())
        state_evals = [0] * self.swap_options

        for i in range(len(node_list)):
            node: Node = node_list[i]
            state_evals[i] = node.value

        return best_action, state_evals, self.transposition_table

    def update_tree(self, master_tree: Dict):
        self.transposition_table = deepcopy(master_tree)


def run_worker(worker: MCTSWorker, root_state: RoutingState, inference_queue: mp.Queue, output_queue: mp.Queue) -> (int, int):
    print(f"Running worker {worker.idx} with pid {os.getpid()}...")
    boost_process_priority()

    worker.set_queues(inference_queue, output_queue)
    try:
        # print(f'Root state: {root_state}')
        return worker.run(root_state, clear_tree=False)
    except Exception as e:
        print(f"Worker {worker.idx} with pid {os.getpid()} died. Exception {e} caught. {e.__traceback__}")
        return None, None, None


def get_action(node_embedding, transformer: Transformer, device: Device) -> (Tensor, Tensor):
    """
    Based on the current environment state, get an action suggested by the agent
    :param node_embedding: Node embedding of the current state, should be an output of self.embed_nodes() right before this function
    :return:
        Action - tensor of the index of the swap option (last option is no swap)
        log_prob - tensor of log probability of the action being optimal as predicted by the model
    """
    with autocast(device.type):
        transformer_output = transformer(node_embedding)
        swap_scores = transformer_output.sum(dim=1)
        softmax_output = torch.softmax(swap_scores, dim=-1)
        dist = torch.distributions.Categorical(probs=softmax_output, validate_args=False)

        action = dist.sample()
        log_prob: Tensor = dist.log_prob(action)

    return action, log_prob


def inference_loop(processes: int, inference_queue: mp.Queue, output_queues: List[mp.Queue], transformer: Transformer,
                   device: Device):
    awaiting_states: List[Tuple[int, Embedding]] = []
    output_queues_states: List[bool] = [True] * len(output_queues)
    running_processes_num = processes

    boost_process_priority()

    # misses = 0
    while True:
        while len(awaiting_states) < running_processes_num:
            try:
                awaiting_states.append(inference_queue.get(timeout=0.25))
                if len(awaiting_states) == 0:
                    continue
                if awaiting_states[-1] is None:
                    return
                if awaiting_states[-1][-1] is None:
                    idx = awaiting_states[-1][0]
                    output_queues_states[idx] = False
                    running_processes_num -= 1
                    # output_queues[idx].close()
                    awaiting_states.pop()
                    if running_processes_num == 0:
                        return
            except:
                if len(awaiting_states) > 0:
                    break

        embeddings = [t for _, t in awaiting_states]
        embedding = torch.stack(embeddings, dim=0)
        # actions, log_probs = get_action(embedding, transformer, device)
        actions, _, swap_scores, _ = transformer(embedding)

        for idx, (idx_sender, _) in enumerate(awaiting_states):
            if output_queues_states[idx_sender] is False:
                continue
            output_queues[idx_sender].put((detach(actions[idx].cpu()), detach(swap_scores[idx][actions[idx]].cpu())))

        awaiting_states.clear()

def merge_trees(main_tree: Dict, workers: List[MCTSWorker]):
    start = time.time()

    new_dict = deepcopy(main_tree)

    for worker in workers:
        for hash, node in worker.transposition_table.items():
            if hash in main_tree and node.state == main_tree[hash]:
                new_dict[hash].value += node.value
                new_dict[hash].visits += node.visits - main_tree[hash].visits
            elif hash in new_dict and node.state == new_dict[hash]:
                new_dict[hash].value += node.value
                new_dict[hash].visits += node.visits
            else:
                new_dict[hash] = node

    end = time.time()
    print(f'Merge trees took: {end - start} seconds')

    return new_dict


class ParallelMCTS:
    def __init__(self, transformer: Transformer, qubit_embedding: nn.Embedding,
                 device: Device, swap_options: int, c: float = math.sqrt(2), rollouts: int = 200,
                 lookahead_length: int = 48, embedding_dim: int = 24, padding_dim: int = 3, processes: int = 0):
        mp.set_start_method("spawn", force=True)
        self.c_constant = c
        self.swap_options = swap_options
        self.transformer = transformer
        self.transformer.share_memory()
        self.qubit_embedding = qubit_embedding
        self.rollouts = rollouts
        self.device = device
        self.scaler = GradScaler(self.device.type)
        self.lookahead_length = lookahead_length
        self.embedding_dim = embedding_dim
        self.padding_dim = padding_dim
        self.rewarder = ManhattanRewarder()
        self.processes = processes
        self.manager = mp.Manager()
        # self.inference_queue = mp.Queue()
        # self.output_queues = [mp.Queue() for _ in range(self.processes)]
        self.inference_queue = self.manager.Queue()
        self.output_queues = [self.manager.Queue() for _ in range(self.processes)]
        self.workers: List[MCTSWorker] = [
            MCTSWorker(i, self.qubit_embedding, self.device, self.swap_options, self.c_constant,
                       self.rollouts, self.lookahead_length, self.embedding_dim, self.padding_dim)

            for i in range(processes)
        ]
        self.pool = mp.Pool(processes=processes)
        self.dict = dict()
        # self.merge_process: Process = Process(target=merge_trees, args=(self.dict, self.workers))
        self.merge_pool = ProcessPoolExecutor(max_workers=1)
        self.merge_process_args = (self.dict, self.workers)
        self.merge_future = None

    # def merge_trees(self):
    #     new_dict = deepcopy(self.dict)
    #     for worker in self.workers:
    #         for hash, node in worker.transposition_table.items():
    #             if hash in self.dict and node.state == self.dict[hash]:
    #                 new_dict[hash].value += node.value
    #                 new_dict[hash].visits += node.visits - self.dict[hash].visits
    #             elif hash in new_dict and node.state == new_dict[hash]:
    #                 new_dict[hash].value += node.value
    #                 new_dict[hash].visits += node.visits
    #             else:
    #                 new_dict[hash] = node
    #
    #     self.dict = new_dict
    #
    #     for worker in self.workers:
    #         worker.update_tree(self.dict)

    # def merge_trees(self):
    #     self.merge_process = Process(target=merge_trees, args=(self.dict, self.workers))
    #     self.merge_process.start()


    def run_parallel(self, root_states: List[RoutingState]) -> (List[int], List[float]):
        # if self.merge_process.is_alive():
        #     self.merge_process.join()

        # try:
        #     for worker in self.workers:
        #         worker.update_tree(self.dict['dict'])
        # except KeyError:
        #     pass

        # if self.merge_future is not None:
        #     self.dict = self.merge_future.result()
        #     for worker in self.workers:
        #         worker.update_tree(self.dict)


        self.transformer.eval()

        # print("### MERGED ###")
        # print(f'len {len(self.dict.keys())}')
        #
        # for worker in self.workers:
        #     print("###")
        #     print(f'worker: {worker.idx}')
        #     print(f'len {len(worker.transposition_table.keys())}')


        while self.inference_queue.qsize() > 0:
            self.inference_queue.get()

        queue_loop = Process(target=inference_loop,
                             args=(self.processes, self.inference_queue, self.output_queues, self.transformer, self.device))
        queue_loop.start()

        args = list(zip(self.workers, root_states, [self.inference_queue] * self.processes, self.output_queues))

        evaluated_roots = self.pool.starmap(run_worker, args)
        # procs = []
        # results = []
        # for arg_list in args:
        #     p = mp.Process(target=run_worker, args=arg_list)
        #     p.start()
        #     procs.append(p)
        # for p in procs:
        #     p.join()

        # Terminate the inference queue loop
        if queue_loop.is_alive():
            self.inference_queue.put(None)
            queue_loop.join()

        # for idx, (_, _, tree) in enumerate(evaluated_roots):
        #     self.workers[idx].transposition_table = tree

        # print("### WORK DONE ###")
        # for worker in self.workers:
        #     print("###")
        #     print(f'worker: {worker.idx}')
        #     print(f'len {len(worker.transposition_table.keys())}')

        # self.merge_process = Process(target=merge_trees, args=(self.dict, self.workers))
        # self.merge_process.start()
        # self.merge_future = self.merge_pool.submit(merge_trees, *self.merge_process_args)

        actions = [a for a, _, _ in evaluated_roots]
        state_values = [v for _, v, _ in evaluated_roots]

        # state_values = [[state for state, _ in worker.root.children] for worker in self.workers]
        return actions, state_values
        # return [root.children[max(root.children.keys(), key=lambda child: child.value)] for root in evaluated_roots]
        # raise Exception
