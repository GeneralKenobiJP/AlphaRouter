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
from utils import embed_nodes, compute_returns, boost_process_priority, compare_routing_states

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

    def __hash__(self):
        node_hash = getattr(self, "hash", None)
        if node_hash is None or node_hash == 0:
            self.hash = gen_hash(self.state)
        return self.hash

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        if self.visits != other.visits:
            return False
        if not math.isclose(self.value, other.value):
            return False
        if not self.children == other.children:
            return False
        return compare_routing_states(self.state, other.state)

    def __reduce__(self):
        # ensure unpickling calls Node(state) -> __init__ runs
        return self.__class__, (self.state,), self.__dict__


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

    def ucb(self, root: Node, node: Node) -> float:
        return (node.value / (node.visits + 1e-6) +
                self.c_constant * math.sqrt(
                    math.log(root.visits + 1e-6) / (node.visits + 1e-6)
                ))

    def select_child(self, root: Node, path_states: List[RoutingState] = None) -> (Node, int):
        max_child = max(root.children.keys(), key=lambda child: -math.inf if child.state in path_states else self.ucb(root, child))
        if max_child in path_states:
            return None, -1
        action = root.children[max_child]
        return max_child, action

    def select(self, root: Node) -> (Node, bool, list[RoutingState], list[int]):
        # print(f'worker: {self.idx} s1')
        node = root
        # print(f'worker: {self.idx} s2')
        path_states = [root.state]
        # print(f'worker: {self.idx} s3')
        path_actions = [0] # dummy 0 action to make the states dim and actions dim match (root does not need any action)

        # print(f'worker: {self.idx} s4')
        while not node.is_terminal():
            # print(f'worker: {self.idx} s5')
            if node.is_exhausted(self.swap_options):
                # print(f'worker: {self.idx} s6')
                node, action = self.select_child(node, path_states)
                if node is None or node.state in path_states:
                    # print(f'worker: {self.idx} None reported')
                    # print(f'len path_states: {len(path_states)}')
                    # print(f'len path_actions: {len(path_actions)}')
                    # print(f'swap_options: {self.swap_options}')
                    return None, True, path_states, path_actions
                # print(f'worker: {self.idx} s7')
                path_states.append(node.state)
                # print(f'worker: {self.idx} s8')
                path_actions.append(action)
                # print(f'worker: {self.idx} s9')
            else:
                # print(f'worker: {self.idx} s10')
                return node, False, path_states, path_actions
        # print(f'worker: {self.idx} s11')
        node.value = len(node.state.interaction_circuit) - len(node.state.swap_gates_inserted) - 1
        # print(f'worker: {self.idx} s12')
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
        used = set()
        for a in leaf_node.children.values():
            if isinstance(a, Tensor):
                used.add(int(a.item()))
            else:
                used.add(int(a))

        action = None
        for a in range(self.swap_options):
            if a not in used:
                action = a
                break

        # action = torch.tensor(len(leaf_node.children), dtype=int)

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
        # print(f'worker: {self.idx} a')
        leaf_node, is_terminal, path_states, path_actions = self.select(root)
        # print(f'worker: {self.idx} b')
        if leaf_node is None:
            root.value = -1
            root.visits += 1
            self.add_child(self.get_transposition(gen_hash(path_states[-1])), path_actions[-1], root.state)
            return
        if is_terminal:
            # print(f'worker: {self.idx} c')
            self.backpropagate(path_states, path_actions, leaf_node.value)
            # print(f'worker: {self.idx} d')
            return
            # returns.append(torch.tensor(leaf_node.value, device=self.device))

        # print(f'worker: {self.idx} e')
        node, action = self.expand(leaf_node)
        # print(f'worker: {self.idx} f')
        path_states.append(node.state)
        # print(f'worker: {self.idx} g')
        path_actions.append(action)
        # print(f'worker: {self.idx} h')

        reward = self.simulate(node)
        # print(f'worker: {self.idx} i')
        self.backpropagate(path_states, path_actions, reward)
        # print(f'worker: {self.idx} j')

    def run(self, root_state: RoutingState, clear_tree: bool = True) -> (int, float, dict):
        if clear_tree:
            self.transposition_table = dict()

        self.set_root(root_state)

        with torch.no_grad():
            for i in range(self.rollouts):
                print(f'worker{self.idx}: rollout {i}')
                self.rollout(self.root)

        start = time.time()

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

        return (best_action, state_evals), self.transposition_table

    def update_tree(self, master_tree: Dict):
        self.transposition_table = deepcopy(master_tree)


def run_worker(worker: MCTSWorker, inference_queue: mp.Queue, output_queue: mp.Queue,
               signal_queue: mp.Queue, result_queue: mp.Queue, tree_queue: mp.Queue = None) -> (int, int):
    print(f"Running worker {worker.idx} with pid {os.getpid()}...")
    boost_process_priority()

    worker.set_queues(inference_queue, output_queue)
    try:
        # print(f'Root state: {root_state}')
        while True:
            root_state = signal_queue.get()
            # print(f'signal received: {root_state}')
            if root_state is None:
                return
            if isinstance(root_state, Tuple) and root_state[0] is None:
                tree = root_state[1]
                worker.update_tree(tree)
                print(f'worker: {worker.idx} updates tree:')
                print(f'len {len(worker.transposition_table.keys())}')
            if isinstance(root_state, RoutingState):
                result, tree = worker.run(root_state, clear_tree=False)
                start = time.time()
                result_queue.put(result)
                end = time.time()
                print(f'worker: {worker.idx} results sending took: {end - start} seconds')
                if tree_queue is not None:
                    start = time.time()
                    tree_queue.put(tree)
                    new_tree = tree_queue.get()
                    worker.update_tree(new_tree)
                    end = time.time()
                    print(f'worker: {worker.idx} processing merge took: {end - start} seconds')
    except Exception as e:
        print(f"Worker {worker.idx} with pid {os.getpid()} died. Exception {e} caught. {e.__traceback__}")
        return


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


def inference_loop(processes: int, inference_queue: mp.Queue, output_queues: List[mp.Queue], transformer: Transformer):
    awaiting_states: List[Tuple[int, Embedding]] = []
    output_queues_states: List[bool] = [False] * len(output_queues)
    running_processes_num = 0

    boost_process_priority()

    while True:
        while len(awaiting_states) < running_processes_num or (running_processes_num == 0 and len(awaiting_states) == 0):
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
                output_queues_states[idx_sender] = True
                running_processes_num += 1
            output_queues[idx_sender].put((detach(actions[idx].cpu()), detach(swap_scores[idx][actions[idx]].cpu())))

        awaiting_states.clear()

def merge_trees(tree_queues: List[mp.Queue]):
    main_tree = dict()

    while True:
        trees = []
        for queue in tree_queues:
            trees.append(queue.get())
            if trees[-1] is None:
                return

        start = time.time()

        new_dict = deepcopy(main_tree)

        for tree in trees:
            for hash, node in tree.items():
                if hash in main_tree and compare_routing_states(node.state, main_tree[hash].state):
                    new_dict[hash].value += node.value - main_tree[hash].value
                    new_dict[hash].visits += node.visits - main_tree[hash].visits
                elif hash in new_dict and compare_routing_states(node.state, new_dict[hash].state):
                    new_dict[hash].value += node.value
                    new_dict[hash].visits += node.visits
                else:
                    new_dict[hash] = deepcopy(node)

        for queue in tree_queues:
            queue.put(new_dict)

        end = time.time()
        print(f'Merge trees took: {end - start} seconds')


class ParallelMCTS:
    def __init__(self, transformer: Transformer, qubit_embedding: nn.Embedding,
                 device: Device, swap_options: int, c: float = math.sqrt(2), rollouts: int = 200,
                 lookahead_length: int = 48, embedding_dim: int = 24, padding_dim: int = 3, processes: int = 0,
                 merge_worker_trees: bool = False):
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
        self.inference_queue = self.manager.Queue()
        self.output_queues = [self.manager.Queue() for _ in range(self.processes)]
        self.signal_queues = [self.manager.Queue() for _ in range(self.processes)]
        self.result_queues = [self.manager.Queue() for _ in range(self.processes)]
        self.workers: List[MCTSWorker] = [
            MCTSWorker(i, self.qubit_embedding, self.device, self.swap_options, self.c_constant,
                       self.rollouts, self.lookahead_length, self.embedding_dim, self.padding_dim)

            for i in range(processes)
        ]
        if merge_worker_trees:
            self.tree_queues = [self.manager.Queue() for _ in range(self.processes)]
            self.pool = ProcessPoolExecutor(max_workers=self.processes + 1)

            args = list(zip(self.workers, [self.inference_queue] * self.processes, self.output_queues,
                            self.signal_queues, self.result_queues, self.tree_queues))
            [self.pool.submit(run_worker, *a) for a in args]
            self.pool.submit(merge_trees, *self.tree_queues)
        else:
            self.pool = ProcessPoolExecutor(max_workers=self.processes)

            args = list(zip(self.workers, [self.inference_queue] * self.processes, self.output_queues,
                            self.signal_queues, self.result_queues, [None for _ in range(self.processes)]))
            [self.pool.submit(run_worker, *a) for a in args]

    def run_parallel(self, root_states: List[RoutingState]) -> (List[int], List[float]):
        self.transformer.eval()

        while self.inference_queue.qsize() > 0:
            self.inference_queue.get()

        queue_loop = Process(target=inference_loop,
                             args=(self.processes, self.inference_queue, self.output_queues, self.transformer))
        queue_loop.start()

        for i in range(self.processes):
            self.signal_queues[i].put(root_states[i])

        evaluated_roots = []
        for i in range(self.processes):
            evaluated_roots.append(self.result_queues[i].get())

        start_huge = time.time()

        # Terminate the inference queue loop
        if queue_loop.is_alive():
            self.inference_queue.put(None)
            queue_loop.join()

        actions = [a for a, _ in evaluated_roots]
        state_values = [v for _, v in evaluated_roots]

        end_huge = time.time()
        print(f'Huge took: {end_huge - start_huge} seconds')

        return actions, state_values

    def close_workers(self):
        for queue in self.signal_queues:
            queue.put(None)
        for queue in self.tree_queues:
            queue.put(None)
            queue.put(None)
        self.pool.shutdown()
