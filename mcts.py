import hashlib
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from queue import Queue
from typing import Optional, List, Dict, Tuple

import torch
from qgym.envs.routing import RoutingState, BasicRewarder
from torch import Tensor, nn, detach, NoneType
from torch.amp import GradScaler, autocast
from torch.nn import Embedding
from torch.types import Device
from tqdm import tqdm

from manhattan_rewarder import ManhattanRewarder
from plotting import plot_mcts_tree
from transformer import Transformer, choose_actions
from utils import embed_nodes, boost_process_priority, compare_routing_states, \
    surpass_all_possible_gates

mp.set_start_method("spawn", force=True)

MODEL_DIR = Path(__file__).resolve().parent.joinpath('models/')
TEMP_MODEL_DIR = MODEL_DIR.joinpath('temp/')
TEMP_TRANSFORMER_DIR = TEMP_MODEL_DIR.joinpath('temp_transformer.pt')
TEMP_EMBEDDING_DIR = TEMP_MODEL_DIR.joinpath('temp_embedding.pt')

CHECKPOINT_PERIOD = 4


def gen_hash(state: RoutingState):
    m = hashlib.sha256(usedforsecurity=False)
    m.update(str(state.n_qubits).encode())
    m.update(str(state.n_connections).encode())
    m.update(state.interaction_circuit[state.position:].data.tobytes())
    m.update(state.mapping.data.tobytes())
    return int(m.hexdigest(), 16)

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
        # if self.visits != other.visits:
        #     return False
        # if not math.isclose(self.value, other.value):
        #     return False
        # if not self.children == other.children:
        #     return False
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
        self.rewarder = BasicRewarder(illegal_action_penalty=0, penalty_per_swap=-1, reward_per_surpass=1)
        self.checkpoint_period = CHECKPOINT_PERIOD
        self.partial_rewarder = ManhattanRewarder(self.swap_options + 1)

        self.qubit_embedding.to(self.device)
        self.transformer.to(self.device)

    def set_root(self, state: RoutingState) -> None:
        surpass_all_possible_gates(state)
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
                    math.log(root.visits + 1 + 1e-6) / (node.visits + 1e-6)
                ))

    def select_child(self, root: Node, path_states: List[RoutingState] = None) -> (Node, int):
        max_child = max(root.children.keys(),
                        key=lambda child: -math.inf if child.state in path_states else self.ucb(root, child))
        action = root.children[max_child]
        return max_child, action

    def select(self, root: Node) -> (Node, bool, list[RoutingState], list[int]):
        node = root
        path_states = [root.state]
        path_actions = [0]  # dummy 0 action to make the states dim and actions dim match (root does not need any action)

        while not node.is_terminal():
            if node.is_exhausted(self.swap_options):
                node, action = self.select_child(node, path_states)
                # Guard against loops
                if node.state in path_states:
                    return None, False, path_states, path_actions
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
        surpass_all_possible_gates(child_state)
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

        embedding = embed_nodes(state, self.lookahead_length, self.device, self.qubit_embedding,
                                       self.embedding_dim, self.padding_dim).to(self.device)
        _, state_eval = self.transformer(embedding.unsqueeze(0))


        return state_eval

    def backpropagate(self, path_states: list[RoutingState], path_actions: list[int], reward: Tensor):
        path_states.reverse()
        path_actions.reverse()
        if reward is None:
            # Terminal node - the reward is known
            reward = self.rewarder.compute_reward(old_state=path_states[1], new_state=path_states[0], action=path_actions[0])

        path_states.pop(0)
        path_actions.pop(0)

        if isinstance(reward, Tensor):
            returns = reward.item()
        else:
            returns = reward

        for i, (state, action) in enumerate(zip(path_states, path_actions)):
            node = self.get_transposition(gen_hash(state))
            node.visits += 1

            node.value += returns

            if i < len(path_states) - 1:
                returns += self.rewarder.compute_reward(old_state=path_states[i+1], new_state=path_states[i], action=action)

    def rollout(self, root: Node):
        leaf_node, is_terminal, path_states, path_actions = self.select(root)
        if leaf_node is None:
            self.add_child(self.get_transposition(gen_hash(path_states[-1])), path_actions[-1], root.state)
            return True
        if is_terminal:
            self.backpropagate(path_states, path_actions, None)
            return False

        node, action = self.expand(leaf_node)
        path_states.append(node.state)
        path_actions.append(action)

        reward = self.simulate(node)
        self.backpropagate(path_states, path_actions, reward)
        # plot_mcts_tree(self.root, mcts=self, max_nodes=10, show=True)

        return False

    def run(self, root_state: RoutingState | List[RoutingState], clear_tree: bool = False) -> (int, float, dict):
        if clear_tree:
            self.transposition_table = dict()

        if isinstance(root_state, list):
            start = time.time()
            best_actions = []
            state_evals = []
            for i, state in enumerate(root_state):
                print(f"MCTS processing state: {i}/{len(root_state)}")
                action, eval = self.run(state)
                best_actions.append(action)
                state_evals.append(eval)
            end = time.time()
            print(f"MCTS rollouts total time: {end - start}")

            return best_actions, state_evals

        self.set_root(root_state)

        with torch.no_grad():
            for i in range(self.rollouts):
                loop = self.rollout(self.root)
                if loop:
                    print(f'Loop detected. Rollouts truncated')
                    break

        # Should NOT happen
        if len(self.root.children) == 0:
            return 0, 1, self.transposition_table

        best_action = self.root.children[
            max(self.root.children.keys(), key=lambda child: child.value / (child.visits + 1e-6))]
        if isinstance(best_action, Tensor):
            best_action = best_action.item()

        # state_evals = [0.0] * self.swap_options
        # # print(f'len self.root.children.items() = {len(self.root.children.items())}')
        # for child_node, action in self.root.children.items():  # child_node is Node, action is int or Tensor
        #     act_idx = int(action.item()) if isinstance(action, torch.Tensor) else int(action)
        #     # print(f'child node value: {child_node.value}')
        #     state_evals[act_idx] = child_node.value / (child_node.visits + 1e-6)

        state_eval = self.root.value / (self.root.visits + 1e-6)

        ###
        # DEBUG
        # print(f'state evals: {state_evals}')
        # plot_mcts_tree(self.root, mcts=self, max_nodes=10, show=True)
        ###

        return best_action, state_eval

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
        self.rewarder = BasicRewarder(illegal_action_penalty=0, penalty_per_swap=-1, reward_per_surpass=1)
        self.inference_queue: mp.Queue | None = None
        self.output_queue: mp.Queue | None = None
        self.idx = idx
        self.checkpoint_period = CHECKPOINT_PERIOD
        self.partial_rewarder = ManhattanRewarder(self.swap_options + 1)

        self.qubit_embedding.to(self.device)

    def set_root(self, state: RoutingState) -> None:
        surpass_all_possible_gates(state)
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
                    math.log(root.visits + 1 + 1e-6) / (node.visits + 1e-6)
                ))

    def select_child(self, root: Node, path_states: List[RoutingState] = None) -> (Node, int):
        max_child = max(root.children.keys(), key=lambda child: -math.inf if child.state in path_states else self.ucb(root, child))
        action = root.children[max_child]
        return max_child, action

    def select(self, root: Node) -> (Node, bool, list[RoutingState], list[int]):
        node = root
        path_states = [root.state]
        path_actions = [0] # dummy 0 action to make the states dim and actions dim match (root does not need any action)

        while not node.is_terminal():
            if node.is_exhausted(self.swap_options):
                node, action = self.select_child(node, path_states)
                # Guard against loops
                if node.state in path_states:
                    return None, False, path_states, path_actions
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
        surpass_all_possible_gates(child_state)
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
        embedding = detach(embed_nodes(state, self.lookahead_length, torch.device('cpu'), self.qubit_embedding,
                                        self.embedding_dim, self.padding_dim).cpu()).share_memory_()
        # print(f'embedding before sending: {embedding}')

        self.inference_queue.put((self.idx, embedding))
        try:
            action, state_eval = self.output_queue.get()
            # print(f'action received: {action}')
            # print(f'state_eval received: {state_eval}')
        except:
            state_eval = None

        return state_eval

    def backpropagate(self, path_states: list[RoutingState], path_actions: list[int], reward: None | Tensor | int):
        path_states.reverse()
        path_actions.reverse()
        if reward is None:
            # Terminal node - the reward is known
            reward = self.rewarder.compute_reward(old_state=path_states[1], new_state=path_states[0],
                                                  action=path_actions[0])

        path_states.pop(0)
        path_actions.pop(0)

        if isinstance(reward, Tensor):
            returns = reward.item()
        else:
            returns = reward

        for i, (state, action) in enumerate(zip(path_states, path_actions)):
            node = self.get_transposition(gen_hash(state))
            node.visits += 1

            node.value += returns

            if i < len(path_states) - 1:
                returns += self.rewarder.compute_reward(old_state=path_states[i + 1], new_state=path_states[i],
                                                        action=action)

    def rollout(self, root: Node):
        leaf_node, is_terminal, path_states, path_actions = self.select(root)
        if leaf_node is None:
            self.add_child(self.get_transposition(gen_hash(path_states[-1])), path_actions[-1], root.state)
            return True
        if is_terminal:
            self.backpropagate(path_states, path_actions, None)
            return False

        node, action = self.expand(leaf_node)
        path_states.append(node.state)
        path_actions.append(action)

        reward = self.simulate(node)
        self.backpropagate(path_states, path_actions, reward)
        # plot_mcts_tree(self.root, mcts=self, max_nodes=10, show=True)

        return False

    def run(self, root_state: RoutingState, clear_tree: bool = True) -> (int, float, dict):
        if clear_tree:
            self.transposition_table = dict()

        self.qubit_embedding.load_state_dict(torch.load(TEMP_EMBEDDING_DIR, weights_only=True, map_location=self.device))

        self.set_root(root_state)

        with torch.no_grad():
            for i in range(self.rollouts):
                # print(f'worker{self.idx}: rollout {i}')
                loop = self.rollout(self.root)
                if loop:
                    print(f'Loop detected. Rollouts truncated')
                    break

        self.inference_queue.put((self.idx, None))

        # Should NOT happen
        if len(self.root.children) == 0:
            return 0, 1, self.transposition_table

        best_action = self.root.children[max(self.root.children.keys(), key=lambda child: child.value / (child.visits + 1e-6))]
        if isinstance(best_action, Tensor):
            best_action = best_action.item()

        # state_evals = [0.0] * self.swap_options
        # # print(f'len self.root.children.items() = {len(self.root.children.items())}')
        # for child_node, action in self.root.children.items():  # child_node is Node, action is int or Tensor
        #     act_idx = int(action.item()) if isinstance(action, torch.Tensor) else int(action)
        #     # print(f'child node value: {child_node.value}')
        #     state_evals[act_idx] = child_node.value / (child_node.visits + 1e-6)
        state_eval = self.root.value / (self.root.visits + 1e-6)

        ###
        # DEBUG
        # print(f'state evals: {state_evals}')
        # plot_mcts_tree(self.root, mcts=self, max_nodes=10, show=True)
        ###

        return (best_action, state_eval), self.transposition_table

    def update_tree(self, master_tree: Dict):
        self.transposition_table = deepcopy(master_tree)


def run_worker(worker: MCTSWorker, inference_queue: mp.Queue, output_queue: mp.Queue,
               signal_queue: mp.Queue, result_queue: mp.Queue, tree_queue: mp.Queue = None) -> (int, int):
    print(f"Running worker {worker.idx} with pid {os.getpid()}...")
    boost_process_priority()

    worker.set_queues(inference_queue, output_queue)
    worker.qubit_embedding.to(torch.device('cpu'))
    try:
        # print(f'Root state: {root_state}')
        while True:
            root_state = signal_queue.get()
            # print(f'signal received: {root_state}')
            if root_state is None:
                return
            if isinstance(root_state, tuple) and root_state[0] is None:
                tree = root_state[1]
                worker.update_tree(tree)
                print(f'worker: {worker.idx} updates tree:')
                print(f'len {len(worker.transposition_table.keys())}')
            if isinstance(root_state, RoutingState):
                result, tree = worker.run(root_state, clear_tree=False)
                start = time.time()
                result_queue.put(result)
                end = time.time()
                # print(f'worker: {worker.idx} results sending took: {end - start} seconds')
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

def inference_loop(processes: int, inference_queue: mp.Queue, output_queues: List[mp.Queue], transformer: Transformer):
    awaiting_states: List[Tuple[int, Embedding]] = []
    output_queues_states: List[bool] = [False] * len(output_queues)
    running_processes_num = 0

    boost_process_priority()

    # transformer = deepcopy(transformer)
    transformer.to(transformer.device)
    # transformer.load_state_dict(torch.load(TEMP_TRANSFORMER_DIR, weights_only=True, map_location=transformer.device))
    # transformer.eval()

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
                    if not output_queues_states[idx]:
                        awaiting_states.pop()
                        continue
                    output_queues_states[idx] = False
                    running_processes_num -= 1
                    # output_queues[idx].close()
                    awaiting_states.pop()
                    if running_processes_num == 0:
                        return
            except:
                if len(awaiting_states) > 0:
                    break

        embeddings = [t.to(transformer.device) for _, t in awaiting_states]
        embedding = torch.stack(embeddings, dim=0)
        # print(f'embeddings: {embeddings}')
        # print(f'embedding: {embedding}')
        # actions, log_probs = get_action(embedding, transformer, device)
        logits, state_values = transformer(embedding)
        actions, _ = choose_actions(logits)
        # print(transformer.state_dict())
        # print(f'inferred state value: {state_values}')

        for idx, (idx_sender, _) in enumerate(awaiting_states):
            if output_queues_states[idx_sender] is False:
                output_queues_states[idx_sender] = True
                running_processes_num += 1
            output_queues[idx_sender].put((detach(actions[idx].cpu()), detach(state_values[idx].cpu())))

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
        # self.rewarder = ManhattanRewarder()
        self.processes = processes
        self.manager = mp.Manager()
        self.inference_queue = self.manager.Queue()
        self.output_queues = [self.manager.Queue() for _ in range(self.processes)]
        self.signal_queues = [self.manager.Queue() for _ in range(self.processes)]
        self.result_queues = [self.manager.Queue() for _ in range(self.processes)]
        self.workers: List[MCTSWorker] = [
            MCTSWorker(i, self.qubit_embedding, torch.device('cpu'), self.swap_options, self.c_constant,
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

            self.tree_queues = []

    def run(self, root_states: List[RoutingState]) -> (List[int], List[float]):
        self.transformer.eval()

        # while self.inference_queue.qsize() > 0:
        #     self.inference_queue.get()

        # queue_loop = Process(target=inference_loop,
        #                      args=(self.processes, self.inference_queue, self.output_queues, self.transformer))
        # queue_loop.start()

        evaluated_roots = []

        start = time.time()
        for k in range((len(root_states) + self.processes - 1) // self.processes):
            print(f'MCTS loop iter: {k+1} / {(len(root_states) + self.processes - 1) // self.processes}...')
            while self.inference_queue.qsize() > 0:
                self.inference_queue.get()

            queue_loop = Process(target=inference_loop,
                                 args=(self.processes, self.inference_queue, self.output_queues, self.transformer))
            queue_loop.start()

            processes_used = self.processes

            for i in range(self.processes):
                idx = k * self.processes + i
                if idx >= len(root_states):
                    processes_used = i
                    break
                self.signal_queues[i].put(root_states[idx])

            for i in range(processes_used):
                evaluated_roots.append(self.result_queues[i].get())

            # Terminate the inference queue loop
            if queue_loop.is_alive():
                self.inference_queue.put(None)
                queue_loop.join()
        end = time.time()
        print(f'MCTS rollouts total time: {end - start} seconds')

        actions = [a for a, _ in evaluated_roots]
        # state_values = [np.mean(v) for _, v in evaluated_roots]
        state_values = [v for _, v in evaluated_roots]

        return actions, state_values

    def close_workers(self):
        for queue in self.signal_queues:
            queue.put(None)
        for queue in self.tree_queues:
            queue.put(None)
            queue.put(None)
        self.pool.shutdown()
