import bz2
import datetime
import gzip
import lzma
from collections import deque
from pathlib import Path
from sys import platform
from typing import List, Set, Optional

import numpy as np
import torch
from gymnasium import Wrapper
from qgym.envs import Routing
from qgym.envs.routing import RoutingState
from torch import nn, Tensor
from torch.types import Device


def _make_python_safe(obj):
    """Convert common non-Python-native types into plain Python types."""
    if obj is None:
        return None
    if isinstance(obj, deque):
        return list(obj)
    if isinstance(obj, np.generic):
        return obj.item()                 # e.g. np.int64 -> int
    if isinstance(obj, np.ndarray):
        # convert arrays (any shape) into nested lists (works for empty arrays too)
        return obj.tolist()
    # try to convert lists/tuples recursively
    if isinstance(obj, (list, tuple)):
        return type(obj)(_make_python_safe(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _make_python_safe(v) for k, v in obj.items()}
    # otherwise leave as-is (strings, numbers, custom objects)
    return obj


class NormalizeInfoWrapper(Wrapper):
    """
    Ensure each step() returns info as a dict with a single key 'env_info'
    whose value is a plain Python object (lists, dicts, scalars). This forces
    VectorEnv to aggregate into an object array and avoids shape-broadcasting errors.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        safe = _make_python_safe(info)
        return obs, reward, terminated, truncated, {"env_info": safe}

    def reset(self, **kwargs):
        # gymnasium reset returns (obs, info)
        obs, info = self.env.reset(**kwargs)
        safe = _make_python_safe(info)
        return obs, {"env_info": safe}

    @property
    def _state(self):
        # delegate to inner env
        # .env should exist because Wrapper provides it
        return getattr(self.env, '_state')

    @_state.setter
    def _state(self, value):
        # delegate to inner env
        # .env should exist because Wrapper provides it
        setattr(self.env, '_state', value)


def embed_nodes(state: RoutingState, lookahead_length: int, device: Device, qubit_embedding: nn.Embedding,
                embedding_dim: int, padding_dim: int):
    """
    Return the node embedding based on the current environment.

    We observe a maximum of self.lookahead_length gates and for each gate we produce a gate embedding in the following way:
    - We retrieve logical qubits involved in the gate
    - We retrieve physical qubits mapped to the logical qubits
    - For each physical qubit we produce a qubit embedding with nn.Embedding
    - We take a mean of these two qubit embeddings
    - We concatenate this embedding with the depth of the gate, where depth is equal to the number of another gates
        this gate depends on + 1
    - We pad the embedding so that the length of the embedding is divisible by the number of the attention heads

    This procedure yields a node embedding matrix of shape (Lookahead Length x (Embedding Dimension + 1))

    :return: Node embedding
    """
    interaction_circuit = state.interaction_circuit[:lookahead_length]  # Limit lookahead depth

    # w = getattr(qubit_embedding, "weight", None)
    # print(f'qubit embedding weight: {w}')
    # if w is not None:
    #     try:
    #         print(f"[weight device: {w.device}, shape: {w.shape}, sum: {float(w.sum()):.6f}")
    #     except Exception as e:
    #         print(f"could not inspect weight: {e}")

    # Pre-generate physical qubit embeddings
    phys_qubits = torch.arange(state.n_qubits, dtype=torch.long, device=device)
    embeddings = qubit_embedding(phys_qubits)
    # print(f'Embed nodes: embeddings: {embeddings}')

    feature_matrix = []
    qubit_gate_dependencies: List[Set[int]] = []
    for _ in range(state.n_qubits):
        qubit_gate_dependencies.append(set())

    for idx, (logical_q1, logical_q2) in enumerate(interaction_circuit):
        physical_q1 = state.mapping[logical_q1]
        physical_q2 = state.mapping[logical_q2]

        embedding_q1 = embeddings[physical_q1]
        embedding_q2 = embeddings[physical_q2]

        gate_dependencies = qubit_gate_dependencies[logical_q1].union(qubit_gate_dependencies[logical_q2])
        depth = len(gate_dependencies) + 1

        gate_vector = (embedding_q1 + embedding_q2) / 2  # Mean of qubit encodings
        gate_vector = nn.functional.pad(gate_vector, (0, 1), value=depth)  # Append depth

        # Pad the input with 0s so that the dimensionality is a multiple of the number of heads
        gate_vector = nn.functional.pad(gate_vector, (0, padding_dim), value=0)  # Append padding dimension

        qubit_gate_dependencies[logical_q1].add(idx)
        qubit_gate_dependencies[logical_q2].add(idx)

        feature_matrix.append(gate_vector)

    while len(feature_matrix) < lookahead_length:
        feature_matrix.append(torch.zeros(embedding_dim + padding_dim + 1, device=device))

    return torch.stack(feature_matrix)

def boost_process_priority():
    if platform == 'Windows':
        import win32api, win32process

        h = win32api.GetCurrentProcess()
        win32process.SetPriorityClass(h, win32process.REALTIME_PRIORITY_CLASS)
        win32process.SetThreadPriority(h, win32process.THREAD_PRIORITY_TIME_CRITICAL)

def compare_routing_states(state1: RoutingState, state2: RoutingState) -> bool:
    if not (isinstance(state1, RoutingState) and isinstance(state2, RoutingState)):
        return False
    if not state1.connection_graph.number_of_nodes() == state2.connection_graph.number_of_nodes():
        return False
    if not state1.n_connections == state2.n_connections:
        return False
    if not state1.n_qubits == state2.n_qubits:
        return False
    edges1 = set(state1.connection_graph.edges)
    edges2 = set(state2.connection_graph.edges)
    if not edges1 == edges2:
        return False
    if not np.array_equal(state1.interaction_circuit[state1.position:], state2.interaction_circuit[state2.position:]):
        return False
    if not np.array_equal(state1.mapping, state2.mapping):
        return False

    return True

def surpass_gate_if_possible(state: RoutingState):
    if state.is_done():
        return False

    if state.is_legal_surpass(*state.interaction_circuit[state.position]):
        state.update_state(state.n_connections)

        return True
    return False

def surpass_all_possible_gates(state: RoutingState):
    while surpass_gate_if_possible(state):
        pass

def env_surpass_all_possible_gates(env: Routing):
    surpass_all_possible_gates(env._state)

def env_surpass_all_possible_gates_with_info(env: Routing, surpass_action):
    state: RoutingState = env._state

    if state.is_done():
        return None, None, True, None

    obs = None
    rew = None
    done = False
    info = None

    while state.is_legal_surpass(*state.interaction_circuit[state.position]):
        obs, rew, ter, trun, inf = env.step(surpass_action)
        done = ter or trun
        if done:
            return obs, rew, done, inf

    return obs, rew, done, info


def open_binary(path: Path, mode: str = 'rb', compress: Optional[str] = None):
    p = str(path)
    if compress == 'gzip' or (compress == 'auto' and p.endswith('.gz')):
        return gzip.open(p, mode)
    if compress == 'bz2'  or (compress == 'auto' and p.endswith('.bz2')):
        return bz2.open(p, mode)
    if compress == 'lzma' or (compress == 'auto' and (p.endswith('.xz') or p.endswith('.lzma'))):
        return lzma.open(p, mode)
    return open(p, mode)

def get_timestamp():
    return str(datetime.datetime.now().isoformat()).replace(':', '-').replace('.', '-')

# def _standard_reward(gates_routed: int, path_length: int) -> int:
#     return gates_routed - path_length
#
# def standard_reward(path_states: list[RoutingState]) -> int:
#     if len(path_states) <= 1:
#         return 0
#
#     new_gates_routed = path_states[0].position - path_states[-1].position
#
#     return _standard_reward(new_gates_routed, len(path_states) - 1)
#
# def _ratio_reward(gates_routed: int, path_length: int) -> float:
#     return gates_routed / path_length
#
# def ratio_reward(path_states: list[RoutingState]) -> float:
#     if len(path_states) <= 1:
#         return 0
#
#     new_gates_routed = path_states[0].position - path_states[-1].position
#
#     return _ratio_reward(new_gates_routed, len(path_states) - 1)
