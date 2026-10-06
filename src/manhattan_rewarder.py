import networkx as nx
import numpy as np
from qgym.envs.routing import RoutingState, BasicRewarder

def _manhattan_distance(state: RoutingState, cache) -> int:
    """
    Calculates Manhattan distance between the qubits of the gate that is currently to be routed.
    Uses cache so that the distance between any 2 qubits needs to be calculated only once.

    Parameters
    ----------
    state - state of the environment
    cache - distance cache

    Returns
    -------
    Manhattan distance between two qubits that are to be routed now.
    """
    q1, q2 = state.interaction_circuit[state.position]

    p1 = state.mapping[q1]
    p2 = state.mapping[q2]

    if cache[p1][p2] is not None:
        return cache[p1][p2]

    dist = nx.shortest_path_length(state.connection_graph, p1, p2)
    cache[p1][p2] = dist
    cache[p2][p1] = dist

    return dist


class ManhattanRewarder(BasicRewarder):
    """
    This rewarder gives the bigger penalty the longer the manhattan distance between 2 qubits of the gate to be routed is.
    """

    def __init__(self, n_qubits):
        super().__init__()
        self.distance_cache = [[None] * n_qubits for _ in range(n_qubits)]
        # print(self.distance_cache)

    def compute_reward(
        self,
        *,
        old_state: RoutingState,
        action: int,
        new_state: RoutingState,
    ) -> float:
        return - _manhattan_distance(new_state, self.distance_cache) + _manhattan_distance(old_state, self.distance_cache)