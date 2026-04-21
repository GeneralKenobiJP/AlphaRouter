import networkx as nx
import numpy as np
from qgym.envs.routing import RoutingState, BasicRewarder

def _manhattan_distance(state: RoutingState, cache):
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
        # interaction_circuit = new_state.interaction_circuit
        # if len(interaction_circuit) < 1:
        #     return 0
        # return -np.abs(new_state.mapping[interaction_circuit[0][0]] - new_state.mapping[interaction_circuit[0][1]])

class GridManhattanRewarder(BasicRewarder):

    def __init__(self, width: int):
        super().__init__()
        self.n = width

    def compute_reward(
        self,
        *,
        old_state: RoutingState,
        action: int,
        new_state: RoutingState,
    ) -> float:
        interaction_circuit = new_state.interaction_circuit
        if len(interaction_circuit) < 1:
            return 0
        q = new_state.mapping[interaction_circuit[0][0]]
        p = new_state.mapping[interaction_circuit[0][1]]
        n1 = q % self.n
        n2 = p % self.n
        m1 = q / self.n
        m2 = p / self.n
        return - (np.abs(n2-n1) + np.abs(m2-m1))