class ManhattanRewarder(BasicRewarder):

    def __init__(self):
        super().__init__()

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
        return -np.abs(new_state.mapping[interaction_circuit[0][0]] - new_state.mapping[interaction_circuit[0][1]])

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