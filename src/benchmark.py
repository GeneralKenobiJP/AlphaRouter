import networkx as nx
from gymnasium import Env
from qgym.envs import Routing
from qgym.envs.routing import RoutingState
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap, SetLayout, ApplyLayout, BasicSwap


def _env_to_coupling_map(env: Env) -> tuple[CouplingMap, int]:
    """
    Convert the connection graph of the env from gym API to CouplingMap from qiskit API.
    Parameters
    ----------
    env - the environment of the routing problem

    Returns
    -------
    tuple consisting of:
        * coupling_map - connection graph translated to qiskit API so that the baseline can be run
        * number of physical qubits
    """
    state: RoutingState = env._state
    graph = state.connection_graph

    edge_list = []
    for u, v in graph.edges():
        edge_list.append([u, v])
        edge_list.append([v, u])

    coupling_map = CouplingMap(edge_list)

    return coupling_map, graph.number_of_nodes()

def _env_to_circuit(env: Env, num_qubits) -> QuantumCircuit:
    """
    Convert the interaction graph of the env from qgym API to CouplingMap from qiskit API.
    Parameters
    ----------
    env - the environment of the routing problem

    Returns
    -------
    interaction graph translated to qiskit API so that the baseline can be run
    """
    state: RoutingState = env._state
    interaction_graph = state.interaction_circuit

    qc = QuantumCircuit(num_qubits)

    for q1, q2 in interaction_graph:
        qc.cx(q1, q2)

    return qc

def run_benchmark(env: Env, seed: int | None = 42, baseline: str = 'SABRE', pass_manager: PassManager = None) -> tuple[QuantumCircuit, QuantumCircuit]:
    """
    Run the given qiskit baseline with the given environment and seed. A custom pass manager can be supplied.
    Parameters
    ----------
    env - environment of the routing problem
    seed - seed of the benchmark
    baseline - type of the baseline to run ('SABRE', 'basic')
    pass_manager - qiskit's manager of compilation passes. If none, a default one is constructed.
        See qiskit API documentation for more details.

    Returns
    -------
    a tuple consisting of:
        * the problem circuit to be routed
        * the output of the routing algorithm
    """
    if pass_manager is None:
        pass_manager = PassManager()

    if seed is not None:
        env.reset(seed=seed)

    coupling_map, num_qubits = _env_to_coupling_map(env)

    initial_layout = list(range(num_qubits))

    pass_manager.append(SetLayout(initial_layout))
    pass_manager.append(ApplyLayout())

    circuit = _env_to_circuit(env, num_qubits)

    match baseline.lower():
        case "sabre":
            pass_manager.append(SabreSwap(coupling_map, trials=8, fake_run=False))
        case 'basic':
            pass_manager.append(BasicSwap(coupling_map))
        case _:
            pass_manager.append(SabreSwap(coupling_map, trials=8, fake_run=False))

    return circuit, pass_manager.run(circuit)

def benchmark(env: Env, seed: int | None = 42, baseline: str = 'SABRE', show_circuits: bool = False) -> int:
    """
    Runs the benchmark and computes statistics.
    Uses the given qiskit baseline with the given environment and seed.

    Parameters
    ----------
    env - environment of the routing problem
    seed - seed of the benchmark
    baseline - type of the baseline to run ('SABRE', 'basic')
    show_circuits - if true, the following information will be printed:
        * initial problem circuit to be solved
        * the circuit after the routing
        * a dict of instructions in the initial circuit
        * a dict of instructions in the circuit after the routing

    Returns
    -------
    the number of swap gates inserted by the baseline algorithm
    """
    initial_circuit, routed_circuit = run_benchmark(env, seed=seed, baseline=baseline)

    routed_ops = routed_circuit.count_ops()

    if show_circuits:
        print(initial_circuit)
        print(routed_circuit)
        print(initial_circuit.count_ops())
        print(routed_ops)
    return routed_ops.get('swap', 0)

if __name__ == '__main__':
    connection_graph = nx.Graph()
    # connection_graph.add_edges_from([(0,1), (1,2), (2,3)])
    connection_graph.add_edges_from([(0,1), (1,2), (2,3), (3,4)])
    # connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,0)])

    # connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,6), (6,7)])

    env = Routing(
        max_observation_reach=48,
        observe_legal_surpasses=True,
        observe_connection_graph=True,
        connection_graph=connection_graph,
        render_mode="rgb_array",
        rewarder=None,
        interaction_generator=None
    )

    swaps = benchmark(env, show_circuits=True)
    print(swaps)
