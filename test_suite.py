import math
from copy import deepcopy

import networkx as nx
import pytest
import torch
import numpy as np
from qgym.envs import Routing
from qgym.envs.routing import RoutingState
from qgym.generators import BasicInteractionGenerator
from torch import nn, Tensor, tensor

from custom_interaction_generator import CustomInteractionGenerator
from main import ReplayBuffer, ManhattanRewarder, AlphaRouterAgent, \
    LOOKAHEAD_LENGTH
from mcts import MCTS, gen_hash, MCTSWorker, inference_loop, Node, ParallelMCTS
from transformer import PositionalEncoding, TransformerLayer, Transformer
from utils import compare_routing_states, embed_nodes

import torch.multiprocessing as mp


# --- utils ---
def test_compare_routing_states():
    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2)])

    interaction_generator = BasicInteractionGenerator(seed=0)
    interaction_generator.set_state_attributes(connection_graph=connection_graph)

    state1 = RoutingState(interaction_generator=interaction_generator, max_observation_reach=40, connection_graph=connection_graph, observe_legal_surpasses=True, observe_connection_graph=True)
    state2 = RoutingState(interaction_generator=interaction_generator, max_observation_reach=40, connection_graph=connection_graph, observe_legal_surpasses=True, observe_connection_graph=True)

    state1.reset()
    state2.reset()

    state2.mapping = deepcopy(state1.mapping)
    state2.interaction_circuit = deepcopy(state1.interaction_circuit)

    assert compare_routing_states(state1, state2) == True

    state2.update_state(1)

    assert compare_routing_states(state1, state2) == False

# --- Node ---
def test_hash():
    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2)])

    interaction_generator = BasicInteractionGenerator(seed=0)
    interaction_generator.set_state_attributes(connection_graph=connection_graph)

    state1 = RoutingState(interaction_generator=interaction_generator, max_observation_reach=40, connection_graph=connection_graph, observe_legal_surpasses=True, observe_connection_graph=True)
    state2 = RoutingState(interaction_generator=interaction_generator, max_observation_reach=40, connection_graph=connection_graph, observe_legal_surpasses=True, observe_connection_graph=True)

    state1.reset()
    state2.reset()

    state2.mapping = deepcopy(state1.mapping)
    state2.interaction_circuit = deepcopy(state1.interaction_circuit)

    node1 = Node(state1)
    node2 = Node(state2)

    assert gen_hash(state1) == gen_hash(state2)
    assert hash(node1) == hash(node2)
    assert node1 == node2

    state2.update_state(1)
    node3 = Node(state2)

    assert gen_hash(state1) != gen_hash(state2)
    assert hash(node1) != hash(node3)
    assert node1 != node3

# -- Transpositions
def test_transpositions():
    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2)])

    interaction_generator = BasicInteractionGenerator(seed=0)
    interaction_generator.set_state_attributes(connection_graph=connection_graph)

    state1 = RoutingState(interaction_generator=interaction_generator, max_observation_reach=40,
                          connection_graph=connection_graph, observe_legal_surpasses=True,
                          observe_connection_graph=True)
    state2 = RoutingState(interaction_generator=interaction_generator, max_observation_reach=40,
                          connection_graph=connection_graph, observe_legal_surpasses=True,
                          observe_connection_graph=True)

    mcts = MCTSWorker(idx=0, qubit_embedding=None, device=torch.device('cpu'), swap_options=4)

    state1.reset()
    state2.reset()

    state2.mapping = deepcopy(state1.mapping)
    state2.interaction_circuit = deepcopy(state1.interaction_circuit)

    node1 = Node(state1)
    node2 = Node(state2)

    mcts.put_transposition(gen_hash(node1.state), node1)

    assert len(mcts.transposition_table) == 1
    assert mcts.get_transposition(gen_hash(node2.state)) == node1

    state3 = deepcopy(state2)
    state3.update_state(1)
    node3 = Node(state3)

    mcts.put_transposition(gen_hash(node3.state), node3)

    assert len(mcts.transposition_table) == 2
    assert mcts.get_transposition(gen_hash(node1.state)) == node1
    assert mcts.get_transposition(gen_hash(node3.state)) == node3

    mcts.put_transposition(gen_hash(node2.state), node2)
    assert len(mcts.transposition_table) == 2

class MockNode:
    def __init__(self, state):
        self.state = state
        self.children = dict()
        self.hash = 0
        self.value = 5
        self.visits = 3

    def is_leaf_node(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        return False

    def is_exhausted(self, swap_options: int = 0) -> bool:
        return len(self.children) == 1

    def __hash__(self):
        if self.hash == 0:
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

def test_loop_selection():
    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2)])

    mcts = MCTSWorker(idx=0, qubit_embedding=None, device=torch.device('cpu'), swap_options=4)

    interaction_generator = BasicInteractionGenerator(seed=0)
    interaction_generator.set_state_attributes(connection_graph=connection_graph)

    parent: MockNode = None

    state_template = RoutingState(interaction_generator=interaction_generator, max_observation_reach=40,
                         connection_graph=connection_graph, observe_legal_surpasses=True,
                         observe_connection_graph=True)

    for i in range(4):
        state = deepcopy(state_template)
        state.reset()

        node = MockNode(state)
        mcts.put_transposition(gen_hash(state), node)
        if parent is not None:
            parent.children[node] = 0
            parent = node
        if i == 0:
            mcts.root = node
            parent = node

    parent.children[mcts.root] = 0

    leaf_node, is_terminal, path_states, path_actions = mcts.select(mcts.root)

    assert leaf_node is None


# --- PositionalEncoding tests ---

def test_positional_encoding_shape():
    embedding_dim = 10
    lookahead_length = 7
    pe = PositionalEncoding(embedding_dim, lookahead_length)
    # Create a dummy input of shape (lookahead_length, embedding_dim)
    x = torch.zeros(1, lookahead_length, embedding_dim)
    out = pe(x)
    assert out.shape == x.shape, "PositionalEncoding should preserve input shape"


def test_positional_encoding_variation():
    embedding_dim = 8
    lookahead_length = 6
    pe = PositionalEncoding(embedding_dim, lookahead_length)
    x = torch.zeros(lookahead_length, embedding_dim)
    out = pe(x)
    # Ensure the positional encoding actually adds non-zero components
    assert not torch.allclose(out, x), "Output should differ from input for non-zero PE"


# --- TransformerLayer tests ---

def test_transformer_layer_forward():
    d_model = 12
    num_heads = 3
    ffn_dim = 16
    layer = TransformerLayer(d_model, num_heads, ffn_dim)
    # Create dummy batch (B, T, D)
    B, T = 2, 5
    x = torch.randn(B, T, d_model)
    out = layer(x)
    assert out.shape == (B, T, d_model), "TransformerLayer forward should preserve (B, T, D)"


# --- Transformer tests ---

def test_transformer_forward_shape():
    embedding_dim = 4
    num_heads = 2
    d_model = 8
    ffn_dim = 8*4
    num_layers = 1
    lookahead = 5
    swap_options = 6
    trans = Transformer(embedding_dim, num_heads, d_model, ffn_dim, num_layers, lookahead, swap_options)
    # Input to transformer.forward should be (lookahead, embedding_dim+1)
    x = torch.randn(lookahead, embedding_dim)
    out, _, _, _, _ = trans(x)

    assert out.shape == (1,), "Transformer should output (T, swap_options)"


# --- ReplayBuffer tests ---

# def test_replay_buffer_store_and_sample():
#     capacity = 8
#     device = torch.device("cpu")
#     rb = ReplayBuffer(capacity, device)
#     state_shape = (3,)
#     # Store more than capacity to test rollover
#     for i in range(capacity + 2):
#         s = torch.randn(*state_shape)
#         a = i % 4
#         r = float(i)
#         ns = torch.randn(*state_shape)
#         done = (i % 2 == 0)
#         rb.store(s, a, r, ns, done)
#     assert len(rb) == capacity, "ReplayBuffer should not exceed its capacity"
#     batch_size = 4
#     states, actions, rewards, next_states, dones = rb.sample(batch_size)
#     assert states.shape == (batch_size, *state_shape)
#     assert actions.shape == (batch_size,)
#     assert rewards.shape == (batch_size,)
#     assert next_states.shape == (batch_size, *state_shape)
#     assert dones.shape == (batch_size,)


# --- ManhattanRewarder tests ---

class DummyState:
    def __init__(self, mapping, circuit):
        self.mapping = mapping
        self.interaction_circuit = circuit
        self.n_qubits = len(mapping) - 1


def test_manhattan_rewarder_empty():
    rew = ManhattanRewarder()
    old = DummyState([0, 1], [])
    new = DummyState([0, 1], [])
    assert rew.compute_reward(old_state=old, action=0, new_state=new) == 0


def test_manhattan_rewarder_nonempty():
    rew = ManhattanRewarder()
    old = DummyState([0, 1, 2], [(0, 2)])
    new = DummyState([0, 2, 1], [(0, 2)])
    assert rew.compute_reward(old_state=old, action=0, new_state=new) == -1


def test_manhattan_rewarder_long():
    rew = ManhattanRewarder()
    old = DummyState([0, 1, 2, 3, 4, 5, 6], [(0, 6)])
    new = DummyState([0, 1, 2, 3, 4, 6, 5], [(0, 6)])
    assert rew.compute_reward(old_state=old, action=0, new_state=new) == -5


# --- embed_nodes tests ---

class DummyEnv:
    pass


def test_embed_nodes_basic():
    # agent = AlphaRouterAgent.__new__(AlphaRouterAgent)
    # Monkey-patch minimal attributes
    lookahead_length = 3
    embedding_dim = 5
    padding_dim = 2

    class S:
        n_qubits = 2
        mapping = [1, 0]
        interaction_circuit = [(0, 1), (1, 0)]

    env = DummyEnv()
    env._state = S()
    # agent.env = env
    qubit_embedding = nn.Embedding(2, embedding_dim)
    # mat = agent.embed_nodes()

    mat = embed_nodes(env._state, lookahead_length, None, qubit_embedding, embedding_dim, padding_dim)
    # Should produce a tensor of shape (2, embedding_dim+1)
    assert mat.shape == (lookahead_length, embedding_dim + padding_dim + 1)


# --- get_action tests ---

# def test_get_action_types_and_range(monkeypatch):
#     agent = AlphaRouterAgent.__new__(AlphaRouterAgent)
#     class DummyTransformer:
#         def __call__(self, x):
#             T = x.size(0)
#             A = 4
#             return torch.zeros(T, A)
#     agent.transformer = DummyTransformer()
#     fake_embedding = torch.randn(5, 10)
#     action, logp = agent.get_action(fake_embedding)
#     assert isinstance(action, int) and 0 <= action < 4
#     assert isinstance(logp, torch.Tensor) and logp.dim() == 0

# test_fp.py

def mock_worker(state_bytes, out_q):
    # state_bytes -> reconstruct or pass pickled state; for a quick test:
    out_q.put(gen_hash(state_bytes))

def test_hashing_across_processes():
    # create a RoutingState-like object S (or serialize your state)
    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])

    env = Routing(
        max_observation_reach=LOOKAHEAD_LENGTH,
        observe_legal_surpasses=True,
        observe_connection_graph=True,
        connection_graph=connection_graph,
        render_mode="rgb_array",
        rewarder=None,
        interaction_generator=None
    )

    env.reset()

    S = env._state   # create a RoutingState
    fp_main = gen_hash(S)
    q = mp.Queue()
    p = mp.Process(target=mock_worker, args=(S, q))
    p.start()
    fp_child = q.get()
    p.join()
    assert fp_main == fp_child
    print("Hashing stable across processes:", fp_main)


def test_mcts():
    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])

    env = Routing(
        max_observation_reach=LOOKAHEAD_LENGTH,
        observe_legal_surpasses=True,
        observe_connection_graph=True,
        connection_graph=connection_graph,
        render_mode="rgb_array",
        rewarder=None,
        interaction_generator=CustomInteractionGenerator()
    )

    embedding_dim = 8
    num_heads = 2
    d_model = 16
    ffn_dim = 4 * d_model
    num_layers = 1
    lookahead = 5
    swap_options = 7
    transformer = Transformer(embedding_dim, num_heads, d_model, ffn_dim, num_layers, lookahead, swap_options)
    device = torch.device("cpu")
    qubit_embedding = nn.Embedding(num_embeddings=connection_graph.number_of_nodes(),
                                   embedding_dim=embedding_dim - 1).to(device)

    mcts = MCTS(env=env, transformer=transformer, qubit_embedding=qubit_embedding, device=device,
                swap_options=swap_options, max_depth=5, lookahead_length=lookahead, embedding_dim=embedding_dim,
                padding_dim=0)

    root = deepcopy(env._state)

    assert len(mcts.transposition_table) == 0

    mcts.set_root(root)
    assert len(mcts.transposition_table) == 1
    assert mcts.root.state == root
    assert mcts.root.value == 0
    assert mcts.root.visits == 0
    assert len(mcts.root.children) == 0
    assert mcts.transposition_table[gen_hash(root)] == mcts.root

    node, action = mcts.expand(mcts.root)
    assert len(mcts.transposition_table) == 2
    assert mcts.root.state == root
    assert mcts.root.value == 0
    assert mcts.root.visits == 0
    assert len(mcts.root.children) == 1
    assert mcts.root.children[node] == 0
    assert node.value == 0
    assert node.visits == 0
    assert len(node.children) == 0
    assert action == 0

    reward = mcts.simulate(node)
    assert reward is not None

    path_states = [mcts.root.state, node.state]
    path_actions = [0, 0]
    mcts.backpropagate(path_states, path_actions, reward)

    assert len(mcts.transposition_table) == 2
    assert mcts.root.state == root
    assert mcts.root.value == reward.item()
    assert mcts.root.visits == 1
    assert len(mcts.root.children) == 1
    assert mcts.root.children[node] == 0
    assert node.value == reward
    assert node.visits == 1
    assert len(node.children) == 0

    assert mcts.transposition_table[gen_hash(root)] == mcts.root
    assert mcts.transposition_table[gen_hash(node.state)] == node

    best_action = mcts.run(deepcopy(node.state))
    assert len(mcts.transposition_table) > 2
    assert mcts.root.state == node.state
    assert node.visits > 1
    assert len(node.children) > 0
    assert 0 <= best_action < swap_options

    max_val = -np.inf
    for child in mcts.root.children:
        if child.value > max_val:
            max_val = child.value

    for child in mcts.root.children:
        if mcts.root.children[child] == best_action:
            assert child.value == max_val
        else:
            assert child.value <= max_val

    # --- ParallelMCTS implementation tests


# def test_mcts_worker():
#     connection_graph = nx.Graph()
#     connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
#
#     print("AZBEST")
#
#     env = Routing(
#         max_observation_reach=LOOKAHEAD_LENGTH,
#         observe_legal_surpasses=True,
#         observe_connection_graph=True,
#         connection_graph=connection_graph,
#         render_mode="rgb_array",
#         rewarder=None,
#         interaction_generator=CustomInteractionGenerator()
#     )
#
#     embedding_dim = 8
#     num_heads = 2
#     ffn_dim = 16
#     num_layers = 1
#     lookahead = 5
#     swap_options = 7
#     transformer = Transformer(embedding_dim, num_heads, ffn_dim, num_layers, lookahead, swap_options)
#     device = torch.device("cpu")
#     qubit_embedding = nn.Embedding(num_embeddings=connection_graph.number_of_nodes(),
#                                    embedding_dim=embedding_dim - 1).to(device)
#
#     worker = MCTSWorker(idx=0, qubit_embedding=qubit_embedding, device=device,
#                         swap_options=swap_options, rollouts=50, max_depth=50, lookahead_length=lookahead,
#                         embedding_dim=embedding_dim,
#                         padding_dim=0)
#
#     root = deepcopy(env._state)
#
#     assert len(worker.transposition_table) == 0
#
#     worker.set_root(root)
#     assert len(worker.transposition_table) == 1
#     assert worker.root.state == root
#     assert worker.root.value == 0
#     assert worker.root.visits == 0
#     assert len(worker.root.children) == 0
#     assert worker.transposition_table[gen_hash(root)] == worker.root
#
#     node, action = worker.expand(worker.root)
#     assert len(worker.transposition_table) == 2
#     assert worker.root.state == root
#     assert worker.root.value == 0
#     assert worker.root.visits == 0
#     assert len(worker.root.children) == 1
#     assert worker.root.children[node] == 0
#     assert node.value == 0
#     assert node.visits == 0
#     assert len(node.children) == 0
#     assert action == 0
#
#     manager = torch.multiprocessing.Manager()
#     output_queue = manager.Queue()
#     inference_queue = manager.Queue()
#     worker.set_queues(inference_queue, output_queue)
#
#     queue_loop = torch.multiprocessing.Process(target=inference_loop,
#                                                args=(1, inference_queue, [output_queue], transformer, device))
#     queue_loop.start()
#
#     with torch.multiprocessing.Pool(processes=1) as pool:
#         reward = pool.starmap(worker.simulate, [(node,)])[0]
#         assert reward is not None
#
#     if queue_loop.is_alive():
#         inference_queue.put(None)
#         queue_loop.join()
#
#     path_states = [worker.root.state, node.state]
#     path_actions = [0, 0]
#     worker.backpropagate(path_states, path_actions, reward)
#
#     assert len(worker.transposition_table) == 2
#     assert worker.root.state == root
#     assert worker.root.value == reward.item()
#     assert worker.root.visits == 1
#     assert len(worker.root.children) == 1
#     assert worker.root.children[node] == 0
#     assert node.value == reward
#     assert node.visits == 1
#     assert len(node.children) == 0
#
#     assert worker.transposition_table[gen_hash(root)] == worker.root
#     assert worker.transposition_table[gen_hash(node.state)] == node
#
#     queue_loop = torch.multiprocessing.Process(target=inference_loop,
#                                                args=(1, inference_queue, [output_queue], transformer, device))
#     queue_loop.start()
#
#     with torch.multiprocessing.Pool(processes=1) as pool:
#         best_action, state_value = pool.starmap(worker.run, [(deepcopy(node.state), False)])[0]
#
#     if queue_loop.is_alive():
#         inference_queue.put(None)
#         queue_loop.join()
#
#     assert node.is_terminal()
#     assert len(worker.transposition_table) > 2
#     assert worker.root.state == node.state
#     assert node.visits > 1
#     assert len(node.children) > 0
#     assert 0 <= best_action < swap_options
#     assert state_value is not None
#
#     max_val = -np.inf
#     for child in worker.root.children:
#         if child.value > max_val:
#             max_val = child.value
#
#     for child in worker.root.children:
#         if worker.root.children[child] == best_action:
#             assert child.value == max_val
#         else:
#             assert child.value <= max_val

def test_validate_mcts_output():
    mp.set_start_method("spawn", force=True)

    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2), (2,3), (3,4), (4,5), (5,6), (6,7)])
    # connection_graph.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6)])

    embedding_dim = 20
    embedding_dim_inc = 21
    padding_dim = 3
    num_heads = 6
    num_layers = 4
    d_model = 216
    lookahead_length = 48


    env = Routing(
        max_observation_reach=LOOKAHEAD_LENGTH,
        observe_legal_surpasses=True,
        observe_connection_graph=True,
        connection_graph=connection_graph,
        render_mode="rgb_array",
        rewarder=None,
        interaction_generator=CustomInteractionGenerator()
    )

    env.reset()

    transformer = Transformer(embedding_dim=embedding_dim_inc + padding_dim, num_heads=num_heads, d_model=d_model, ffn_dim=4*d_model,
                              num_layers=num_layers, lookahead_length=lookahead_length,
                              swap_options=connection_graph.number_of_edges(), device=torch.device("cpu"))

    rollouts = 200
    processes = 1
    # c_constant = math.sqrt(2)
    c_constant = 100

    mcts = ParallelMCTS(transformer=transformer,
                        qubit_embedding=nn.Embedding(num_embeddings=connection_graph.number_of_nodes(), embedding_dim=embedding_dim),
                        device=torch.device("cpu"),swap_options=connection_graph.number_of_edges(), rollouts=rollouts,
                        lookahead_length=lookahead_length, embedding_dim=embedding_dim, padding_dim=padding_dim,
                        processes=processes, merge_worker_trees=False, c=c_constant)

    actions, state_values = mcts.run([env._state])
    mcts.close_workers()

    print(f'Output actions: {actions}')
    print(f'Output state values: {state_values}')

    assert actions[0] == 0 or actions[0] == connection_graph.number_of_edges() - 1
