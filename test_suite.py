from copy import deepcopy

import networkx as nx
import pytest
import torch
import numpy as np
from qgym.envs import Routing
from torch import nn, Tensor, tensor

from custom_interaction_generator import CustomInteractionGenerator
from main import ReplayBuffer, ManhattanRewarder, AlphaRouterAgent, \
    LOOKAHEAD_LENGTH
from mcts import MCTS, gen_hash
from transformer import PositionalEncoding, TransformerLayer, Transformer


# --- PositionalEncoding tests ---

def test_positional_encoding_shape():
    embedding_dim = 10
    lookahead_length = 7
    pe = PositionalEncoding(embedding_dim, lookahead_length)
    # Create a dummy input of shape (lookahead_length, embedding_dim + 1)
    x = torch.zeros(lookahead_length, embedding_dim + 1)
    out = pe(x)
    assert out.shape == x.shape, "PositionalEncoding should preserve input shape"

def test_positional_encoding_variation():
    embedding_dim = 8
    lookahead_length = 6
    pe = PositionalEncoding(embedding_dim, lookahead_length)
    x = torch.zeros(lookahead_length, embedding_dim + 1)
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
    ffn_dim = 16
    num_layers = 1
    lookahead = 5
    swap_options = 6
    trans = Transformer(embedding_dim, num_heads, ffn_dim, num_layers, lookahead, swap_options)
    # Input to transformer.forward should be (lookahead, embedding_dim+1)
    x = torch.randn(lookahead, embedding_dim + 1)
    out = trans(x)
    # Output should be (lookahead, swap_options)
    assert out.shape == (lookahead, swap_options), "Transformer should output (T, swap_options)"

# --- ReplayBuffer tests ---

def test_replay_buffer_store_and_sample():
    capacity = 8
    device = torch.device("cpu")
    rb = ReplayBuffer(capacity, device)
    state_shape = (3,)
    # Store more than capacity to test rollover
    for i in range(capacity + 2):
        s = torch.randn(*state_shape)
        a = i % 4
        r = float(i)
        ns = torch.randn(*state_shape)
        done = (i % 2 == 0)
        rb.store(s, a, r, ns, done)
    assert len(rb) == capacity, "ReplayBuffer should not exceed its capacity"
    batch_size = 4
    states, actions, rewards, next_states, dones = rb.sample(batch_size)
    assert states.shape == (batch_size, *state_shape)
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert next_states.shape == (batch_size, *state_shape)
    assert dones.shape == (batch_size,)

# --- ManhattanRewarder tests ---

class DummyState:
    def __init__(self, mapping, circuit):
        self.mapping = mapping
        self.interaction_circuit = circuit
        self.n_qubits = len(mapping) - 1

def test_manhattan_rewarder_empty():
    rew = ManhattanRewarder()
    old = DummyState([0,1], [])
    new = DummyState([0,1], [])
    assert rew.compute_reward(old_state=old, action=0, new_state=new) == 0

def test_manhattan_rewarder_nonempty():
    rew = ManhattanRewarder()
    old = DummyState([0,1,2], [(0,2)])
    new = DummyState([0,2,1], [(0,2)])
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
    agent = AlphaRouterAgent.__new__(AlphaRouterAgent)
    # Monkey-patch minimal attributes
    agent.lookahead_length = 3
    agent.embedding_dim = 5
    class S:
        n_qubits = 2
        mapping = [1, 0]
        interaction_circuit = [(0,1), (1,0)]
    env = DummyEnv()
    env._state = S()
    agent.env = env
    agent.qubit_embedding = nn.Embedding(2, agent.embedding_dim)
    mat = agent.embed_nodes()
    # Should produce a tensor of shape (2, embedding_dim+1)
    assert mat.shape == (2, agent.embedding_dim + 1)

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
    ffn_dim = 16
    num_layers = 1
    lookahead = 5
    swap_options = 7
    transformer = Transformer(embedding_dim, num_heads, ffn_dim, num_layers, lookahead, swap_options)
    device = torch.device("cpu")
    qubit_embedding = nn.Embedding(num_embeddings=connection_graph.number_of_nodes(),
                                            embedding_dim=embedding_dim-1).to(device)

    mcts = MCTS(env=env, transformer=transformer, qubit_embedding=qubit_embedding, device=device,
                swap_options=swap_options, max_depth=5, lookahead_length=lookahead, embedding_dim=embedding_dim,padding_dim=0)

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
