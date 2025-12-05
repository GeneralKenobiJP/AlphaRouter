import networkx as nx
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
import numpy as np
from typing import Sequence, Tuple, Union, Optional, Iterable, Any

import math
from collections import deque


def plot(x_axis, values, label, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_axis, values, label=label, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def moving_average(values, window):
    ma_values = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        ma_values.append(sum(window_values) / len(window_values))

    return ma_values


def plot_mapping_and_circuit(
    connection_graph: nx.Graph,
    mapping: Union[np.ndarray, Sequence[int]],
    interaction_circuit: Union[np.ndarray, Sequence[Tuple[int,int]]],
    position: int = 0,
    max_gates: int = 48,
    figsize: Tuple[int,int] = (14, 6),
    title: str | None = None
):
    """
    Plot the current mapping (logical -> physical) on the connection graph
    and plot the remaining interaction circuit.

    Parameters
    ----------
    connection_graph : networkx.Graph
        Physical topology graph. Node ids should be integers 0..N-1.
    mapping : array-like
        Either:
          - logical->physical array: mapping[logical_index] = physical_index
          - OR physical->logical array: mapping[physical_index] = logical_index
        The function attempts to detect which form you passed.
    interaction_circuit : sequence of pairs (q1,q2)
        Sequence of logical two-qubit gates (each gate given by pair of logical qubit indices).
    position : int
        Index inside interaction_circuit that is the "current position" (gates before are executed).
    max_gates : int
        How many upcoming gates to render in the circuit panel.
    figsize : tuple
        Matplotlib figure size.
    """
    print(f'mapping: {mapping}')
    print(f'connection_graph: {connection_graph.edges}')
    print(f'interaction_circuit: {interaction_circuit}')
    n_qubits = connection_graph.number_of_nodes()

    # Build figure with two panels
    fig = plt.figure(figsize=figsize)
    ax_map = fig.add_subplot(1, 2, 1)

    # Layout for graph; fixed seed for reproducibility
    try:
        pos = nx.spring_layout(connection_graph, seed=42)
    except Exception:
        pos = nx.circular_layout(connection_graph)

    phys_to_logical = {int(phys): int(log) for log, phys in enumerate(mapping)}
    # Node labels: show physical id and logical occupant if present
    node_labels = {}
    for n in connection_graph.nodes():
        if n in phys_to_logical:
            node_labels[n] = f"P{n}\nL{phys_to_logical[n]}"
        else:
            node_labels[n] = f"P{n}\n-"

    # Colors: assign color per logical qubit (if present on node) otherwise grey
    cmap = plt.get_cmap("tab20")
    node_colors = []
    for n in connection_graph.nodes():
        if n in phys_to_logical:
            logical_idx = phys_to_logical[n] % 20
            node_colors.append(cmap(logical_idx))
        else:
            node_colors.append("#DDDDDD")

    # draw base graph
    nx.draw_networkx_nodes(connection_graph, pos=pos, ax=ax_map, node_color=node_colors, node_size=900)
    nx.draw_networkx_edges(connection_graph, pos=pos, ax=ax_map, width=1.2)
    nx.draw_networkx_labels(connection_graph, pos=pos, labels=node_labels, font_size=9, ax=ax_map)

    # Highlight upcoming gates on the mapping: draw the path between the physical locations
    upcoming = interaction_circuit[position: position + max_gates]
    for i, (q1, q2) in enumerate(upcoming):
        # ensure logical indices valid
        if q1 >= n_qubits or q2 >= n_qubits:
            continue
        p1 = int(mapping[q1])
        p2 = int(mapping[q2])
        if p1 == -1 or p2 == -1:
            continue
        # draw highlighted edge
        nx.draw_networkx_edges(connection_graph, pos=pos, edgelist=[(p1, p2)], ax=ax_map,
                               width=3.0, edge_color='tab:red', style='dashed')
        # annotate gate number near midpoint
        x_mid = (pos[p1][0] + pos[p2][0]) / 2
        y_mid = (pos[p1][1] + pos[p2][1]) / 2
        ax_map.text(x_mid, y_mid, str(i), color='tab:red', fontsize=9, fontweight='bold', ha='center', va='center')

    ax_map.set_title("Physical topology with current logical→physical mapping")
    ax_map.axis('off')

    # Right panel: draw the remaining circuit (logical gates). Prefer Qiskit circuit diagram if available.
    ax_circ = fig.add_subplot(1, 2, 2)
    ax_circ.axis('off')

    remaining = interaction_circuit[position: position + max_gates]

    if len(remaining) > 0:
        # build a small Qiskit circuit on logical qubits
        qc = QuantumCircuit(n_qubits, name="remaining")
        for (q1, q2) in remaining:
            qc.cx(int(q1), int(q2))
            qc.barrier()
        # draw using Qiskit's matplotlib drawer to a temporary figure, then blit it into ax_circ
        try:
            # qc.draw returns a matplotlib.figure.Figure when output='mpl'
            circ_fig = qc.draw(output='mpl', idle_wires=False)
            # convert circ_fig to an image array and plot it inside ax_circ
            circ_fig.canvas.draw()
            w, h = circ_fig.canvas.get_width_height()
            img = np.frombuffer(circ_fig.canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
            ax_circ.imshow(img)
            ax_circ.set_title("Next gates (logical circuit)")
            ax_circ.axis('off')
            plt.close(circ_fig)
        except Exception:
            # fallback to textual listing if drawing fails
            ax_circ.text(0.01, 0.99, "\n".join([f"{i}: CX L{a} -> L{b}" for i,(a,b) in enumerate(remaining)]),
                         va='top', family='monospace')
    else:
        # empty remaining -> textual panel
        if len(remaining) == 0:
            ax_circ.text(0.5, 0.5, "No remaining gates", ha='center', va='center', fontsize=12)
        else:
            lines = []
            for i, (a, b) in enumerate(remaining):
                p_a = mapping[a] if a < n_qubits else -1
                p_b = mapping[b] if b < n_qubits else -1
                lines.append(f"{i:02d}: CX L{a}({p_a}) -> L{b}({p_b})")
            ax_circ.text(0.01, 0.99, "\n".join(lines), va='top', family='monospace', fontsize=10)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# helper for detection of tensor types (optional)
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _action_to_int(action: Any) -> str:
    """Return a short printable representation for an action (int or tensor or other)."""
    if _HAS_TORCH and isinstance(action, torch.Tensor):
        try:
            return str(int(action.item()))
        except Exception:
            return str(action)
    try:
        return str(int(action))
    except Exception:
        return str(action)


def _mapping_snippet(mapping, max_chars: int = 80) -> str:
    """Short textual representation of a mapping (logical->physical)."""
    try:
        if _HAS_TORCH and isinstance(mapping, torch.Tensor):
            arr = mapping.detach().cpu().numpy().reshape(-1)
        else:
            arr = np.array(mapping).reshape(-1)
        pairs = [f"{i}->{int(arr[i])}" for i in range(len(arr))]
        s = ", ".join(pairs)
    except Exception:
        s = str(mapping)
    if len(s) > max_chars:
        s = s[:max_chars] + " …"
    return s


def _get_children_action_child_pairs(node) -> Iterable[Tuple[Any, Any]]:
    """
    Yield (action, child_node) pairs regardless of how node.children is stored.
    Handles both: parent.children[action] = child  and parent.children[child] = action.
    """
    try:
        items = list(node.children.items())
    except Exception:
        return []

    if len(items) == 0:
        return []

    key0, val0 = items[0]
    # if keys are ints (or tensors) -> action -> child
    # if keys are Node objects -> child -> action
    from types import SimpleNamespace
    # a cheap duck-typing: check if key has attribute 'state' to guess Node
    def looks_like_node(x):
        return hasattr(x, "state") and hasattr(x, "visits")

    if looks_like_node(key0):
        # keys are child nodes: key=node, value=action
        for child_node, action in items:
            yield (action, child_node)
    else:
        # keys are actions (or ints) and values are child nodes
        for action, child_node in items:
            yield (action, child_node)


def plot_mcts_tree(
    root_node,
    mcts=None,
    max_nodes: int = 200,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    include_ucb: bool = True,
    cmap_name: str = "coolwarm",
    node_size_base: float = 1000.0
):
    """
    Plot an MCTS tree rooted at `root_node` using matplotlib.

    Parameters
    ----------
    root_node : Node
        An instance of your Node class (must have .children (dict), .visits, .value, .state.mapping)
    mcts : MCTS-like instance or None
        If provided and has mcts.ucb(parent, child), function will compute child UCB values.
    max_nodes : int
        Maximum number of nodes to render (BFS truncated).
    figsize : (w,h)
    title : optional figure title
    save_path : optional path to save the figure (PNG). If provided, saved and figure is closed (unless show=True).
    show : whether to call plt.show()
    include_ucb : whether to include UCB in node labels (requires mcts)
    cmap_name : name of matplotlib colormap for node colors (value -> color)
    node_size_base : base multiplier for node circle sizes (scaled by visits)
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Build DiGraph and metadata by BFS up to max_nodes
    G = nx.DiGraph()
    q = deque()
    q.append((root_node, None, None))  # (node, parent_node, action_from_parent)
    seen = set()
    node_index_map = {}  # map id(node) -> small idx
    node_meta = {}  # idx -> dict with meta

    idx_counter = 0
    while q and len(node_index_map) < max_nodes:
        node, parent, action = q.popleft()
        nid = id(node)
        if nid in seen:
            # still add edge if needed
            if parent is not None:
                G.add_edge(node_index_map[id(parent)], node_index_map[nid], action=_action_to_int(action))
            continue

        seen.add(nid)
        node_index_map[nid] = idx_counter
        node_id = idx_counter
        idx_counter += 1

        # gather metadata
        # visits = getattr(node, "visits", 0)
        visits = node.visits
        # value = float(getattr(node, "value", 0.0))
        value = node.value
        try:
            # mapping = getattr(node.state, "mapping", None)
            mapping = node.state.mapping
        except Exception:
            mapping = None

        node_meta[node_id] = {
            "node_obj": node,
            "visits": visits,
            "value": value,
            "mapping_snip": _mapping_snippet(mapping) if mapping is not None else "<no mapping>",
            "action_from_parent": _action_to_int(action) if action is not None else None,
            "ucb": None,  # filled later when parent available
        }

        G.add_node(node_id)

        if parent is not None:
            G.add_edge(node_index_map[id(parent)], node_id, action=_action_to_int(action))

        # enqueue children (action, child) pairs
        for action_child, child_node in _get_children_action_child_pairs(node):
            q.append((child_node, node, action_child))

    # compute UCB values for edges if possible
    if include_ucb and mcts is not None:
        # iterate edges: for each parent->child we compute ucb
        for u, v, data in G.edges(data=True):
            parent_node = node_meta[u]["node_obj"]
            child_node = node_meta[v]["node_obj"]
            try:
                ucb_val = float(mcts.ucb(parent_node, child_node))
            except Exception:
                ucb_val = None
            node_meta[v]["ucb"] = ucb_val
            data["ucb"] = ucb_val

    # prepare layout
    if G.number_of_nodes() == 0:
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "Empty tree", ha="center", va="center")
        if title:
            plt.title(title)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    try:
        pos = nx.spring_layout(G, seed=42, k=0.8)  # nice force-directed layout
    except Exception:
        pos = nx.kamada_kawai_layout(G)

    # node sizes and colors
    visits_list = np.array([node_meta[n]["visits"] for n in sorted(node_meta.keys())], dtype=float)
    # avoid zero sizes
    visits_scaled = (visits_list + 1.0)
    node_sizes = node_size_base * (visits_scaled / max(1.0, visits_scaled.max()))

    values = np.array([node_meta[n]["value"] for n in sorted(node_meta.keys())], dtype=float)
    # normalize values to [-1,1] for color mapping
    if math.isfinite(values.max()) and math.isfinite(values.min()) and values.max() != values.min():
        vmin, vmax = values.min(), values.max()
    else:
        vmin, vmax = -1.0, 1.0
    # create color map
    cmap = plt.get_cmap(cmap_name)
    norm_vals = (values - vmin) / (vmax - vmin + 1e-12)
    node_colors = [cmap(v) for v in norm_vals]

    # draw
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title or "MCTS tree")
    ax.axis("off")

    # draw nodes as scatter (so we can control size and color)
    nodes_sorted = sorted(node_meta.keys())
    node_pos = [pos[n] for n in nodes_sorted]
    xs = [p[0] for p in node_pos]
    ys = [p[1] for p in node_pos]

    sc = ax.scatter(xs, ys, s=node_sizes, c=node_colors, zorder=3, edgecolors="k", linewidths=0.6)

    # draw edges with arrows
    # networkx draws arrows poorly with scatter; draw simple curved lines
    for (u, v, data) in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        # draw a slightly curved arrow line
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist == 0:
            continue
        # control point for curvature
        midx = (x1 + x2) / 2.0
        midy = (y1 + y2) / 2.0
        # orthogonal offset
        ox = -dy * 0.15
        oy = dx * 0.15
        # create a simple bezier-ish curve
        curve_xs = [x1, midx + ox, x2]
        curve_ys = [y1, midy + oy, y2]
        ax.plot(curve_xs, curve_ys, linewidth=0.8, color="gray", zorder=1, alpha=0.8)
        # add an arrowhead at 70% of the curve
        t = 0.7
        ax.annotate(
            "", xy=(x2, y2), xytext=(midx + ox * t, midy + oy * t),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.6),
            zorder=2
        )
        # optional edge label: action or ucb
        label = data.get("action")
        ucb = data.get("ucb")
        if ucb is not None:
            edge_label = f"{label}\nUCB={ucb:.2f}"
        else:
            edge_label = f"{label}"
        # place edge label near the mid of straight line (not perfect for curves)
        ax.text((x1 + x2) / 2.0, (y1 + y2) / 2.0, edge_label, fontsize=7, ha="center", va="center", zorder=4)

    # draw node labels (action, visits, value, ucb, mapping snippet)
    for n in nodes_sorted:
        x, y = pos[n]
        meta = node_meta[n]
        label_lines = []
        if meta["action_from_parent"] is not None:
            label_lines.append(f"A={meta['action_from_parent']}")
        label_lines.append(f"V={meta['value']:.2f}")
        label_lines.append(f"N={int(meta['visits'])}")
        if meta["ucb"] is not None:
            try:
                label_lines.append(f"UCB={meta['ucb']:.2f}")
            except Exception:
                pass
        label_lines.append(meta["mapping_snip"])
        lbl = "\n".join(label_lines)
        ax.text(x, y, lbl, fontsize=7, ha="center", va="center", zorder=5, family="monospace")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[plot_mcts_tree_matplotlib] saved to {save_path}")
    if show:
        plt.show()
    return fig
