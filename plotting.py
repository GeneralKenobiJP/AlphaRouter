import networkx as nx
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
import numpy as np
from typing import Sequence, Tuple, Union


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
