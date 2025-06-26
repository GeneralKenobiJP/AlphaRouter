from typing import SupportsInt, Generator, Any

import networkx as nx
import numpy as np
from numpy._typing import NDArray
from qgym.generators.interaction import InteractionGenerator
from qgym.utils.input_parsing import parse_seed
from qgym.utils.input_validation import check_int, check_graph_is_valid_topology


class CustomInteractionGenerator(InteractionGenerator):
    """:class:`BasicInteractionGenerator` is an interaction generation implementation.

    Interactions are completely randomly generated using the :func:`numpy.random.choice`
    method.
    """

    def __init__(
        self,
        max_length: SupportsInt = 10,
        seed: Generator | SupportsInt | None = None,
    ) -> None:
        """Init of the :class:`BasicInteractionGenerator`.

        Args:
            max_length: Maximum length of the generated interaction circuits. Defaults
                to 10.
            seed: Seed to use.
        """
        self.max_length = check_int(max_length, "max_length", l_bound=1)
        self.rng = parse_seed(seed)
        self.finite = False
        self.n_qubits: int
        super().__init__()

    def set_state_attributes(
        self, *, connection_graph: nx.Graph | None = None, **kwargs: Any
    ) -> None:
        """Set the `n_qubits` attribute.

        Args:
            connection_graph: A :class:`~networkx.Graph` representation of the
                connection graph.
            kwargs: Additional keyword arguments. These are not used.
        """
        connection_graph = check_graph_is_valid_topology(
            connection_graph, "connection_graph"
        )
        self.n_qubits = connection_graph.number_of_nodes()

    def __repr__(self) -> str:
        """String representation of the :class:`BasicInteractionGenerator`."""
        return (
            f"BasicInteractionGenerator[max_length={self.max_length}, "
            f"rng={self.rng}, "
            f"finite={self.finite}]"
        )

    def __next__(self) -> NDArray[np.int_]:
        """Create a new randomly generated interaction circuit."""
        # length = self.rng.integers(1, self.max_length + 1)
        length = 1

        circuit = np.zeros((length, 2), dtype=int)
        circuit[0] = (0, self.n_qubits - 1)
        # for idx in range(length):
        #     circuit[idx] = self.rng.choice(self.n_qubits, size=2, replace=False)

        return circuit