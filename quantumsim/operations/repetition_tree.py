# -------------------------------------------
# Module describing Lazy Decision-tree caching
# (init ρ) ──SchMap──► ρ₀
#                 │
#                 │  p_state(ρ₀)  choose  “10”
#                 ▼
#         (branch key “10“)
#         ρ₁ = collapse(ρ₀,"10")
#                 │
#                 │  SchMap
#                 ▼
#                … …
# -------------------------------------------
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from numpy._typing import NDArray

from quantumsim import State
from quantumsim.circuits import FinalizedCircuit
from quantumsim.operations.state_projector import IStateKey, StateProjector


@dataclass
class Node:
    depth: int
    """round index (0 … qec_rounds)."""
    rho_pre: State
    """ρ before measuring this round."""
    p_state: np.ndarray
    """probability vector for that ρ."""
    children: Dict[IStateKey, "Node"] = field(default_factory=dict)


def fast_choice(p: np.ndarray, rng: np.random.Generator) -> int:
    """
    Choose index based on probability array p quickly.
    :param p: probability array summing to 1
    :param rng: random number generator
    :return: selected index
    """
    cumulative_p = np.cumsum(p)
    random_number = rng.random()
    idx = np.searchsorted(cumulative_p, random_number, side='right')
    return idx


class RepeatedCircuitEvaluation:
    """
    Behaviour class, exposing methods for repeated transformation of Pauli state vector under circuit evaluation.
    """

    # region Class Properties
    @property
    def output_state_vector_keys(self) -> List[IStateKey]:
        """:return: Array-like of output states in string representation."""
        return self._state_projector.output_state_vector_keys
    # endregion

    # region Class Constructor
    def __init__(self, initial_state: State, circuit: FinalizedCircuit, state_projector: StateProjector):
        self._initial_state: State = initial_state
        self._circuit: FinalizedCircuit = circuit
        self._state_projector: StateProjector = state_projector
        self.root = Node(
            depth=0,
            rho_pre=initial_state,
            p_state=state_projector.construct_output_vector(state=initial_state)
        )
    # endregion

    # region Class Methods
    def add_node(self, parent: Node, state_key: IStateKey) -> Node:
        """:return: Node after extending parent evaluation."""
        if state_key in parent.children:  # Cached
            return parent.children[state_key]

        # Trace state according to measured state-key
        rho_post: State = self._state_projector.trace_out(
            state=parent.rho_pre,
            state_key=state_key,
        )
        # Evaluate circuit
        self._circuit @ rho_post

        child_node: Node = Node(
            depth=parent.depth + 1,
            rho_pre=rho_post,
            p_state=self._state_projector.construct_output_vector(state=rho_post)
        )
        parent.children[state_key] = child_node
        return child_node

    def _evaluate_as_index(self, times: int = 1, rng: Optional = None) -> NDArray[np.int8]:
        """:return: Array-like of 'evaluated' state outcomes."""
        if rng is None:
            rng = np.random.default_rng()

        result: np.ndarray = np.empty(shape=times, dtype=np.int8)
        node = self.root

        for i in range(times):
            idx = fast_choice(node.p_state, rng)
            state_key: IStateKey = self.output_state_vector_keys[idx]
            node = self.add_node(
                parent=node,
                state_key=state_key
            )
            result[i] = idx
        return result

    def evaluate(self, times: int = 1, rng: Optional = None) -> List[IStateKey]:
        """:return: Array-like of 'evaluated' state outcomes."""
        evaluate_as_index: NDArray[np.int8] = self._evaluate_as_index(
            times=times,
            rng=rng,
        )
        result: List[IStateKey] = [
            self.output_state_vector_keys[idx]
            for idx in evaluate_as_index
        ]
        return result

    def evaluate_as_array(self, times: int = 1, rng: Optional = None) -> NDArray[np.int8]:
        """:return: Array-like of 'evaluated' state outcomes."""
        evaluate_as_index: NDArray[np.int8] = self._evaluate_as_index(
            times=times,
            rng=rng,
        )
        result = self._state_projector.output_state_vector[evaluate_as_index]
        return result
    # endregion
