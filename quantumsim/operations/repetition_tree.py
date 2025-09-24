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
from numpy.typing import NDArray
from tqdm import tqdm
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


class LazyTreeCircuitEvaluation:
    """
    Behaviour class, exposing methods for repeated transformation of Pauli state vector under circuit evaluation.
    """

    # region Class Constructor
    def __init__(self, initial_state: State, circuit: FinalizedCircuit, state_projector: StateProjector, final_state_projector: StateProjector, max_depth: int):
        self._initial_state: State = initial_state
        self._circuit: FinalizedCircuit = circuit
        self._state_projector: StateProjector = state_projector
        self._final_state_projector: StateProjector = final_state_projector
        self._max_depth: int = max_depth
        # Initial circuit process step
        self._circuit @ initial_state

        self.root = Node(
            depth=0,
            rho_pre=initial_state,
            p_state=state_projector.construct_output_vector(state=initial_state)
        )
    # endregion

    # region Class Methods
    def get_state_projector_at_depth(self, depth: int, max_depth: int) -> StateProjector:
        """Return the correct measurement channel for a given depth."""
        return self._state_projector if depth < max_depth - 1 else self._final_state_projector

    def add_node(self, parent: Node, state_key: IStateKey, max_depth: int) -> Node:
        """:return: Node after extending parent evaluation."""
        if state_key in parent.children:  # Cached
            return parent.children[state_key]

        # Trace state according to measured state-key
        rho_post: State = self.get_state_projector_at_depth(depth=parent.depth, max_depth=max_depth).trace_out(
            state=parent.rho_pre,
            state_key=state_key,
        )
        # Evaluate circuit
        self._circuit @ rho_post

        child_node: Node = Node(
            depth=parent.depth + 1,
            rho_pre=rho_post,
            p_state=self.get_state_projector_at_depth(depth=parent.depth + 1, max_depth=max_depth).construct_output_vector(state=rho_post)
        )
        parent.children[state_key] = child_node
        return child_node

    def _evaluate_as_index(self, depth: int = 1, rng: Optional = None) -> NDArray[np.int8]:
        """:return: Array-like of 'evaluated' state outcomes."""
        if rng is None:
            rng = np.random.default_rng()

        result: np.ndarray = np.empty(shape=depth, dtype=np.int8)
        node = self.root

        for i in range(depth):
            idx = fast_choice(node.p_state, rng)
            state_key: IStateKey = self.get_state_projector_at_depth(depth=i, max_depth=depth).output_state_vector_keys[idx]
            node = self.add_node(
                parent=node,
                state_key=state_key,
                max_depth=depth,
            )
            result[i] = idx
        return result

    def evaluate_as_index_array(self, depth: int = 1, repetitions: int = 1, rng: Optional = None, verbose: bool = False) -> NDArray[np.int8]:
        """:return: Array-like of 'evaluated' state outcomes."""
        result: NDArray[np.int8] = np.zeros(shape=(
            repetitions,
            depth,
        ), dtype=np.int8)
        for i in tqdm(range(repetitions), desc="Decision-tree cached repetitions", disable=not verbose):
            result[i, :] = self._evaluate_as_index(
                depth=depth,
                rng=rng,
            )
        return result

    def evaluate_as_array(self, depth: int = 1, repetitions: int = 1, rng: Optional = None, verbose: bool = False) -> NDArray[np.int8]:
        """:return: Array-like of 'evaluated' state outcomes."""
        depth_plus_data = depth + 1
        index_array: NDArray[np.int8] = self.evaluate_as_index_array(
            depth=depth_plus_data,
            repetitions=repetitions,
            rng=rng,
            verbose=verbose,
        )
        result: NDArray[np.int8] = np.zeros(shape=(
            repetitions,
            depth,
            len(self._final_state_projector.ordered_qubit_ids),
        ), dtype=np.int8)
        for i in range(repetitions):
            for j in range(depth_plus_data):
                meas_index: int = index_array[i, j]
                at_max_depth: bool = j == depth_plus_data - 1
                if not at_max_depth:
                    for k, qubit_id in enumerate(self._state_projector.ordered_qubit_ids):
                        acquisition_index: Optional[int] = self._state_projector.get_acquisition_index(qubit_id=qubit_id)
                        if acquisition_index is not None:
                            result[i, j, k] = self._state_projector.output_state_vector[meas_index][acquisition_index]  # qec round 1 to N - 1
                else:
                    if at_max_depth:
                        for k, qubit_id in enumerate(self._final_state_projector.ordered_qubit_ids):
                            acquisition_index: Optional[int] = self._final_state_projector.get_acquisition_index(qubit_id=qubit_id)
                            if acquisition_index is not None:
                                result[i, j-1, k] = self._final_state_projector.output_state_vector[meas_index][acquisition_index]  # qec round 1 to N - 1

        return result
    # endregion
