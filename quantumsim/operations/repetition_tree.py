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
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from quantumsim import State
from quantumsim.circuits import FinalizedCircuit, Circuit
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
    idx = np.searchsorted(cumulative_p, random_number, side="right")
    return idx


class LazyTreeCircuitEvaluation:
    """
    Behaviour class, exposing methods for repeated transformation of Pauli state vector under circuit evaluation.
    """

    # region Class Constructor
    def __init__(
        self,
        initial_state: State,
        circuit: FinalizedCircuit,
        state_projector: StateProjector,
        final_state_projector: StateProjector,
        max_depth: int,
    ):
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
            p_state=state_projector.construct_output_vector(state=initial_state),
        )

    # endregion

    # region Class Methods
    def get_state_projector_at_depth(
        self, depth: int, max_depth: int
    ) -> StateProjector:
        """Return the correct measurement channel for a given depth."""
        return (
            self._state_projector
            if depth < max_depth - 1
            else self._final_state_projector
        )

    def add_node(self, parent: Node, state_key: IStateKey, max_depth: int) -> Node:
        """:return: Node after extending parent evaluation."""
        if state_key in parent.children:  # Cached
            return parent.children[state_key]

        # Trace state according to measured state-key
        rho_post: State = self.get_state_projector_at_depth(
            depth=parent.depth, max_depth=max_depth
        ).trace_out(
            state=parent.rho_pre,
            state_key=state_key,
        )
        # Evaluate circuit
        self._circuit @ rho_post

        child_node: Node = Node(
            depth=parent.depth + 1,
            rho_pre=rho_post,
            p_state=self.get_state_projector_at_depth(
                depth=parent.depth + 1, max_depth=max_depth
            ).construct_output_vector(state=rho_post),
        )
        parent.children[state_key] = child_node
        return child_node

    def _evaluate_as_index(
        self, depth: int = 1, rng: Optional = None
    ) -> NDArray[np.int8]:
        """:return: Array-like of 'evaluated' state outcomes."""
        if rng is None:
            rng = np.random.default_rng()

        result: np.ndarray = np.empty(shape=depth, dtype=np.int8)
        node = self.root

        for i in range(depth):
            idx = fast_choice(node.p_state, rng)
            state_key: IStateKey = self.get_state_projector_at_depth(
                depth=i, max_depth=depth
            ).output_state_vector_keys[idx]
            node = self.add_node(
                parent=node,
                state_key=state_key,
                max_depth=depth,
            )
            result[i] = idx
        return result

    def evaluate_as_index_array(
        self,
        depth: int = 1,
        repetitions: int = 1,
        rng: Optional = None,
        verbose: bool = False,
    ) -> NDArray[np.int8]:
        """:return: Array-like of 'evaluated' state outcomes."""
        result: NDArray[np.int8] = np.zeros(
            shape=(
                repetitions,
                depth,
            ),
            dtype=np.int8,
        )
        for i in tqdm(
            range(repetitions),
            desc="Decision-tree cached repetitions",
            disable=not verbose,
        ):
            result[i, :] = self._evaluate_as_index(
                depth=depth,
                rng=rng,
            )
        return result

    def evaluate_as_array(
        self,
        depth: int = 1,
        repetitions: int = 1,
        rng: Optional = None,
        verbose: bool = False,
    ) -> NDArray[np.int8]:
        """:return: Array-like of 'evaluated' state outcomes."""
        depth_plus_data = depth + 1
        index_array: NDArray[np.int8] = self.evaluate_as_index_array(
            depth=depth_plus_data,
            repetitions=repetitions,
            rng=rng,
            verbose=verbose,
        )
        result: NDArray[np.int8] = np.zeros(
            shape=(
                repetitions,
                depth,
                len(self._final_state_projector.ordered_qubit_ids),
            ),
            dtype=np.int8,
        )
        for i in range(repetitions):
            for j in range(depth_plus_data):
                meas_index: int = index_array[i, j]
                at_max_depth: bool = j == depth_plus_data - 1
                if not at_max_depth:
                    for k, qubit_id in enumerate(
                        self._state_projector.ordered_qubit_ids
                    ):
                        acquisition_index: Optional[int] = (
                            self._state_projector.get_acquisition_index(
                                qubit_id=qubit_id
                            )
                        )
                        if acquisition_index is not None:
                            result[i, j, k] = self._state_projector.output_state_vector[
                                meas_index
                            ][
                                acquisition_index
                            ]  # qec round 1 to N - 1
                else:
                    if at_max_depth:
                        for k, qubit_id in enumerate(
                            self._final_state_projector.ordered_qubit_ids
                        ):
                            acquisition_index: Optional[int] = (
                                self._final_state_projector.get_acquisition_index(
                                    qubit_id=qubit_id
                                )
                            )
                            if acquisition_index is not None:
                                result[
                                    i, j - 1, k
                                ] = self._final_state_projector.output_state_vector[
                                    meas_index
                                ][
                                    acquisition_index
                                ]  # qec round 1 to N - 1

        return result

    # endregion


class RepeatedCircuitEvaluation:
    """
    Behaviour class, exposing methods for repeated transformation of Pauli state vector under circuit evaluation.
    """

    # region Class Constructor
    def __init__(
        self,
        initial_state: State,
        circuit: FinalizedCircuit,
        state_projector: StateProjector,
        final_state_projector: StateProjector,
    ):
        self._initial_state: State = initial_state
        self._circuit: FinalizedCircuit = circuit
        self._state_projector: StateProjector = state_projector
        self._final_state_projector: StateProjector = final_state_projector
    # endregion

    # region Class Methods
    def get_state_projector_at_depth(
        self, depth: int, max_depth: int
    ) -> StateProjector:
        """Return the correct measurement channel for a given depth."""
        return (
            self._state_projector
            if depth < max_depth - 1
            else self._final_state_projector
            # self._final_state_projector
        )

    def evaluate_as_index_array(
        self,
        model,
        depth: int = 1,
        repetitions: int = 1,
        verbose: bool = False,
    ) -> NDArray[np.int8]:
        """:return: Array-like of 'evaluated' state outcomes."""
        random_array = np.random.rand(repetitions, depth)
        meas_outcome_index: NDArray[np.int8] = np.zeros(
            shape=(
                repetitions,
                depth,
            ),
            dtype=np.int8,
        )

        def distance_3_prep_initial_state_gates():
            return [
                model.idle_sq_gate("D4", t_start=0),
                model.idle_sq_gate("Z1", t_start=0),
                model.rotate_x_180("D1", t_start=0),
                model.idle_sq_gate("X1", t_start=0),
                model.idle_sq_gate("D2", t_start=0),
            ]

        qubits = ["D4", "Z1", "D1", "X1", "D2"]
        initial_circuit = model.finalize(Circuit(qubits=qubits, gates=distance_3_prep_initial_state_gates()))

        for i in tqdm(
            range(repetitions),
            desc="Repetitions",
            disable=not verbose,
        ):
            initial_state = State(qubits, dim=3, force=False)  # initialization in ground state
            initial_circuit @ initial_state

            # initial_state = self._initial_state.copy()

            for j in tqdm(range(depth), desc="QEC rounds", disable=True):
                # Apply QEC-cycle for DEPTH - 1. Where last iteration is reserved for final data qubit measurement
                state_projector: StateProjector = self._final_state_projector
                if j < depth - 1:
                    self._circuit @ initial_state
                    state_projector: StateProjector = self._state_projector

                probability_spike: float = float(random_array[i, j])

                for k, meas_outcome_key in tqdm(enumerate(state_projector.output_state_vector_keys), desc="Measurement trials", disable=True):
                    trial_state: State = initial_state.copy()
                    meas_project_circuit = state_projector.meas_outcome_to_circuit[meas_outcome_key]
                    meas_project_circuit @ trial_state
                    # trial_probability_state = trial_state.trace()
                    # probability_spike -= trial_probability_state  # Probability of measuring trial state
                    accepted_trial: bool = probability_spike <= 0
                    if accepted_trial:
                        # Continue with trial state as initial state
                        initial_state = trial_state.copy()
                        initial_state.pauli_vector.renormalize()
                        meas_outcome_index[i, j] = k
                        break

            # DEBUG
            # for j in range(depth):
            #     probability_spike: float = float(random_array[i, j])
            #     state_projector: StateProjector = self.get_state_projector_at_depth(
            #         depth=j, max_depth=depth
            #     )
            #     # Execute ancilla readout every qec-round
            #     for k, meas_outcome_key in enumerate(state_projector.output_state_vector_keys):
            #         trial_state: State = initial_state.copy()
            #         meas_project_circuit = state_projector.meas_outcome_to_circuit[
            #             meas_outcome_key
            #         ]
            #         meas_project_circuit @ trial_state
            #         _ = trial_state.pauli_vector.diagonal()
            #         # trial_probability_state: float = trial_state.trace()
            #         # # Process probability spike
            #         # probability_spike -= trial_probability_state
            #         # accepted_trial: bool = probability_spike <= 0
            #         # if accepted_trial:
            #         #     break
            #         # else:
            #         #     del trial_state
            #
            #     trial_state.pauli_vector.renormalize()
            #     # WARNING, CUDA gives memory error when just calling 'initial_state = trial_state'
            #     initial_state = trial_state.copy()
            #     result[i, j] = 0
            # DEBUG

            # # Iterate through qec-cycles
            # for j in range(depth):
            #     probability_spike: float = float(random_array[i, j])
            #     state_projector: StateProjector = self.get_state_projector_at_depth(
            #         depth=j, max_depth=depth
            #     )
            #     # Execute ancilla readout every qec-round
            #     for k, meas_outcome_key in enumerate(state_projector.output_state_vector_keys):
            #         trial_state: State = initial_state.copy()
            #         meas_project_circuit = state_projector.meas_outcome_to_circuit[
            #             meas_outcome_key
            #         ]
            #         meas_project_circuit @ trial_state
            #         trial_probability_state: float = trial_state.trace()
            #         # Process probability spike
            #         probability_spike -= trial_probability_state
            #         accepted_trial: bool = probability_spike <= 0
            #         # initial_state = trial_state
            #
            #         if accepted_trial:
            #             # trial_state.pauli_vector.renormalize()
            #             # WARNING, CUDA gives memory error when just calling 'initial_state = trial_state'
            #             # initial_state = trial_state
            #             # meas_project_circuit @ initial_state
            #             result[i, j] = k
            #             break
        return meas_outcome_index

    def evaluate_as_array(
        self,
        model,
        depth: int = 1,
        repetitions: int = 1,
        verbose: bool = False,
    ) -> NDArray[np.int8]:
        """:return: Array-like of 'evaluated' state outcomes."""
        depth_plus_data = depth + 1
        index_array: NDArray[np.int8] = self.evaluate_as_index_array(
            model=model,
            depth=depth_plus_data,
            repetitions=repetitions,
            verbose=verbose,
        )
        result: NDArray[np.int8] = np.zeros(
            shape=(
                repetitions,
                depth,
                len(self._final_state_projector.ordered_qubit_ids),
            ),
            dtype=np.int8,
        )
        for i in range(repetitions):
            for j in range(depth_plus_data):
                meas_index: int = index_array[i, j]
                at_max_depth: bool = j == depth_plus_data - 1
                if not at_max_depth:
                    for k, qubit_id in enumerate(
                        self._state_projector.ordered_qubit_ids
                    ):
                        acquisition_index: Optional[int] = (
                            self._state_projector.get_acquisition_index(
                                qubit_id=qubit_id
                            )
                        )
                        if acquisition_index is not None:
                            result[i, j, k] = self._state_projector.output_state_vector[
                                meas_index
                            ][
                                acquisition_index
                            ]  # qec round 1 to N - 1
                else:
                    if at_max_depth:
                        for k, qubit_id in enumerate(
                            self._final_state_projector.ordered_qubit_ids
                        ):
                            acquisition_index: Optional[int] = (
                                self._final_state_projector.get_acquisition_index(
                                    qubit_id=qubit_id
                                )
                            )
                            if acquisition_index is not None:
                                result[
                                    i, j - 1, k
                                ] = self._final_state_projector.output_state_vector[
                                    meas_index
                                ][
                                    acquisition_index
                                ]  # qec round 1 to N - 1

        return result
    # endregion


class FlattenedCircuitEvaluation:
    """
    Behaviour class, designed for CUDA execution of constant-depth circuit evaluations.
    """

    # region Class Properties
    @property
    def available_index_count(self) -> int:
        return self.total_chains - self.count_unavailable()
    # endregion

    # region Class Constructor
    def __init__(self, dimension: int, ancilla_count: int, rounds: int):
        """
        Initializes the MaskManager.

        :param dimension: The output dimension (d).
        :param ancilla_count: The number of ancilla qubits to measure (a).
        :param rounds: The number of QEC cycle rounds (r).
        """
        self.dimension: int = dimension
        self.ancilla_count: int = ancilla_count
        self.rounds: int = rounds
        self.base: int = self.dimension ** self.ancilla_count
        self.total_chains: int = self.base ** self.rounds

        # Pre-compute the divisors (B**(r-n)) for O(1) lookup.
        # The key is the prefix length 'n'.
        self.divisors: Dict[int, int] = {
            n: self.base ** (self.rounds - n) for n in range(1, self.rounds + 1)
        }

        # Storage for the pre-computed mask prefixes.
        # Key: prefix length 'n'. Value: a set of masked prefixes.
        self.masks: Dict[int, Set[int]] = {n: set() for n in range(1, self.rounds + 1)}
    # endregion

    # region Class Methods
    def get_state_indices(self, index: int) -> np.ndarray:
        """Calculates the unique sequence of output state indices for a given input index.

        This function maps a single integer `index` to a unique sequence of
        state indices of length `rounds`. This is mathematically equivalent to
        representing the number `index` in base B, where B = dimension ** ancilla_count.

        The mathematical relationship is:
        index = s_{r-1}*B^{r-1} + ... + s_1*B^1 + s_0*B^0

        :param index: The input index, from 0 to d^(a*r) - 1.
        :return: A numpy array of length `rounds` containing the unique sequence of output state indices.
        """
        base: int = self.base
        total_chains: int = self.total_chains
        rounds: int = self.rounds

        if not (0 <= index < total_chains):
            raise ValueError(
                f"Input index {index} is out of the valid range [0, {total_chains - 1}]"
            )

        output_indices: np.ndarray = np.zeros(rounds, dtype=int)
        current_value: int = index

        # Iterate from the last position (least significant) to the first (most significant)
        for i in range(rounds - 1, -1, -1):
            # The remainder gives the digit (state index)
            output_indices[i] = current_value % base
            # Update the value for the next iteration
            current_value //= base

        return output_indices

    def add_mask(self, mask_index: int, n: int):
        """
        Adds a new mask to the manager.

        :param mask_index: The index whose first n sequence values define the mask.
        :param n: The length of the prefix to mask (from 1 to r).
        """
        if not (1 <= n <= self.rounds):
            raise ValueError(f"Mask length n={n} must be in [1, {self.rounds}]")

        # Calculate the prefix value that defines this mask
        divisor: int = self.divisors[n]
        prefix: int = mask_index // divisor

        # Add the prefix to the set for the given length n
        self.masks[n].add(prefix)

    def is_available(self, index: int) -> bool:
        """
        Checks if a given index is available (i.e., not masked).

        The check is O(M) where M is the number of distinct mask lengths defined,
        which is at most O(r).

        :param index: The index to check.
        :return: True if the index is not masked, False otherwise.
        """
        # Check against masks for every possible prefix length
        for n, masked_prefixes in self.masks.items():
            if not masked_prefixes:
                continue

            # Calculate the prefix for the index being checked
            divisor: int = self.divisors[n]
            current_prefix: int = index // divisor

            # If its prefix is in the set of masked prefixes, it's not available
            if current_prefix in masked_prefixes:
                return False

        # If no mask was matched, the index is available
        return True

    def find_next_available(self, start_index: int) -> Optional[int]:
        """
        Finds the first available index at or after a given starting index.

        This method efficiently jumps over blocks of masked indices rather
        than checking them one by one.

        :param start_index: The index from which to start the search (inclusive).
        :return: The first available index, or None if no available index is found.
        """
        candidate_index: int = start_index

        while candidate_index < self.total_chains:
            # Find all masks that block the current candidate index
            blocking_masks_info: List[Tuple[int, int]] = []
            for n, masked_prefixes in self.masks.items():
                if not masked_prefixes:
                    continue

                divisor: int = self.divisors[n]
                current_prefix: int = candidate_index // divisor

                if current_prefix in masked_prefixes:
                    blocking_masks_info.append((current_prefix, divisor))

            # If the list is empty, the index is available
            if not blocking_masks_info:
                return candidate_index

            # If blocked, calculate the next potential candidate by "jumping"
            # over the end of the largest blocking range.
            max_jump_target: int = 0
            for prefix, divisor in blocking_masks_info:
                jump_target = (prefix + 1) * divisor
                if jump_target > max_jump_target:
                    max_jump_target = jump_target

            candidate_index = max_jump_target

        # If the loop completes, no available index was found
        return None

    def count_unavailable(self) -> int:
        """
        Calculates the total number of unique unavailable (masked) indices.

        This method avoids double-counting by converting masks into integer
        intervals and calculating the size of their union. The complexity is
        dominated by sorting the intervals, making it O(M log M) where M is the
        total number of unique masks.

        :return: The total count of unavailable indices.
        """
        # Generate all [start, end] intervals from the masks
        intervals: List[Tuple[int, int]] = []
        for n, prefixes in self.masks.items():
            divisor: int = self.divisors[n]
            block_size: int = self.base ** (self.rounds - n)
            for prefix in prefixes:
                start_index: int = prefix * block_size
                end_index: int = start_index + block_size - 1
                intervals.append((start_index, end_index))

        if not intervals:
            return 0

        # Sort intervals based on their start index
        intervals.sort()

        # Merge overlapping intervals and sum their sizes
        total_unavailable_count: int = 0
        merged_start, merged_end = intervals[0]

        for i in range(1, len(intervals)):
            next_start, next_end = intervals[i]

            # Check for overlap with the current merged interval
            if next_start <= merged_end:
                # They overlap, so extend the merged interval
                merged_end = max(merged_end, next_end)
            else:
                # No overlap, the previous merged interval is complete.
                # Add its size to the total count.
                total_unavailable_count += (merged_end - merged_start + 1)
                # Start a new merged interval
                merged_start, merged_end = next_start, next_end

        # Add the size of the last merged interval
        total_unavailable_count += (merged_end - merged_start + 1)

        return total_unavailable_count
    # endregion


if __name__ == '__main__':

    logic = FlattenedCircuitEvaluation(ancilla_count=2, dimension=2, rounds=5)
    print(logic.count_unavailable(), logic.available_index_count)
    logic.add_mask(mask_index=10, n=2)
    print(logic.base, logic.total_chains)

    # for i in range(100):
    #     print(i, logic.get_state_indices(i), logic.is_available(i))
    print(logic.find_next_available(start_index=0))
    print(logic.count_unavailable(), logic.available_index_count)
