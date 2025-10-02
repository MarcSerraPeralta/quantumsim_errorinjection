# -------------------------------------------
# Module describing state measurement projector
# -------------------------------------------
from itertools import product, chain
from typing import List, Dict, Callable, Iterable, Optional, Any
import numpy as np
from numpy._typing import NDArray
from quantumsim import DiCarloLabModel, State
from quantumsim.circuits import FinalizedCircuit, Circuit


IStateKey = str


class StateProjector:
    """
    Behaviour class, processes Pauli-Vector-Base by calculating qubit-state output probability vector.
    """

    # region Class Properties
    @property
    def output_state_vector(self) -> np.ndarray:
        """:return: Array-like of output states in ndarray representation."""
        return self._cartesian_product_array

    @property
    def output_state_vector_keys(self) -> List[IStateKey]:
        """:return: Array-like of output states in string representation."""
        return self._cartesian_product_array_keys

    @property
    def ordered_qubit_ids(self) -> List[str]:
        """:return: Array-like of order qubit-IDs"""
        return self._ordered_qubit_ids

    @property
    def ordered_measured_qubit_ids(self) -> List[str]:
        """:return: Array-like of order qubit-IDs"""
        return self._ordered_meas_qubit_ids

    @property
    def nr_measured_qubits(self) -> int:
        """:return: Number of measured qubits"""
        return len(self._ordered_meas_qubit_ids)

    # endregion

    # region Class Constructor
    def __init__(
        self,
        ordered_qubit_ids: List[str],
        meas_qubit_ids: List[str],
        dim_per_qubit: Dict[str, int],
        model: DiCarloLabModel,
    ):
        self._ordered_qubit_ids: List[str] = ordered_qubit_ids
        self._ordered_meas_qubit_ids: List[str] = [
            qubit_id for qubit_id in ordered_qubit_ids if qubit_id in meas_qubit_ids
        ]
        # self._dim_per_qubit: Dict[str, int] = dim_per_qubit
        invalid_dim: bool = any(value <= 0 for key, value in dim_per_qubit.items())
        assert (
            not invalid_dim
        ), f"dim_per_qubit value needs to be at least 1 or higher, instead: {dim_per_qubit}"

        self._cartesian_product_array: np.ndarray = np.asarray(
            list(
                product(
                    *[
                        range(dim_per_qubit[qubit_id])
                        for qubit_id in self._ordered_meas_qubit_ids
                    ]
                )
            ),
            dtype=np.int8,
        )
        self._cartesian_product_array_keys: List[str] = (
            self.convert_array_of_lists_to_strings(
                array_of_integer_lists=self._cartesian_product_array,
            )
        )
        invalid_cartesian_product: bool = len(self._cartesian_product_array) < 1
        assert (
            not invalid_cartesian_product
        ), "Requires at least 1 state for measurement projection, check if the specified measurement qubits are correct or if dim_per_qubit is correct."

        self._model = model
        self._meas_map: Dict[int, Callable] = {
            0: model.measure_project_0,
            1: model.measure_project_1,
            2: model.measure_project_2,
        }
        self.meas_outcome_to_circuit: Dict[IStateKey, FinalizedCircuit] = {}
        for meas_outcome, meas_outcome_key in zip(
            self._cartesian_product_array, self._cartesian_product_array_keys
        ):
            """
            _gates = []
            for qubit_id in self._ordered_qubit_ids:
                if qubit_id in self._ordered_meas_qubit_ids:
                    meas_outcome_index: int = self._ordered_meas_qubit_ids.index(
                        qubit_id
                    )
                    _gates.append(
                        self._meas_map[meas_outcome[meas_outcome_index]](
                            qubit_id, t_start=0
                        )
                    )
                else:
                    _gates.append(model.idle_meas_gate(qubit_id, t_start=0))

            self.meas_outcome_to_circuit[meas_outcome_key] = model.finalize(
                Circuit(
                    qubits=self._ordered_qubit_ids,
                    gates=_gates,
                ),
                bases_in=model.get_ordered_qubits_to_basis(self._ordered_qubit_ids),
            )
            """
            self.meas_outcome_to_circuit[meas_outcome_key] = None

    # endregion

    # region Class Methods
    # TODO: Also store finalized meas circuit with all output probabilities to prevent re-calculation.
    # TODO: For final (Data) qubit measurement, just take diagonal of density matrix as proxi for probability.
    def construct_output_vector(self, state: State) -> NDArray[float]:
        """:return: Probability vector for each product state measured."""
        result: NDArray[float] = np.empty(
            shape=(len(self.output_state_vector)), dtype=float
        )

        for i, meas_outcome_key in enumerate(
            self._cartesian_product_array_keys
        ):  # '00', '01', '02', '10', ...
            output_state = state.copy()
            finalized_meas_circuit = self.meas_outcome_to_circuit[
                meas_outcome_key
            ]  # Circuit containing only measure_project
            finalized_meas_circuit @ output_state
            prob_out_state = output_state.trace()
            result[i] = prob_out_state  # Already normalized
        return result

    def trace_out(self, state: State, state_key: IStateKey) -> State:
        """:return: Traced-out Pauli state vector based on measured product state."""
        output_state = state.copy()
        finalized_meas_circuit = self.meas_outcome_to_circuit[state_key]
        finalized_meas_circuit @ output_state
        output_state.pauli_vector.renormalize()  # Renormalized Pauli vector
        return output_state

    def get_acquisition_index(self, qubit_id: str) -> Optional[int]:
        if qubit_id not in self._ordered_meas_qubit_ids:
            return None
        return self._ordered_meas_qubit_ids.index(qubit_id)

    def get_meas_outcome_to_circuit_gates(self, state_key: IStateKey) -> List[Any]:
        meas_outcome: np.ndarray = self._cartesian_product_array[self._cartesian_product_array_keys.index(state_key)]
        _gates = []
        for qubit_id in self._ordered_qubit_ids:
            if qubit_id in self._ordered_meas_qubit_ids:
                meas_outcome_index: int = self._ordered_meas_qubit_ids.index(
                    qubit_id
                )
                _gates.append(
                    self._meas_map[int(meas_outcome[meas_outcome_index])](
                        qubit_id, t_start=0
                    )
                )
            else:
                _gates.append(self._model.idle_meas_gate(qubit_id, t_start=0))

        return list(chain(*_gates))

    def get_meas_outcome_to_circuit(self, state_key: IStateKey) -> FinalizedCircuit:
        return self._model.finalize(
            Circuit(
                qubits=self._ordered_qubit_ids,
                gates=self.get_meas_outcome_to_circuit_gates(state_key=state_key),
            )
        )

    def construct_assignment_matrix(self) -> np.ndarray:
        assignment_matrices: List[np.ndarray] = [
            self._model.assign_matrix(qubit)
            for qubit in self.ordered_measured_qubit_ids
        ]
        result: np.ndarray = assignment_matrices[0]
        for i in range(1, len(assignment_matrices)):
            result = np.kron(result, assignment_matrices[i])
        return result

    def apply_assignment_uncertainty(self, input_meas_indices: np.ndarray) -> np.ndarray:
        """
        Simulates random state transitions for a batch of input states based on a probability matrix.

        This function uses a vectorized implementation of inverse transform sampling.
        For each input state, it selects an output state based on the probability
        distribution given by the corresponding row in the combined_matrix.

        :param input_meas_indices: A numpy array of integers of any shape, representing the
                             current states of the system.
        :return: A numpy array of the same shape as input_states, containing
                 the randomly chosen output states.
        """
        # Store original shape and flatten the input array for vectorized processing.
        # .ravel() is efficient as it returns a view of the original array when possible.
        assignment_matrix: np.ndarray = self.construct_assignment_matrix()
        original_shape: tuple = input_meas_indices.shape
        flat_input_meas_indices = input_meas_indices.ravel()

        # Select the probability distributions for each flattened input state.
        # If flat_input_states has shape (M,), this will be an (M, N) matrix.
        probabilities: np.ndarray = assignment_matrix[flat_input_meas_indices]
        cdfs: np.ndarray = np.cumsum(probabilities, axis=1)

        # Generate a uniform random number for each input state.
        # Shape is (M, 1) to enable broadcasting against the (M, N) cdfs matrix.
        num_inputs: int = flat_input_meas_indices.size
        random_uniforms: np.ndarray = np.random.rand(num_inputs, 1)
        # Find the first index where the CDF is greater than the random number.
        # `argmax` on a boolean array returns the index of the first True value.
        flat_output_states: np.ndarray = (random_uniforms < cdfs).argmax(axis=1)
        output_assign_indices: np.ndarray = flat_output_states.reshape(original_shape)

        return output_assign_indices
    # endregion

    # region Static Class Methods
    @staticmethod
    def convert_integer_list_to_string(integer_list: Iterable[int]) -> str:
        """
        Converts an iterable of integers into a single concatenated string.
        For example, the input [1, 2, 0] would be converted to the string "120".
        :param integer_list: A list of integers to be converted.
        :return: A string representing the concatenated integers.
        """
        return "".join(map(str, integer_list))

    @staticmethod
    def convert_array_of_lists_to_strings(
        array_of_integer_lists: Iterable[Iterable[int]],
    ) -> List[str]:
        """
        Applies the list-to-string conversion to each sublist in a list of lists.
        :param array_of_integer_lists: A list where each element is a list of integers.
        :return: A list of strings, where each string is the concatenated representation of the corresponding inner list.
        """
        return [
            StateProjector.convert_integer_list_to_string(sublist)
            for sublist in array_of_integer_lists
        ]

    # endregion


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
