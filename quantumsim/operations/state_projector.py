# -------------------------------------------
# Module describing state measurement projector
# -------------------------------------------
from itertools import product
from typing import List, Dict, Callable, Iterable
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
    def nr_measured_qubits(self) -> int:
        """:return: Number of measured qubits"""
        return len(self._ordered_meas_qubit_ids)
    # endregion

    # region Class Constructor
    def __init__(self, ordered_qubit_ids: List[str], meas_qubit_ids: List[str], dim_per_qubit: Dict[str, int], model: DiCarloLabModel):
        self._ordered_qubit_ids: List[str] = ordered_qubit_ids
        self._ordered_meas_qubit_ids: List[str] = [
            qubit_id
            for qubit_id in ordered_qubit_ids if qubit_id in meas_qubit_ids
        ]
        # self._dim_per_qubit: Dict[str, int] = dim_per_qubit
        invalid_dim: bool = any(value <= 0 for key, value in dim_per_qubit.items())
        assert not invalid_dim, f"dim_per_qubit value needs to be at least 1 or higher, instead: {dim_per_qubit}"

        self._cartesian_product_array: np.ndarray = np.asarray(list(product(*[
            range(dim_per_qubit[qubit_id])
            for qubit_id in self._ordered_meas_qubit_ids
        ])), dtype=np.int8)
        self._cartesian_product_array_keys: List[str] = self.convert_array_of_lists_to_strings(
            array_of_integer_lists=self._cartesian_product_array,
        )
        invalid_cartesian_product: bool = len(self._cartesian_product_array) < 1
        assert not invalid_cartesian_product, "Requires at least 1 state for measurement projection, check if the specified measurement qubits are correct or if dim_per_qubit is correct."

        self._meas_map: Dict[int, Callable] = {
            0: model.measure_project_0,
            1: model.measure_project_1,
            2: model.measure_project_2,
        }
        self._meas_outcome_to_circuit: Dict[IStateKey, FinalizedCircuit] = {}
        for meas_outcome, meas_outcome_key in zip(self._cartesian_product_array, self._cartesian_product_array_keys):
            _gates = []
            for qubit_id in self._ordered_qubit_ids:
                if qubit_id in self._ordered_meas_qubit_ids:
                    meas_outcome_index: int = self._ordered_meas_qubit_ids.index(qubit_id)
                    _gates.append(self._meas_map[meas_outcome[meas_outcome_index]](qubit_id, t_start=0))
                else:
                    _gates.append(model.idle_meas_gate(qubit_id, t_start=0))

            self._meas_outcome_to_circuit[meas_outcome_key] = model.finalize(Circuit(
                qubits=self._ordered_qubit_ids,
                gates=_gates,
            ))
    # endregion

    # region Class Methods
    def construct_output_vector(self, state: State) -> NDArray[np.float32]:
        """:return: Probability vector for each product state measured."""
        result: NDArray[np.float32] = np.empty(shape=(len(self.output_state_vector)), dtype=np.float32)
        prob_init_state = np.trace(state.pauli_vector.to_dm())

        for i, meas_outcome_key in enumerate(self._cartesian_product_array_keys):
            output_state = state.copy()
            finalized_meas_circuit = self._meas_outcome_to_circuit[meas_outcome_key]
            finalized_meas_circuit @ output_state
            prob_out_state = np.trace(output_state.pauli_vector.to_dm())
            result[i] = np.real(prob_out_state / prob_init_state)
        return result

    def trace_out(self, state: State, state_key: IStateKey) -> State:
        """:return: Traced-out Pauli state vector based on measured product state."""
        finalized_meas_circuit = self._meas_outcome_to_circuit[state_key]
        finalized_meas_circuit @ state
        return state
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
    def convert_array_of_lists_to_strings(array_of_integer_lists: Iterable[Iterable[int]]) -> List[str]:
        """
        Applies the list-to-string conversion to each sublist in a list of lists.
        :param array_of_integer_lists: A list where each element is a list of integers.
        :return: A list of strings, where each string is the concatenated representation of the corresponding inner list.
        """
        return [
            StateProjector.convert_integer_list_to_string(sublist) for sublist in array_of_integer_lists
        ]
    # endregion
