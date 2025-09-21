############################################################
# Module describing state measurement projector
############################################################
from itertools import product
from typing import List, Dict, Callable, Tuple
import numpy as np
from numpy._typing import NDArray
from quantumsim import DiCarloLabModel, State
from quantumsim.circuits import FinalizedCircuit, Circuit


class StateProjector:
    """
    Behaviour class, processes Pauli-Vector-Base by calculating qubit-state output probability vector.
    """

    # region Class Properties
    @property
    def output_state_vector(self) -> np.ndarray:
        return self._cartesian_product_array
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
        ])))
        invalid_cartesian_product: bool = len(self._cartesian_product_array) < 1
        assert not invalid_cartesian_product, "Requires at least 1 state for measurement projection, check if the specified measurement qubits are correct or if dim_per_qubit is correct."

        self._meas_map: Dict[int, Callable] = {
            0: model.measure_project_0,
            1: model.measure_project_1,
            2: model.measure_project_2,
        }
        self._meas_outcome_to_circuit: Dict[Tuple, FinalizedCircuit] = {}
        for meas_outcome in self._cartesian_product_array:
            meas_outcome = tuple(meas_outcome)
            _gates = []
            for qubit_id in self._ordered_qubit_ids:
                if qubit_id in self._ordered_meas_qubit_ids:
                    meas_outcome_index: int = self._ordered_meas_qubit_ids.index(qubit_id)
                    _gates.append(self._meas_map[meas_outcome[meas_outcome_index]](qubit_id, t_start=0))
                else:
                    _gates.append(model.idle_meas_gate(qubit_id, t_start=0))

            self._meas_outcome_to_circuit[meas_outcome] = model.finalize(Circuit(
                qubits=self._ordered_qubit_ids,
                gates=_gates,
            ))
    # endregion

    # region Class Methods
    def construct_output_vector(self, state: State) -> NDArray[np.float32]:
        result: NDArray[np.float32] = np.empty(shape=(len(self.output_state_vector)), dtype=np.float32)

        for i, meas_outcome in enumerate(self._cartesian_product_array):
            output_state = state.copy()
            finalized_meas_circuit = self._meas_outcome_to_circuit[tuple(meas_outcome)]
            finalized_meas_circuit @ output_state
            prob_out_state = np.trace(output_state.pauli_vector.to_dm())
            result[i] = np.real(prob_out_state)
        return result
    # endregion
