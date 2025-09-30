from typing import Optional
import numpy as np
from numpy.typing import NDArray
from quantumsim.operations.state_projector import StateProjector


def process_state_projection(state_projection_array: NDArray[int], state_projector: StateProjector, final_state_projector: StateProjector) -> np.ndarray:
    repetitions, depth_plus_data = state_projection_array.shape
    qec_rounds = depth_plus_data - 1

    result: NDArray[np.int8] = np.zeros(
        shape=(
            repetitions,
            qec_rounds,
            len(final_state_projector.ordered_qubit_ids),
        ),
        dtype=np.int8,
    )
    for i in range(repetitions):
        for j in range(depth_plus_data):
            meas_index: int = state_projection_array[i, j]
            at_max_depth: bool = j == depth_plus_data - 1
            if not at_max_depth:
                for k, qubit_id in enumerate(state_projector.ordered_qubit_ids):
                    acquisition_index: Optional[int] = (
                        state_projector.get_acquisition_index(
                            qubit_id=qubit_id
                        )
                    )
                    if acquisition_index is not None:
                        result[i, j, k] = state_projector.output_state_vector[meas_index][acquisition_index]  # qec round 1 to N - 1
            else:
                if at_max_depth:
                    for k, qubit_id in enumerate(final_state_projector.ordered_qubit_ids):
                        acquisition_index: Optional[int] = (
                            final_state_projector.get_acquisition_index(
                                qubit_id=qubit_id
                            )
                        )
                        if acquisition_index is not None:
                            result[i, j - 1, k] = final_state_projector.output_state_vector[meas_index][acquisition_index]  # qec round 1 to N - 1

    return result