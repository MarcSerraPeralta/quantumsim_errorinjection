from typing import Optional
import numpy as np
from numpy.typing import NDArray
from quantumsim.operations.state_projector import StateProjector


def process_assignment_errors(
    meas_outcome_indices: np.ndarray,
    state_projector: StateProjector,
    final_state_projector: StateProjector
) -> np.ndarray:
    """
    Splits measurement outcomes, processes them separately, and recombines.

    This function is designed for scenarios where the final measurement in a sequence
    (e.g., the last column in a 2D array) is subject to a different noise
    model than the preceding (bulk) measurements.

    :param meas_outcome_indices: A 2D numpy array of measurement outcomes, where rows typically represent shots and columns represent measurement steps.
    :param state_projector: An StateProjector object with an `apply_assignment_uncertainty` method for processing the bulk measurements.
    :param final_state_projector: An StateProjector object with an `apply_assignment_uncertainty` method for processing the final measurement.
    :return: A new 2D numpy array with the processed and recombined outcomes, having the same shape as the input.
    """
    if meas_outcome_indices.ndim != 2:
        raise ValueError("meas_outcome_indices must be a 2D array.")

    # Split the data into bulk (all columns but the last) and final (last column).
    meas_bulk: np.ndarray = meas_outcome_indices[:, :-1]
    meas_final: np.ndarray = meas_outcome_indices[:, -1:]

    # Process each part with its corresponding uncertainty model.
    processed_meas_bulk: np.ndarray = state_projector.apply_assignment_uncertainty(input_meas_indices=meas_bulk)
    processed_meas_final: np.ndarray = final_state_projector.apply_assignment_uncertainty(input_meas_indices=meas_final)

    # Recombine the processed arrays back into a single array along the column axis.
    recombined_outcomes: np.ndarray = np.concatenate((processed_meas_bulk, processed_meas_final), axis=1)

    return recombined_outcomes


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