import os
from typing import List, Any
import xarray as xr
import numpy as np
from pathlib import Path
from qce_circuit.language.intrf_declarative_circuit import InitialStateContainer
from qce_circuit.structure.acquisition_indexing.kernel_repetition_code import (
    SimulatedRepetitionExperimentKernel,
)
from qce_circuit.connectivity.intrf_channel_identifier import IQubitID
from qce_circuit.connectivity.intrf_connectivity_surface_code import ISurfaceCodeLayer
from qce_interp.utilities.expected_parities import initial_state_to_expected_parity
from qce_interp.utilities.serialize_error_identifier import construct_processed_dataset
from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier
from qce_interp.interface_definitions.intrf_state_classification import (
    StateClassifierContainer,
    ParityType,
)


def tensor_to_error_identifier(
        result_tensor: np.ndarray,
        depth: int,
        repetitions: int,
        initial_state: InitialStateContainer,
        code_layout: ISurfaceCodeLayer,
        involved_qubits_qubit_ids: List[IQubitID],  # Ordered
        **kwargs,
) -> ErrorDetectionIdentifier:
    # Extract keyword arguments
    inverse_parity: bool = kwargs.get("inverse_parity", False)
    stabilizer_active: bool = kwargs.get("stabilizer_active", True)

    # Data allocation
    involved_data_qubit_ids: List[IQubitID] = [
        qubit_id
        for qubit_id in involved_qubits_qubit_ids
        if qubit_id in code_layout.data_qubit_ids
    ]
    involved_ancilla_qubit_ids: List[IQubitID] = [
        qubit_id
        for qubit_id in involved_qubits_qubit_ids
        if qubit_id in code_layout.ancilla_qubit_ids
    ]
    expected_parity = initial_state_to_expected_parity(
        initial_state=initial_state,
        parity_layout=code_layout,
        involved_data_qubit_ids=involved_data_qubit_ids,
        involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
        inverse_parity=inverse_parity,
        stabilizer_active=stabilizer_active,
    )

    classifier_lookup = {}
    for i, qubit_id in enumerate(involved_qubits_qubit_ids):
        state_classification = result_tensor[:, :, i].flatten()
        state_classification[state_classification == 2] = (
            1  # Map 0, 1, 2 outcomes to 0, 1
        )

        classifier_lookup[qubit_id] = StateClassifierContainer(
            state_classification=state_classification,
            _expected_parity=(
                ParityType.EVEN
                if qubit_id not in expected_parity
                else expected_parity[qubit_id]
            ),
            _stabilizer_reset=False,
        )

    return ErrorDetectionIdentifier(
        classifier_lookup=classifier_lookup,
        index_kernel=SimulatedRepetitionExperimentKernel(
            rounds=[depth],
            involved_data_qubit_ids=involved_data_qubit_ids,
            involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
            experiment_repetitions=repetitions,
        ),
        involved_qubit_ids=involved_qubits_qubit_ids,
        device_layout=code_layout,
        qec_rounds=[depth],
        use_heralded_post_selection=False,
        use_projected_leakage_post_selection=False,
        use_stabilizer_leakage_post_selection=False,
        post_selection_qubits=None,
        use_computational_parity=True,
    )


def error_identifier_to_dataset(
        error_identifier: ErrorDetectionIdentifier,
        initial_state: InitialStateContainer,
        code_layout: ISurfaceCodeLayer,
        noise_config_string: str,
) -> xr.Dataset:
    dataset: xr.Dataset = construct_processed_dataset(
        error_identifier=error_identifier,
        initial_state=initial_state,
        qec_rounds=error_identifier.qec_rounds,
        code_layout=code_layout,
    )
    dataset.attrs["noise_config"] = noise_config_string

    return dataset


def save_dataset_to_netcdf(
        dataset: xr.Dataset,
        file_path: Path,
        overwrite: bool = False,
        **kwargs: Any
) -> None:
    """
    Saves an xarray.Dataset to a NetCDF file.

    :param dataset: The xarray.Dataset to be saved.
    :param file_path: The full path (including filename) for the output .nc file.
    :param overwrite: If True, it will overwrite the file if it already exists.
                      If False, it will raise a FileExistsError.
    :param kwargs: Additional keyword arguments to be passed directly to the
                   xarray.Dataset.to_netcdf() method (e.g., 'engine', 'encoding').
    :return: None. The function saves the file as a side effect.
    :raises FileExistsError: If the file already exists and overwrite is False.
    :raises IOError: For other file system-related errors.
    """
    # Check if the file exists and handle the overwrite logic
    if overwrite:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(
            f"File '{file_path}' already exists. Set overwrite=True to replace it."
        )
    try:
        # Save the dataset to a NetCDF file
        dataset.to_netcdf(path=file_path, **kwargs)
        print(f"Dataset successfully saved to: {file_path}")
    except IOError as e:
        print(f"An I/O error occurred while saving the file: {e}")
        raise