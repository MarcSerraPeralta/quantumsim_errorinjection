import os
import xarray as xr
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from quantumsim import Setup
from quantumsim.models import DiCarloLabModel
from example_circuits.meas_control_repeated_circuits import (
    InitialStateType,
    MeasurementControl,
    CodeSettings,
    RepetitionCodeType,
)
from qce_circuit.connectivity.intrf_channel_identifier import QubitIDObj
from qce_circuit.connectivity.intrf_connectivity_surface_code import IParityGroup, ISurfaceCodeLayer
from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier


if __name__ == '__main__':

    noise_file = "setup_noisy_test.yaml"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = Path(os.path.join(current_dir, ".."))
    setup = Setup.from_file(os.path.join(ROOT_DIR, noise_file))
    state_dim: int = 3
    dim_lookup = {
        "D7": 2,
        "Z3": state_dim,
        "D4": state_dim,
        "Z1": state_dim,
        "D1": 2,
        "X1": state_dim,
        "D2": 2,
        "X2": state_dim,
        "D3": 2,
    }
    model = DiCarloLabModel(
        setup=setup,
        dim_lookup=dim_lookup,
    )

    common_path: Path = ROOT_DIR / "simulated_datasets"
    qec_rounds: int = 20
    repetitions = 10_000

    code_settings = CodeSettings.from_code_type(RepetitionCodeType.DISTANCE_5)
    # meas_ctrl = MeasurementControl(
    #     noise_model=model,
    #     code_settings=code_settings,
    #     dim_lookup=dim_lookup,
    #     initial_state_type=InitialStateType.ODD,
    # )

    code_layout = code_settings.code_layout

    list_of_datasets: List[xr.Dataset] = []
    for parity_group in code_layout.parity_group_z + code_layout.parity_group_x:
        print(parity_group.ancilla_id)

        error_budget_circuits: List[Tuple[str, Tuple[List[str], List], InitialStateType]] = [
            ("defect_rates_baseline", MeasurementControl.construct_detector_only_circuit_gates(model, parity_group, code_layout), InitialStateType.ALL_ZERO),
            ("defect_rates_switching", MeasurementControl.construct_detector_inverse_parity_circuit_gates(model, parity_group, code_layout), InitialStateType.ALL_ZERO),
            ("defect_rates_decoherence", MeasurementControl.construct_decoherence_only_circuit_gates(model, parity_group, code_layout), InitialStateType.ALL_ZERO),
            ("defect_rates_active_zz_data_bitflip", MeasurementControl.construct_relaxation_data_constant_circuit_gates(model, parity_group, code_layout), InitialStateType.ALL_ONES),
            ("defect_rates_full_even_parity", MeasurementControl.construct_parity_circuit_args(model, parity_group, code_layout), InitialStateType.EVEN),
            ("defect_rates_full_odd_parity", MeasurementControl.construct_parity_circuit_args(model, parity_group, code_layout), InitialStateType.ODD),
        ]

        for i, (label, cycle_circuit, initial_state_type) in enumerate(error_budget_circuits):
            qubits, gates = cycle_circuit
            meas_ctrl = MeasurementControl(
                noise_model=model,
                code_settings=CodeSettings(
                    qubits=qubits,
                    code_layout=code_settings.code_layout,
                ),
                dim_lookup=dim_lookup,
                initial_state_type=initial_state_type,
            )

            error_identifier: ErrorDetectionIdentifier = meas_ctrl.construct_error_identifier(
                cycle_circuit_gates=gates,
                depth=qec_rounds,
                repetitions=repetitions,
                **dict(
                    inverse_parity=True if label == "defect_rates_switching" else False,
                    stabilizer_active=True if label in ["defect_rates_full_even_parity", "defect_rates_full_odd_parity"] else False,
                )
            )

            dataset = meas_ctrl.construct_dataset(
                error_identifier=error_identifier,
                noise_file_path=ROOT_DIR / noise_file,
            )
            dataset = dataset.expand_dims(
                dim={
                "anillca_id": [parity_group.ancilla_id.id],
                "defect_label": [label],
            })

            # Define the target (new) names
            new_name_1: str = "defect_rates"
            new_name_2: str = "defect_rates_post_selected"

            # Define the dynamic (old) names based on the coordinate
            old_name_1: str = f"defect_rates_{parity_group.ancilla_id.id}"
            old_name_2: str = f"defect_rates_post_selected_{parity_group.ancilla_id.id}"

            # Create a mapping of {old_name: new_name}
            # This map will only include variables that actually exist in the dataset
            rename_map: Dict[str, str] = {}
            if old_name_1 in dataset.data_vars:
                rename_map[old_name_1] = new_name_1
            if old_name_2 in dataset.data_vars:
                rename_map[old_name_2] = new_name_2
            dataset = dataset.rename_vars(rename_map)

            # print(dataset)
            list_of_datasets.append(dataset)

            # print(dataset)
            # meas_ctrl.save_dataset_to_netcdf(dataset, common_path / f"defect_rate_p{parity_group.ancilla_id.id}-{label}.nc", overwrite=True)

    # # Concatenate along a new, unnamed dimension
    combined_dataset: xr.Dataset = xr.combine_by_coords(list_of_datasets)
    print(combined_dataset)
    # # Assign the coordinate values to the new dimension
    # combined_stepwise.coords["error_probability"] = error_probabilities
    meas_ctrl.save_dataset_to_netcdf(combined_dataset, common_path / f"defect_rate_error_budget_d5_all.nc", overwrite=True)
