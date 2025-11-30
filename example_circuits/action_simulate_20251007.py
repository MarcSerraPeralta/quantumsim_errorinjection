import os
from pathlib import Path
from quantumsim import Setup
from quantumsim.models import DiCarloLabModel
from example_circuits.meas_control_repeated_circuits import (
    InitialStateType,
    MeasurementControl,
    CodeSettings,
    RepetitionCodeType,
)
from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier


if __name__ == '__main__':
    import numpy as np

    noise_file = "setup_noisy_perfect_case.yaml"
    # noise_file = "setup_noisy_experimental_match.yaml"
    # noise_file = "setup_noisy_perfect_to_experiment_case.yaml"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = Path(os.path.join(current_dir, ".."))
    setup = Setup.from_file(os.path.join(ROOT_DIR, noise_file))
    state_dim: int = 2
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
    qec_rounds: int = 10
    repetitions = 10_000

    meas_ctrl = MeasurementControl(
        noise_model=model,
        code_settings=CodeSettings.from_code_type(RepetitionCodeType.DISTANCE_5),
        dim_lookup=dim_lookup,
        initial_state_type=InitialStateType.ODD,
    )
    _, gates = meas_ctrl.construct_distance_5_bitflip_circuit_gates_fermioniclike(
        error_probability=float(0.0),
        error_coherence=True,
    )
    error_identifier, raw_dataset = meas_ctrl.construct_error_identifier(
        cycle_circuit_gates=gates,
        depth=qec_rounds,
        repetitions=repetitions,
    )
    error_identifier: ErrorDetectionIdentifier
    dataset = meas_ctrl.construct_dataset(
        error_identifier=error_identifier,
        noise_file_path=ROOT_DIR / noise_file,
    )
    dataset.attrs["error_probability"] = 0.0
    dataset.attrs["error_coherence"] = int(True)
    meas_ctrl.save_dataset_to_netcdf(dataset, common_path / f"defect_rate_d5_baseline.nc", overwrite=True)

    # meas_ctrl = MeasurementControl(
    #     noise_model=model,
    #     code_settings=CodeSettings.from_code_type(RepetitionCodeType.DISTANCE_3),
    #     dim_lookup=dim_lookup,
    #     initial_state_type=InitialStateType.ODD,
    # )
    # _, gates = meas_ctrl.construct_distance_3_bitflip_circuit_gates_fermioniclike(
    #     error_probability=float(0.0),
    #     error_coherence=True,
    # )
    # # print(gates)
    # error_identifier, raw_dataset = meas_ctrl.construct_error_identifier(
    #     cycle_circuit_gates=gates,
    #     depth=qec_rounds,
    #     repetitions=repetitions,
    # )
    # error_identifier: ErrorDetectionIdentifier
    # dataset = meas_ctrl.construct_dataset(
    #     error_identifier=error_identifier,
    #     noise_file_path=ROOT_DIR / noise_file,
    # )
    # dataset["simulated_dataset"] = raw_dataset
    # dataset.attrs.update(raw_dataset.attrs)
    # dataset.attrs["error_probability"] = 0.0
    # dataset.attrs["error_coherence"] = int(True)
    # print(dataset)
    # meas_ctrl.save_dataset_to_netcdf(dataset, common_path / f"defect_rate_d3_baseline.nc", overwrite=True)
