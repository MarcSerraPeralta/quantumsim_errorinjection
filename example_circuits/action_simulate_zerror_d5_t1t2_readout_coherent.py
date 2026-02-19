import os
import xarray as xr
import numpy as np
from typing import List
from pathlib import Path
from quantumsim import Setup
from quantumsim.models .model_dclab import (
    DiCarloLabModel,
    DiCarloLabModelNoIdle,
    DiCarloLabModelT2Ratio,
)
from example_circuits.meas_control_repeated_circuits import (
    InitialStateType,
    MeasurementControl,
    CodeSettings,
    RepetitionCodeType,
)
from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier


if __name__ == '__main__':

    noise_file = "setup_noisy_d5_t1t2_readout.yaml"
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
    model = DiCarloLabModelT2Ratio(
        setup=setup,
        dim_lookup=dim_lookup,
    )

    common_path: Path = ROOT_DIR / "simulated_datasets"
    qec_rounds: int = 10
    repetitions = 100_000

    z_phase_error_rates = np.linspace(0.0, 1000e3, 20)  #np.asarray([0.0, 50e3, 100e3, 500e3])

    meas_ctrl = MeasurementControl(
        noise_model=model,
        code_settings=CodeSettings.from_code_type(RepetitionCodeType.DISTANCE_5),
        dim_lookup=dim_lookup,
        initial_state_type=InitialStateType.ODD,
    )
    list_of_datasets: List[xr.Dataset] = []
    for i, z_phase_error_rate in enumerate(z_phase_error_rates):
        error_probability: float = 0.1
        involved_qubits, gates = meas_ctrl.construct_distance_5_bitflip_circuit_gates_global_phase_error(
            phi_rate=z_phase_error_rate,
            error_probability=0.1,  # Hardcoded to 10% error probability
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
        dataset["simulated_dataset"] = raw_dataset
        dataset.attrs.update(raw_dataset.attrs)
        dataset.attrs["error_z_rate_per_cycle"] = z_phase_error_rate
        dataset.attrs["error_probability"] = error_probability
        dataset.attrs["error_coherence"] = int(True)
        list_of_datasets.append(dataset)

    # Concatenate along a new, unnamed dimension
    combined_stepwise: xr.Dataset = xr.concat(list_of_datasets, dim="error_z_rate_per_cycle")
    # Assign the coordinate values to the new dimension
    combined_stepwise.coords["error_z_rate_per_cycle"] = z_phase_error_rates
    meas_ctrl.save_dataset_to_netcdf(combined_stepwise, common_path / f"defect_rate_zerror_d5_coherent_t1t2_readout.nc", overwrite=True)
