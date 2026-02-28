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

    noise_file = "setup_noisy_d3_t1t2_readout.yaml"
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

    #angle_drifts = np.linspace(-1e-3, 1e-3, 11) #GHz because time is given in ns
    angle_drifts = np.linspace(0, 1e-3, 11) #GHz because time is given in ns
    common_path: Path = ROOT_DIR / "simulated_datasets"
    qec_rounds: int = 10
    repetitions = 50_000


    list_of_datasets: List[xr.Dataset] = []
    for i, angle_drift in enumerate(angle_drifts):
        model = DiCarloLabModelT2Ratio(
            setup=setup,
            dim_lookup=dim_lookup,
            angle_drift=angle_drift,
        )

        meas_ctrl = MeasurementControl(
            noise_model=model,
            code_settings=CodeSettings.from_code_type(RepetitionCodeType.DISTANCE_3),
            dim_lookup=dim_lookup,
            initial_state_type=InitialStateType.ODD,
        )

        error_probability=0.1
        involved_qubits, gates = meas_ctrl.construct_distance_3_bitflip_circuit_gates_global_phase_error(
            phi_rate=0,
            error_probability=error_probability,
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
        dataset.attrs["error_z_rate_per_cycle"] = angle_drift
        dataset.attrs["error_probability"] = error_probability
        dataset.attrs["error_coherence"] = int(True)
        list_of_datasets.append(dataset)

    # Concatenate along a new, unnamed dimension
    combined_stepwise: xr.Dataset = xr.concat(list_of_datasets, dim="error_z_rate_per_cycle")
    # Assign the coordinate values to the new dimension
    combined_stepwise.coords["error_z_rate_per_cycle"] = angle_drifts
    #meas_ctrl.save_dataset_to_netcdf(combined_stepwise, common_path / f"defect_rate_zerror_d3_stochastic_t1t2_readout.nc", overwrite=True)
    meas_ctrl.save_dataset_to_netcdf(
            combined_stepwise,
            common_path / f"defect_rate_zerror_d3_coherent_t1t2_readout_{np.random.randint(999_999_999)}.nc",
            overwrite=True,
            engine="h5netcdf",
    )
