

if __name__ == '__main__':
    import os
    import numpy as np
    import xarray as xr
    from typing import List, Tuple
    from pathlib import Path
    from quantumsim import Setup
    from quantumsim.circuits.circuit import Circuit
    from quantumsim.models.model_dclab import DiCarloLabModel
    from meas_control_repeated_circuits import MeasurementControl, CodeSettings, RepetitionCodeDistance5, InitialStateType
    from qce_circuit.connectivity.intrf_channel_identifier import QubitIDObj
    from qce_circuit.connectivity.intrf_connectivity_surface_code import IParityGroup, ISurfaceCodeLayer
    from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier
    from qce_interp.visualization.plot_defect_rate import (
        plot_all_defect_rate,
    )
    import matplotlib.pyplot as plt

    noise_file = "setup_noisy.yaml"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = Path(os.path.join(current_dir, ".."))
    setup = Setup.from_file(os.path.join(ROOT_DIR, noise_file))
    model = DiCarloLabModel(setup=setup)

    code_layout: ISurfaceCodeLayer = RepetitionCodeDistance5()
    parity_group: IParityGroup = code_layout.get_parity_group(element=QubitIDObj("Z1"))[0]

    common_path: Path = ROOT_DIR / "simulated_datasets"
    qec_rounds: int = 20
    error_budget_circuits: List[Tuple[str, Circuit, InitialStateType]] = [
        ("defect_rates_baseline", MeasurementControl.construct_detector_only_circuit(model, parity_group, code_layout), InitialStateType.ALL_ZERO),
        ("defect_rates_switching", MeasurementControl.construct_detector_inverse_parity_circuit(model, parity_group, code_layout), InitialStateType.ALL_ZERO),
        ("defect_rates_decoherence", MeasurementControl.construct_decoherence_only_circuit(model, parity_group, code_layout), InitialStateType.ALL_ZERO),
        ("defect_rates_active_zz_data_bitflip", MeasurementControl.construct_relaxation_data_constant_circuit(model, parity_group, code_layout), InitialStateType.ALL_ONES),
        ("defect_rates_full_even_parity", MeasurementControl.construct_parity_circuit(model, parity_group, code_layout), InitialStateType.EVEN),
        ("defect_rates_full_odd_parity", MeasurementControl.construct_parity_circuit(model, parity_group, code_layout), InitialStateType.ODD),
    ]
    list_of_datasets: List[Tuple[str, xr.Dataset]] = []
    for i, (label, cycle_circuit, initial_state_type) in enumerate(error_budget_circuits):
        meas_ctrl = MeasurementControl(
            noise_model=model,
            code_settings=CodeSettings.as_single_stabilizer(parity_group=parity_group),
            initial_state_type=initial_state_type
        )

        error_identifier: ErrorDetectionIdentifier = meas_ctrl.construct_error_identifier(
            cycle_circuit=cycle_circuit,
            depth=qec_rounds,
            repetitions=10_000,
            **dict(
                inverse_parity=True if label == "inverse_parity" else False,
                stabilizer_active=True if label in ["full_parity_even", "full_parity_odd"] else False,
            )
        )

        dataset = meas_ctrl.construct_dataset(
            error_identifier=error_identifier,
            noise_file_path=ROOT_DIR / noise_file,
        )
        dataset.attrs["circuit_type"] = label
        list_of_datasets.append((label, dataset))

        # print(dataset)
        # meas_ctrl.save_dataset_to_netcdf(dataset, common_path / f"{parity_group.ancilla_id.id}_{label}.nc", overwrite=True)
        # fig, ax = plot_all_defect_rate(
        #     error_identifier=error_identifier,
        #     included_rounds=qec_rounds,
        # )

    data_vars_for_new_dataset: dict = {
        new_name: source_dataset[f"defect_rates_{parity_group.ancilla_id.id}"]
        for new_name, source_dataset in list_of_datasets
    }
    final_dataset: xr.Dataset = xr.Dataset(data_vars_for_new_dataset)
    MeasurementControl.save_dataset_to_netcdf(final_dataset, common_path / f"defect_rate_{parity_group.ancilla_id.id}_error_budget.nc", overwrite=True)

    # plt.show()
