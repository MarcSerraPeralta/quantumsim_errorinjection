import warnings
from quantumsim.operations.repetition_tree import LazyTreeCircuitEvaluation, RepeatedCircuitEvaluation
import os
import numpy as np
from typing import List, Any
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from quantumsim.circuits import Circuit
from quantumsim import Setup
from quantumsim.states.state import State
from quantumsim.models import DiCarloLabModel
from quantumsim.operations.state_projector import StateProjector
from qce_circuit.language.intrf_declarative_circuit import (
    InitialStateEnum,
    InitialStateContainer,
)
from qce_circuit.structure.acquisition_indexing.kernel_repetition_code import (
    SimulatedRepetitionExperimentKernel,
)
from qce_circuit.utilities.singleton_base import SingletonABCMeta
from qce_circuit.connectivity.intrf_channel_identifier import (
    IQubitID,
    QubitIDObj,
    EdgeIDObj,
)
from qce_circuit.connectivity.intrf_connectivity_gate_sequence import (
    GateSequenceLayer,
    Operation,
)
from qce_circuit.connectivity.intrf_connectivity_surface_code import (
    ISurfaceCodeLayer,
    StabilizerType,
)
from qce_circuit.connectivity.connectivity_surface_code import ParityGroup
from qce_circuit.connectivity.generic_gate_sequence import (
    IGenericSurfaceCodeLayer,
    GenericSurfaceCode,
)
from qce_interp.interface_definitions.intrf_error_identifier import (
    ErrorDetectionIdentifier,
)
from qce_interp.utilities.expected_parities import initial_state_to_expected_parity
from qce_interp.utilities.serialize_error_identifier import construct_processed_dataset
from qce_interp.interface_definitions.intrf_state_classification import (
    StateClassifierContainer,
    ParityType,
)
from qce_interp.visualization.plot_defect_rate import (
    plot_all_defect_rate,
)
from quantumsim.circuits.plotter import plot
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)


class RepetitionCodeDistance3(
    GenericSurfaceCode, IGenericSurfaceCodeLayer, metaclass=SingletonABCMeta
):

    # region Class Constructor
    def __init__(self):
        super().__init__(
            gate_sequences=[
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj("D1")),
                    ],
                    _gate_operations=[
                        Operation.type_gate(
                            EdgeIDObj(QubitIDObj("X1"), QubitIDObj("D2"))
                        ),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj("D2")),
                    ],
                    _gate_operations=[
                        Operation.type_gate(
                            EdgeIDObj(QubitIDObj("X1"), QubitIDObj("D1"))
                        ),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj("D2")),
                    ],
                    _gate_operations=[
                        Operation.type_gate(
                            EdgeIDObj(QubitIDObj("Z1"), QubitIDObj("D1"))
                        ),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj("Z3")),
                        Operation.type_park(QubitIDObj("X3")),
                    ],
                    _gate_operations=[
                        Operation.type_gate(
                            EdgeIDObj(QubitIDObj("D4"), QubitIDObj("Z1"))
                        ),
                    ],
                ),
            ],
            parity_group_z=[
                ParityGroup(
                    _parity_type=StabilizerType.STABILIZER_Z,
                    _ancilla_qubit=QubitIDObj("X1"),
                    _data_qubits=[QubitIDObj("D1"), QubitIDObj("D2")],
                ),
                ParityGroup(
                    _parity_type=StabilizerType.STABILIZER_Z,
                    _ancilla_qubit=QubitIDObj("Z1"),
                    _data_qubits=[QubitIDObj("D4"), QubitIDObj("D1")],
                ),
            ],
            parity_group_x=[],
        )

    # endregion


class RepetitionCodeDistance5(
    GenericSurfaceCode, IGenericSurfaceCodeLayer, metaclass=SingletonABCMeta
):
    # region Class Constructor
    def __init__(self):
        super().__init__(
            gate_sequences=[
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj('D1')),
                        Operation.type_park(QubitIDObj('Z1')),
                        Operation.type_park(QubitIDObj('X3')),
                    ],
                    _gate_operations=[
                        Operation.type_gate(EdgeIDObj(QubitIDObj('X1'), QubitIDObj('D2'))),
                        Operation.type_gate(EdgeIDObj(QubitIDObj('Z3'), QubitIDObj('D4'))),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj('D2')),
                    ],
                    _gate_operations=[
                        Operation.type_gate(EdgeIDObj(QubitIDObj('X1'), QubitIDObj('D1'))),
                        Operation.type_gate(EdgeIDObj(QubitIDObj('Z3'), QubitIDObj('D7'))),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj('Z3')),
                        Operation.type_park(QubitIDObj('X3')),
                        Operation.type_park(QubitIDObj('D3')),
                    ],
                    _gate_operations=[
                        Operation.type_gate(EdgeIDObj(QubitIDObj('D4'), QubitIDObj('Z1'))),
                        Operation.type_gate(EdgeIDObj(QubitIDObj('X2'), QubitIDObj('D2'))),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj('D2')),
                    ],
                    _gate_operations=[
                        Operation.type_gate(EdgeIDObj(QubitIDObj('Z1'), QubitIDObj('D1'))),
                        Operation.type_gate(EdgeIDObj(QubitIDObj('X2'), QubitIDObj('D3'))),
                    ],
                ),
            ],
            parity_group_z=[
                ParityGroup(
                    _parity_type=StabilizerType.STABILIZER_Z,
                    _ancilla_qubit=QubitIDObj('X2'),
                    _data_qubits=[QubitIDObj('D2'), QubitIDObj('D3')]
                ),
                ParityGroup(
                    _parity_type=StabilizerType.STABILIZER_Z,
                    _ancilla_qubit=QubitIDObj('X1'),
                    _data_qubits=[QubitIDObj('D1'), QubitIDObj('D2')]
                ),
                ParityGroup(
                    _parity_type=StabilizerType.STABILIZER_Z,
                    _ancilla_qubit=QubitIDObj('Z1'),
                    _data_qubits=[QubitIDObj('D4'), QubitIDObj('D1')]
                ),
                ParityGroup(
                    _parity_type=StabilizerType.STABILIZER_Z,
                    _ancilla_qubit=QubitIDObj('Z3'),
                    _data_qubits=[QubitIDObj('D4'), QubitIDObj('D7')]
                ),
            ],
            parity_group_x=[],
        )
    # endregion


def construct_error_identifier(
    result_tensor: np.ndarray,
    depth: int,
    repetitions: int,
    initial_state: InitialStateContainer,
    code_layout: ISurfaceCodeLayer,
    involved_qubits_qubit_ids: List[IQubitID],  # Ordered
) -> ErrorDetectionIdentifier:
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
        inverse_parity=False,
        stabilizer_active=True,
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
    noise_file_path: Path,
) -> xr.Dataset:
    dataset: xr.Dataset = construct_processed_dataset(
        error_identifier=error_identifier,
        initial_state=initial_state,
        qec_rounds=error_identifier.qec_rounds,
        code_layout=code_layout,
    )
    with open(noise_file_path, 'r') as file:
        # config_data = yaml.safe_load(file)
        config_data_str = file.read()
    dataset.attrs["noise_config"] = config_data_str

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


noise_file = "setup_noisy.yaml"
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(os.path.join(current_dir, ".."))
# noise_file = "setup_perfect.yaml"
DISTANCE = 3

# Ryzen 7 7800X3D (4.2 GHz); d=3, r=10, seed=10.000 => 3min
qec_rounds: int = 10
repetitions: int = 1000  # 10_000
state_dim: int = 3

PLOT_CIRCUITS = False

if DISTANCE == 3:
    qubits = ["D4", "Z1", "D1", "X1", "D2"]
elif DISTANCE == 5:
    qubits = ["D7", "Z3", "D4", "Z1", "D1", "X1", "D2", "X2", "D3"]
else:
    raise ValueError(f"Distance {DISTANCE} not implemented.")

setup = Setup.from_file(os.path.join(ROOT_DIR, noise_file))
model = DiCarloLabModel(setup=setup)
state_projector = StateProjector(
    ordered_qubit_ids=qubits,
    meas_qubit_ids=qubits[1::2],
    dim_per_qubit={anc: 3 for anc in qubits[1::2]},
    model=model,
)
final_state_projector = StateProjector(
    ordered_qubit_ids=qubits,
    meas_qubit_ids=qubits[::2],
    dim_per_qubit={data: 3 for data in qubits[::2]},
    model=model,
)
t_sqg: float = 20
t_tqg: float = 60


def distance_3_bitflip_gates():
    gates = []
    t_ref: float = 0
    # Start of Parities
    gates.extend(
        [
            model.idle_sq_gate("D4", t_start=t_ref),
            model.idle_sq_gate("Z1", t_start=t_ref),
            model.idle_sq_gate("D1", t_start=t_ref),
            model.rotate_y_90("X1", t_start=t_ref),
            model.idle_sq_gate("D2", t_start=t_ref),
        ]
    )
    t_ref += t_sqg
    # X1-D2
    gates.extend(
        [
            model.idle_tq_gate("D4", t_start=t_ref),
            model.idle_tq_gate("Z1", t_start=t_ref),
            model.idle_tq_gate("D1", t_start=t_ref),
            model.cphase("D2", "X1", t_start=t_ref),
            model.zz_coupling_2q("D1", "X1", t_start=t_ref),
        ]
    )
    t_ref += t_tqg
    # X1-D1
    gates.extend(
        [
            model.idle_tq_gate("D4", t_start=t_ref),
            model.idle_tq_gate("Z1", t_start=t_ref),
            model.idle_tq_gate("D2", t_start=t_ref),
            model.cphase("D1", "X1", t_start=t_ref),
            model.zz_coupling_2q("D2", "X1", t_start=t_ref),
        ]
    )
    t_ref += t_tqg
    # Middle of Parities
    gates.extend(
        [
            model.idle_sq_gate("D4", t_start=t_ref),
            model.rotate_y_90("Z1", t_start=t_ref),
            model.idle_sq_gate("D1", t_start=t_ref),
            model.rotate_y_m90("X1", t_start=t_ref),
            model.idle_sq_gate("D2", t_start=t_ref),
        ]
    )
    t_ref += t_sqg
    # Z1-D1
    gates.extend(
        [
            model.idle_tq_gate("D4", t_start=t_ref),
            model.idle_tq_gate("X1", t_start=t_ref),
            model.idle_tq_gate("D2", t_start=t_ref),
            model.cphase("D1", "Z1", t_start=t_ref),
            model.zz_coupling_2q("Z1", "D4", t_start=t_ref),
            model.zz_coupling_2q("D2", "Z1", t_start=t_ref),
        ]
    )
    t_ref += t_tqg
    # Z1-D4
    gates.extend(
        [
            model.idle_tq_gate("D1", t_start=t_ref),
            model.idle_tq_gate("X1", t_start=t_ref),
            model.idle_tq_gate("D2", t_start=t_ref),
            model.cphase("Z1", "D4", t_start=t_ref),
            model.zz_coupling_2q("D1", "Z1", t_start=t_ref),
            model.zz_coupling_2q("D2", "Z1", t_start=t_ref),
        ]
    )
    t_ref += t_tqg
    # End of Parities
    gates.extend(
        [
            model.idle_sq_gate("D4", t_start=t_ref),
            model.rotate_y_m90("Z1", t_start=t_ref),
            model.idle_sq_gate("D1", t_start=t_ref),
            model.idle_sq_gate("X1", t_start=t_ref),
            model.idle_sq_gate("D2", t_start=t_ref),
        ]
    )
    t_ref += t_sqg
    # Refocusing operations
    gates.extend(
        [
            model.rotate_x_180("D4", t_start=t_ref),
            model.idle_sq_gate("Z1", t_start=t_ref),
            model.rotate_x_180("D1", t_start=t_ref),
            model.idle_sq_gate("X1", t_start=t_ref),
            model.rotate_x_180("D2", t_start=t_ref),
        ]
    )

    return gates


def distance_5_bitflip_gates():
    # picture from Sean's WhatsApp conversation
    gates = []
    t_ref: float = 0
    gates.extend(
        [
            model.idle_sq_gate("D7", t_start=t_ref),
            model.idle_sq_gate("D4", t_start=t_ref),
            model.idle_sq_gate("D1", t_start=t_ref),
            model.idle_sq_gate("D2", t_start=t_ref),
            model.idle_sq_gate("D3", t_start=t_ref),
            model.rotate_y_90("Z3", t_start=t_ref),
            model.idle_sq_gate("Z1", t_start=t_ref),
            model.rotate_y_90("X1", t_start=t_ref),
            model.idle_sq_gate("X2", t_start=t_ref),
        ]
    )
    t_ref += t_sqg

    gates.extend(
        [
            model.idle_tq_gate("D7", t_start=t_ref),
            model.cphase("Z3", "D4", t_start=t_ref),
            model.idle_tq_gate("D1", t_start=t_ref),
            model.cphase("D2", "X1", t_start=t_ref),
            model.idle_tq_gate("D3", t_start=t_ref),
            model.idle_tq_gate("Z1", t_start=t_ref),
            model.idle_tq_gate("X2", t_start=t_ref),
            model.zz_coupling_2q("Z3", "D7", t_start=t_ref),
            model.zz_coupling_2q("X1", "D1", t_start=t_ref),
        ]
    )
    t_ref += t_tqg

    gates.extend(
        [
            model.cphase("D7", "Z3", t_start=t_ref),
            model.idle_tq_gate("D4", t_start=t_ref),
            model.cphase("D1", "X1", t_start=t_ref),
            model.idle_tq_gate("D2", t_start=t_ref),
            model.idle_tq_gate("D3", t_start=t_ref),
            model.idle_tq_gate("Z1", t_start=t_ref),
            model.idle_tq_gate("X2", t_start=t_ref),
            model.zz_coupling_2q("Z3", "D4", t_start=t_ref),
            model.zz_coupling_2q("X1", "D2", t_start=t_ref),
        ]
    )
    t_ref += t_tqg

    gates.extend(
        [
            model.idle_sq_gate("D7", t_start=t_ref),
            model.idle_sq_gate("D4", t_start=t_ref),
            model.idle_sq_gate("D1", t_start=t_ref),
            model.idle_sq_gate("D2", t_start=t_ref),
            model.idle_sq_gate("D3", t_start=t_ref),
            model.rotate_y_90("Z3", t_start=t_ref),
            model.rotate_y_90("Z1", t_start=t_ref),
            model.rotate_y_90("X1", t_start=t_ref),
            model.rotate_y_90("X2", t_start=t_ref),
        ]
    )
    t_ref += t_sqg

    gates.extend(
        [
            model.idle_tq_gate("D7", t_start=t_ref),
            model.cphase("Z1", "D4", t_start=t_ref),
            model.idle_tq_gate("D1", t_start=t_ref),
            model.cphase("D2", "X2", t_start=t_ref),
            model.idle_tq_gate("D3", t_start=t_ref),
            model.idle_tq_gate("Z3", t_start=t_ref),
            model.idle_tq_gate("X1", t_start=t_ref),
            model.zz_coupling_2q("Z1", "D1", t_start=t_ref),
            model.zz_coupling_2q("X2", "D3", t_start=t_ref),
        ]
    )
    t_ref += t_tqg

    gates.extend(
        [
            model.idle_tq_gate("D7", t_start=t_ref),
            model.idle_tq_gate("D4", t_start=t_ref),
            model.cphase("D1", "Z1", t_start=t_ref),
            model.idle_tq_gate("D2", t_start=t_ref),
            model.cphase("D3", "X2", t_start=t_ref),
            model.idle_tq_gate("Z3", t_start=t_ref),
            model.idle_tq_gate("X1", t_start=t_ref),
            model.zz_coupling_2q("Z1", "D4", t_start=t_ref),
            model.zz_coupling_2q("X2", "D2", t_start=t_ref),
        ]
    )
    t_ref += t_tqg

    gates.extend(
        [
            model.idle_sq_gate("D7", t_start=t_ref),
            model.idle_sq_gate("D4", t_start=t_ref),
            model.idle_sq_gate("D1", t_start=t_ref),
            model.idle_sq_gate("D2", t_start=t_ref),
            model.idle_sq_gate("D3", t_start=t_ref),
            model.idle_sq_gate("Z3", t_start=t_ref),
            model.rotate_y_90("Z1", t_start=t_ref),
            model.idle_sq_gate("X1", t_start=t_ref),
            model.rotate_y_90("X2", t_start=t_ref),
        ]
    )
    t_ref += t_sqg

    gates.extend(
        [
            model.rotate_x_180("D7", t_start=t_ref),
            model.rotate_x_180("D4", t_start=t_ref),
            model.rotate_x_180("D1", t_start=t_ref),
            model.rotate_x_180("D2", t_start=t_ref),
            model.rotate_x_180("D3", t_start=t_ref),
            model.idle_sq_gate("Z3", t_start=t_ref),
            model.idle_sq_gate("Z1", t_start=t_ref),
            model.idle_sq_gate("X1", t_start=t_ref),
            model.idle_sq_gate("X2", t_start=t_ref),
        ]
    )

    return gates


def distance_3_prep_initial_state_gates():
    return [
        model.idle_sq_gate("D4", t_start=0),
        model.idle_sq_gate("Z1", t_start=0),
        model.rotate_x_180("D1", t_start=0),
        model.idle_sq_gate("X1", t_start=0),
        model.idle_sq_gate("D2", t_start=0),
    ]


def distance_5_prep_initial_state_gates():
    return [model.idle_sq_gate(qubit, t_start=0) for qubit in qubits]


if DISTANCE == 3:
    init_gates = distance_3_prep_initial_state_gates()
    qec_gates = distance_3_bitflip_gates()
elif DISTANCE == 5:
    init_gates = distance_5_prep_initial_state_gates()
    qec_gates = distance_5_bitflip_gates()
else:
    raise ValueError(f"Distance {DISTANCE} not implemented.")

initial_circuit = model.finalize(Circuit(qubits=qubits, gates=init_gates))
qec_circuit = model.finalize(Circuit(qubits=qubits, gates=qec_gates))

if PLOT_CIRCUITS:
    plot(Circuit(qubits=qubits, gates=qec_gates))
    plt.show()

state = State(qubits, dim=state_dim, force=False)  # initialization in ground state
initial_circuit @ state
# repetition_tree = LazyTreeCircuitEvaluation(
#     initial_state=state,
#     circuit=qec_circuit,
#     state_projector=state_projector,
#     final_state_projector=final_state_projector,
#     max_depth=qec_rounds,
# )
repetition_tree = RepeatedCircuitEvaluation(
    initial_state=state,
    circuit=qec_circuit,
    state_projector=state_projector,
    final_state_projector=final_state_projector,
)
result_tensor = repetition_tree.evaluate_as_array(
    model=model,
    depth=qec_rounds,
    repetitions=repetitions,
    verbose=True,
)
#
# if DISTANCE == 3:
#     code_layout = RepetitionCodeDistance3()
# elif DISTANCE == 5:
#     code_layout = RepetitionCodeDistance5()
# else:
#     raise ValueError(f"Distance {DISTANCE} not implemented.")
#
# involved_qubits_qubit_ids = [QubitIDObj(q) for q in qubits]
# initial_state = InitialStateContainer.from_ordered_list(
#     initial_states=[
#         InitialStateEnum.ONE if (i % 2 == 1) else InitialStateEnum.ZERO
#         for i, qubit_id in enumerate([qubit_id for qubit_id in involved_qubits_qubit_ids if qubit_id in code_layout.data_qubit_ids])
#     ]
# )
# error_identifier = construct_error_identifier(
#     result_tensor=result_tensor,
#     depth=qec_rounds,
#     repetitions=repetitions,
#     code_layout=code_layout,
#     initial_state=initial_state,
#     involved_qubits_qubit_ids=involved_qubits_qubit_ids,
# )
# dataset = error_identifier_to_dataset(
#     error_identifier=error_identifier,
#     initial_state=initial_state,
#     code_layout=code_layout,
#     noise_file_path=ROOT_DIR / noise_file,
# )
# print(dataset)
# common_path: Path = ROOT_DIR / "simulated_datasets"
# save_dataset_to_netcdf(dataset, common_path / "defect_rate_d3.nc", overwrite=True)
#
# for qubit in qubits:
#     print(
#         qubit,
#         error_identifier._classifier_lookup[QubitIDObj(qubit)].state_classification[
#             :qec_rounds
#         ],
#     )
# fig, ax = plot_all_defect_rate(
#     error_identifier=error_identifier,
#     included_rounds=qec_rounds,
# )
#
# plt.show()
