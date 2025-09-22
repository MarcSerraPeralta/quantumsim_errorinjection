import warnings

from quantumsim.operations.repetition_tree import RepeatedCircuitEvaluation

warnings.filterwarnings("ignore")
import os
import numpy as np
from typing import List
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
from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier
from qce_interp.utilities.expected_parities import initial_state_to_expected_parity
from qce_interp.interface_definitions.intrf_state_classification import (
    StateClassifierContainer,
    ParityType,
)

np.set_printoptions(precision=4, suppress=True)


class RepetitionCodeDistance3(GenericSurfaceCode, IGenericSurfaceCodeLayer, metaclass=SingletonABCMeta):

    # region Class Constructor
    def __init__(self):
        super().__init__(
            gate_sequences=[
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj('D1')),
                    ],
                    _gate_operations=[
                        Operation.type_gate(EdgeIDObj(QubitIDObj('X1'), QubitIDObj('D2'))),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj('D2')),
                    ],
                    _gate_operations=[
                        Operation.type_gate(EdgeIDObj(QubitIDObj('X1'), QubitIDObj('D1'))),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj('D2')),
                    ],
                    _gate_operations=[
                        Operation.type_gate(EdgeIDObj(QubitIDObj('Z1'), QubitIDObj('D1'))),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj('Z3')),
                        Operation.type_park(QubitIDObj('X3')),
                    ],
                    _gate_operations=[
                        Operation.type_gate(EdgeIDObj(QubitIDObj('D4'), QubitIDObj('Z1'))),
                    ],
                ),
            ],
            parity_group_z=[
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
            ],
            parity_group_x=[],
        )
    # endregion


def construct_error_identifier(
    result_tensor: np.ndarray,
    depth: int,
    repetitions: int,
    code_layout: ISurfaceCodeLayer,
    involved_data_qubit_ids: List[IQubitID],  # Ordered
    involved_ancilla_qubit_ids: List[IQubitID],
) -> ErrorDetectionIdentifier:
    expected_parity = initial_state_to_expected_parity(
        initial_state=InitialStateContainer.from_ordered_list(
            initial_states=[
                InitialStateEnum.ONE if (i % 2 == 1) else InitialStateEnum.ZERO
                for i, qubit_id in enumerate(involved_data_qubit_ids)
            ]
        ),
        parity_layout=code_layout,
        involved_data_qubit_ids=involved_data_qubit_ids,
        involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
        inverse_parity=False,
        stabilizer_active=True,
    )

    return ErrorDetectionIdentifier(
        classifier_lookup={
            qubit_id: StateClassifierContainer(
                state_classification=result_tensor[:, :, i].flatten(),
                _expected_parity=ParityType.EVEN if qubit_id not in expected_parity else expected_parity[qubit_id],
                _stabilizer_reset=False,
            )
            for i, qubit_id in enumerate(involved_data_qubit_ids + involved_ancilla_qubit_ids)
        },
        index_kernel=SimulatedRepetitionExperimentKernel(
            rounds=[depth],
            involved_data_qubit_ids=involved_data_qubit_ids,
            involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
            experiment_repetitions=repetitions,
        ),
        involved_qubit_ids=involved_data_qubit_ids + involved_ancilla_qubit_ids,
        device_layout=code_layout,
        qec_rounds=[depth],
        use_heralded_post_selection=False,
        use_projected_leakage_post_selection=False,
        use_stabilizer_leakage_post_selection=False,
        post_selection_qubits=None,
        use_computational_parity=True,
    )

###################
# Parameters


noise_file = "setup_noisy.yaml"
# noise_file = "setup_conditional_oscillation.yaml"
# noise_file = "setup_perfect.yaml"

qubit_d4 = "D4"
qubit_z1 = "Z1"
qubit_d1 = "D1"
qubit_x1 = "X1"
qubit_d2 = "D2"
qubits = [qubit_d4, qubit_z1, qubit_d1, qubit_x1, qubit_d2]

PLOT_CIRCUITS = True

###################

current_dir = os.path.dirname(os.path.abspath(__file__))
setup = Setup.from_file(os.path.join(current_dir, "..", noise_file))
model = DiCarloLabModel(setup=setup)
state_projector = StateProjector(
    ordered_qubit_ids=qubits,
    meas_qubit_ids=[qubit_z1, qubit_x1],
    dim_per_qubit={
        qubit_z1: 3,
        qubit_x1: 3,
    },
    model=model,
)
final_state_projector = StateProjector(
    ordered_qubit_ids=qubits,
    meas_qubit_ids=qubits,
    dim_per_qubit={
        qubit_id: 2
        for qubit_id in qubits
    },
    model=model,
)
t_sqg: float = 20
t_tqg: float = 60


def distance_3_bitflip_gates():
    gates = []
    t_ref: float = 0
    # Start of Parities
    gates.extend([
        model.idle_sq_gate(qubit_d4, t_start=t_ref),
        model.idle_sq_gate(qubit_z1, t_start=t_ref),
        model.idle_sq_gate(qubit_d1, t_start=t_ref),
        model.rotate_y_90(qubit_x1, t_start=t_ref),
        model.idle_sq_gate(qubit_d2, t_start=t_ref),
    ])
    t_ref += t_sqg
    # X1-D2
    gates.extend([
        model.idle_tq_gate(qubit_d4, t_start=t_ref),
        model.idle_tq_gate(qubit_z1, t_start=t_ref),
        model.idle_tq_gate(qubit_d1, t_start=t_ref),
        model.cphase(qubit_x1, qubit_d2, t_start=t_ref),
    ])
    t_ref += t_tqg
    # X1-D1
    gates.extend([
        model.idle_tq_gate(qubit_d4, t_start=t_ref),
        model.idle_tq_gate(qubit_z1, t_start=t_ref),
        model.idle_tq_gate(qubit_d2, t_start=t_ref),
        model.cphase(qubit_x1, qubit_d1, t_start=t_ref),
    ])
    t_ref += t_tqg
    # Middle of Parities
    gates.extend([
        model.idle_sq_gate(qubit_d4, t_start=t_ref),
        model.rotate_y_90(qubit_z1, t_start=t_ref),
        model.idle_sq_gate(qubit_d1, t_start=t_ref),
        model.rotate_y_m90(qubit_x1, t_start=t_ref),
        model.idle_sq_gate(qubit_d2, t_start=t_ref),
    ])
    t_ref += t_sqg
    # Z1-D1
    gates.extend([
        model.idle_tq_gate(qubit_d4, t_start=t_ref),
        model.idle_tq_gate(qubit_x1, t_start=t_ref),
        model.idle_tq_gate(qubit_d2, t_start=t_ref),
        model.cphase(qubit_z1, qubit_d1, t_start=t_ref),
    ])
    t_ref += t_tqg
    # Z1-D4
    gates.extend([
        model.idle_tq_gate(qubit_d1, t_start=t_ref),
        model.idle_tq_gate(qubit_x1, t_start=t_ref),
        model.idle_tq_gate(qubit_d2, t_start=t_ref),
        model.cphase(qubit_z1, qubit_d4, t_start=t_ref),
    ])
    t_ref += t_tqg
    # End of Parities
    gates.extend([
        model.idle_sq_gate(qubit_d4, t_start=t_ref),
        model.rotate_y_m90(qubit_z1, t_start=t_ref),
        model.idle_sq_gate(qubit_d1, t_start=t_ref),
        model.idle_sq_gate(qubit_x1, t_start=t_ref),
        model.idle_sq_gate(qubit_d2, t_start=t_ref),
    ])
    t_ref += t_sqg
    # Refocusing operations
    gates.extend([
        model.rotate_x_180(qubit_d4, t_start=t_ref),
        model.rotate_x_180(qubit_d1, t_start=t_ref),
        model.rotate_x_180(qubit_d2, t_start=t_ref),
    ])

    return gates

# gates = [
#     model.idle_sq_gate(qubit_d4, t_start=0),
#     model.idle_sq_gate(qubit_z1, t_start=0),
#     model.idle_sq_gate(qubit_d1, t_start=0),
#     model.idle_sq_gate(qubit_x1, t_start=0),
#     model.idle_sq_gate(qubit_d2, t_start=0),
#
#     model.idle_sq_gate(qubit_d4, t_start=t_sqg),
#     model.idle_sq_gate(qubit_z1, t_start=t_sqg),
#     model.idle_sq_gate(qubit_d1, t_start=t_sqg),
#     model.idle_sq_gate(qubit_x1, t_start=t_sqg),
#     model.idle_sq_gate(qubit_d2, t_start=t_sqg),
# ]

circuit = Circuit(qubits=qubits, gates=distance_3_bitflip_gates())
finalized_circuit = model.finalize(circuit)
state = State(qubits, dim=3)  # initialization in ground state

repetition_tree = RepeatedCircuitEvaluation(
    initial_state=state,
    circuit=finalized_circuit,
    state_projector=state_projector,
    final_state_projector=final_state_projector,
)
qec_rounds: int = 10
repetitions: int = 10
result_tensor = repetition_tree.evaluate_as_array(
    depth=qec_rounds,
    repetitions=repetitions,
)

error_identifier = construct_error_identifier(
    result_tensor=result_tensor,
    depth=qec_rounds,
    repetitions=repetitions,
    code_layout=RepetitionCodeDistance3(),
    involved_data_qubit_ids=[
        QubitIDObj('D4'),
        QubitIDObj('D1'),
        QubitIDObj('D2'),
    ],
    involved_ancilla_qubit_ids=[
        QubitIDObj('Z1'),
        QubitIDObj('X1'),
    ],
)
print(error_identifier)

# finalized_circuit @ state
#
# print(state_projector.output_state_vector)
# outcome_vector = state_projector.construct_output_vector(state=state)
# print(outcome_vector)
