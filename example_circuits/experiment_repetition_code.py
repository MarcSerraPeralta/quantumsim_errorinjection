import warnings
from quantumsim.operations.repetition_tree import RepeatedCircuitEvaluation
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
from qce_interp.visualization.plot_defect_rate import (
    plot_all_defect_rate,
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
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
    involved_qubits_qubit_ids: List[IQubitID],  # Ordered
) -> ErrorDetectionIdentifier:
    involved_data_qubit_ids: List[IQubitID] = [qubit_id for qubit_id in involved_qubits_qubit_ids if qubit_id in code_layout.data_qubit_ids]
    involved_ancilla_qubit_ids: List[IQubitID] = [qubit_id for qubit_id in involved_qubits_qubit_ids if qubit_id in code_layout.ancilla_qubit_ids]
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

    classifier_lookup = {}
    for i, qubit_id in enumerate(involved_qubits_qubit_ids):
        state_classification = result_tensor[:, :, i].flatten()
        state_classification[state_classification == 2] = 1  # Map 0, 1, 2 outcomes to 0, 1

        classifier_lookup[qubit_id] = StateClassifierContainer(
            state_classification=state_classification,
            _expected_parity=ParityType.EVEN if qubit_id not in expected_parity else expected_parity[qubit_id],
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
# final_state_projector = StateProjector(
#     ordered_qubit_ids=qubits,
#     meas_qubit_ids=qubits,
#     dim_per_qubit={
#         qubit_id: 3
#         for qubit_id in qubits
#     },
#     model=model,
# )
final_state_projector = StateProjector(
    ordered_qubit_ids=qubits,
    meas_qubit_ids=[qubit_d4, qubit_d1, qubit_d2],
    dim_per_qubit={
        qubit_id: 3
        for qubit_id in [qubit_d4, qubit_d1, qubit_d2]
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
        model.cphase(qubit_d2, qubit_x1, t_start=t_ref),
    ])
    t_ref += t_tqg
    # X1-D1
    gates.extend([
        model.idle_tq_gate(qubit_d4, t_start=t_ref),
        model.idle_tq_gate(qubit_z1, t_start=t_ref),
        model.idle_tq_gate(qubit_d2, t_start=t_ref),
        model.cphase(qubit_d1, qubit_x1, t_start=t_ref),
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
        model.cphase(qubit_d1, qubit_z1, t_start=t_ref),
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
        model.idle_sq_gate(qubit_z1, t_start=t_ref),
        model.rotate_x_180(qubit_d1, t_start=t_ref),
        model.idle_sq_gate(qubit_x1, t_start=t_ref),
        model.rotate_x_180(qubit_d2, t_start=t_ref),
    ])

    return gates


def prep_initial_state_gates():
    return [
        model.idle_sq_gate(qubit_d4, t_start=0),
        model.idle_sq_gate(qubit_z1, t_start=0),
        model.rotate_x_180(qubit_d1, t_start=0),
        model.idle_sq_gate(qubit_x1, t_start=0),
        model.idle_sq_gate(qubit_d2, t_start=0),
    ]

# gates = []
# t_ref: float = 0
# # Start of Parities
# gates.extend([
#     model.idle_sq_gate(qubit_d4, t_start=t_ref),
#     model.rotate_x_180(qubit_z1, t_start=t_ref),
#     model.idle_sq_gate(qubit_d1, t_start=t_ref),
#     model.rotate_x_180(qubit_x1, t_start=t_ref),
#     model.idle_sq_gate(qubit_d2, t_start=t_ref),
# ])


initial_circuit = Circuit(qubits=qubits, gates=prep_initial_state_gates())
circuit = Circuit(qubits=qubits, gates=distance_3_bitflip_gates())  # gates)  #
finalized_circuit = model.finalize(circuit)
finalized_circuit = finalized_circuit(angle=180)
state = State(qubits, dim=3)  # initialization in ground state
model.finalize(initial_circuit) @ state

repetition_tree = RepeatedCircuitEvaluation(
    initial_state=state,
    circuit=finalized_circuit,
    state_projector=state_projector,
    final_state_projector=final_state_projector,
)
qec_rounds: int = 7
repetitions: int = 10000
result_tensor = repetition_tree.evaluate_as_array(
    depth=qec_rounds + 1,
    repetitions=repetitions,
    verbose=True,
)

error_identifier = construct_error_identifier(
    result_tensor=result_tensor,
    depth=qec_rounds,
    repetitions=repetitions,
    code_layout=RepetitionCodeDistance3(),
    involved_qubits_qubit_ids=[
        QubitIDObj('D4'),
        QubitIDObj('Z1'),
        QubitIDObj('D1'),
        QubitIDObj('X1'),
        QubitIDObj('D2'),
    ],
)
print("D4", error_identifier._classifier_lookup[QubitIDObj('D4')].state_classification[:qec_rounds])
print("Z1", error_identifier._classifier_lookup[QubitIDObj('Z1')].state_classification[:qec_rounds])
print("D1", error_identifier._classifier_lookup[QubitIDObj('D1')].state_classification[:qec_rounds])
print("X1", error_identifier._classifier_lookup[QubitIDObj('X1')].state_classification[:qec_rounds])
print("D2", error_identifier._classifier_lookup[QubitIDObj('D2')].state_classification[:qec_rounds])
print(np.unique(error_identifier.get_binary_stabilizer_classification(cycle_stabilizer_count=qec_rounds)))
fig, ax = plot_all_defect_rate(
    error_identifier=error_identifier,
    included_rounds=qec_rounds,
)
# ax.set_ylim(-1, +1)
plt.show()

# finalized_circuit @ state
#
# print(state_projector.output_state_vector)
# outcome_vector = state_projector.construct_output_vector(state=state)
# print(outcome_vector)
