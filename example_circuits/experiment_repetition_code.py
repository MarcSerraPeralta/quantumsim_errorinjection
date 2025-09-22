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
from qce_circuit.connectivity.intrf_connectivity_surface_code import ISurfaceCodeLayer, IQubitID
from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier
from qce_interp.utilities.expected_parities import initial_state_to_expected_parity
from qce_interp.interface_definitions.intrf_state_classification import (
    StateClassifierContainer,
    ParityType,
)

np.set_printoptions(precision=4, suppress=True)


def construct_error_identifier(
    result_tensor: np.ndarray,
    depth: int,
    repetitions: int,
    code_layout: ISurfaceCodeLayer,
    involved_data_qubit_ids: List[IQubitID],
    involved_ancilla_qubit_ids: List[IQubitID],
) -> ErrorDetectionIdentifier:
    expected_parity = initial_state_to_expected_parity(
        initial_state=InitialStateContainer.from_ordered_list(
            initial_states=[
                InitialStateEnum.ONE if (i % 2 == 1) and not schedule_settings.initial_state_even else InitialStateEnum.ZERO
                for i, qubit_id in enumerate(noise_settings.involved_data_qubit_ids)
            ]
        ),
        parity_layout=code_layout,
        involved_data_qubit_ids=noise_settings.involved_data_qubit_ids,
        involved_ancilla_qubit_ids=noise_settings.involved_ancilla_qubit_ids,
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
            for i, qubit_id in enumerate(noise_settings.involved_qubit_ids)
        },
        index_kernel=SimulatedRepetitionExperimentKernel(
            rounds=[schedule_settings.qec_round],
            involved_data_qubit_ids=noise_settings.involved_data_qubit_ids,
            involved_ancilla_qubit_ids=noise_settings.involved_ancilla_qubit_ids,
            experiment_repetitions=schedule_settings.repetitions,
        ),
        involved_qubit_ids=noise_settings.involved_qubit_ids,
        device_layout=noise_settings.connectivity,
        qec_rounds=[schedule_settings.qec_round],
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

gates = [
    model.idle_sq_gate(qubit_d4, t_start=0),
    model.rotate_y_180(qubit_z1, t_start=0),
    model.idle_sq_gate(qubit_d1, t_start=0),
    model.idle_sq_gate(qubit_x1, t_start=0),
    model.idle_sq_gate(qubit_d2, t_start=0),

    model.idle_sq_gate(qubit_d4, t_start=t_sqg),
    model.idle_sq_gate(qubit_z1, t_start=t_sqg),
    model.idle_sq_gate(qubit_d1, t_start=t_sqg),
    model.idle_sq_gate(qubit_x1, t_start=t_sqg),
    model.idle_sq_gate(qubit_d2, t_start=t_sqg),
]
circuit = Circuit(qubits=qubits, gates=gates)
finalized_circuit = model.finalize(circuit)
state = State(qubits, dim=3)  # initialization in ground state

repetition_tree = RepeatedCircuitEvaluation(
    initial_state=state,
    circuit=finalized_circuit,
    state_projector=state_projector,
    final_state_projector=final_state_projector,
)
print(state_projector.output_state_vector_keys)
print(repetition_tree.evaluate_as_array(depth=10, repetitions=2).shape)

# finalized_circuit @ state
#
# print(state_projector.output_state_vector)
# outcome_vector = state_projector.construct_output_vector(state=state)
# print(outcome_vector)
