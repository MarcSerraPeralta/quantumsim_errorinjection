import warnings
import os
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from quantumsim.circuits import Circuit
from quantumsim import Setup
from quantumsim.states.state import State
from quantumsim.models import DiCarloLabModel
from quantumsim.operations.state_projector import StateProjector

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)
noise_file = "setup_noisy.yaml"

qubit_d4 = "D4"
qubit_z1 = "Z1"
qubit_d1 = "D1"
qubit_x1 = "X1"
qubit_d2 = "D2"
qubits = [qubit_d4, qubit_z1, qubit_d1, qubit_x1, qubit_d2]

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
        model.zz_coupling_2q(qubit_d1, qubit_x1, t_start=t_ref),
    ])
    t_ref += t_tqg
    # X1-D1
    gates.extend([
        model.idle_tq_gate(qubit_d4, t_start=t_ref),
        model.idle_tq_gate(qubit_z1, t_start=t_ref),
        model.idle_tq_gate(qubit_d2, t_start=t_ref),
        model.cphase(qubit_d1, qubit_x1, t_start=t_ref),
        model.zz_coupling_2q(qubit_d2, qubit_x1, t_start=t_ref),
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
        model.zz_coupling_2q(qubit_z1, qubit_d4, t_start=t_ref),
        model.zz_coupling_2q(qubit_d2, qubit_z1, t_start=t_ref),
    ])
    t_ref += t_tqg
    # Z1-D4
    gates.extend([
        model.idle_tq_gate(qubit_d1, t_start=t_ref),
        model.idle_tq_gate(qubit_x1, t_start=t_ref),
        model.idle_tq_gate(qubit_d2, t_start=t_ref),
        model.cphase(qubit_z1, qubit_d4, t_start=t_ref),
        model.zz_coupling_2q(qubit_d1, qubit_z1, t_start=t_ref),
        model.zz_coupling_2q(qubit_d2, qubit_z1, t_start=t_ref),
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


qec_rounds: int = 10
repetitions: int = 100
state_dim: int = 3


random_array = np.random.rand(repetitions, qec_rounds)
meas_outcome_index: NDArray[int] = np.zeros(shape=(
    repetitions,
    qec_rounds,
), dtype=int)
# Construct coherent qec-circuit
initial_circuit = model.finalize(Circuit(qubits=qubits, gates=prep_initial_state_gates()))
qec_circuit = model.finalize(Circuit(qubits=qubits, gates=distance_3_bitflip_gates()))(angle=180)
# Populate measurement outcome indices
state: State = State(qubits, dim=state_dim)  # initialization in ground state
initial_circuit @ state
for i in tqdm(range(repetitions), desc="Iterate Repetitions", disable=False):
    case_state: State = state.copy()
    qec_circuit @ case_state
    # Iterate through qec-cycles
    for j in range(qec_rounds):
        probability_spike: float = random_array[i, j]
        # Execute ancilla readout every qec-round
        for k, meas_outcome_key in enumerate(state_projector.output_state_vector_keys):
            trial_state: State = case_state.copy()
            meas_project_circuit = state_projector.meas_outcome_to_circuit[meas_outcome_key]
            meas_project_circuit @ trial_state
            trial_probability_state: float = trial_state.trace()
            # Process probability spike
            probability_spike -= trial_probability_state
            accepted_trial: bool = probability_spike <= 0
            if accepted_trial:
                trial_state.pauli_vector.renormalize()
                case_state = trial_state
                meas_outcome_index[i, j] = k
                break
        # Execute data readout only at last qec-round
        last_qec_round: bool = j == qec_rounds - 1
        if last_qec_round:
            pass

print(meas_outcome_index)
