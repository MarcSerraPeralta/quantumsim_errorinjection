import warnings

from quantumsim.operations.repetition_tree import RepeatedCircuitEvaluation

warnings.filterwarnings("ignore")
import os
import numpy as np
from quantumsim.circuits import Circuit
from quantumsim import Setup
from quantumsim.states.state import State
from quantumsim.models import DiCarloLabModel
from quantumsim.operations.state_projector import StateProjector

np.set_printoptions(precision=4, suppress=True)

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
print(repetition_tree.evaluate_as_array(depth=10, repetitions=2))

# finalized_circuit @ state
#
# print(state_projector.output_state_vector)
# outcome_vector = state_projector.construct_output_vector(state=state)
# print(outcome_vector)
