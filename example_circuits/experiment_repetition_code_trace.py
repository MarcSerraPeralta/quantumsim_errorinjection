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

from rep_code_utils import (
    distance_3_bitflip_gates,
    distance_5_bitflip_gates,
    distance_3_prep_initial_state_gates,
    distance_5_prep_initial_state_gates,
)

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)

noise_file = "setup_noisy.yaml"
noise_file = "setup_perfect.yaml"

DISTANCE = 5

# Ryzen 7 7800X3D (4.2 GHz); d=3, r=10, seed=10.000 => 3min
qec_rounds: int = 10
repetitions: int = 10_000
state_dim: int = 3


PLOT_CIRCUITS = False

###################

if DISTANCE == 3:
    qubits = ["D4", "Z1", "D1", "X1", "D2"]
elif DISTANCE == 5:
    qubits = ["D7", "Z3", "D4", "Z1", "D1", "X1", "D2", "X2", "D3"]
else:
    raise ValueError(f"Distance {DISTANCE} not implemented.")

anc_qubits = [q for q in qubits if "D" not in q]
data_qubits = [q for q in qubits if "D" in q]

current_dir = os.path.dirname(os.path.abspath(__file__))
setup = Setup.from_file(os.path.join(current_dir, "..", noise_file))
model = DiCarloLabModel(setup=setup)
state_projector = StateProjector(
    ordered_qubit_ids=qubits,
    meas_qubit_ids=anc_qubits,
    dim_per_qubit={anc: 3 for anc in anc_qubits},
    model=model,
)
final_state_projector = StateProjector(
    ordered_qubit_ids=qubits,
    meas_qubit_ids=data_qubits,
    dim_per_qubit={data: 3 for data in data_qubits},
    model=model,
)

if DISTANCE == 3:
    init_gates = distance_3_prep_initial_state_gates(model)
    qec_gates = distance_3_bitflip_gates(model)
elif DISTANCE == 5:
    init_gates = distance_5_prep_initial_state_gates(model)
    qec_gates = distance_5_bitflip_gates(model)
else:
    raise ValueError(f"Distance {DISTANCE} not implemented.")

initial_circuit = model.finalize(Circuit(qubits=qubits, gates=init_gates))
qec_circuit = model.finalize(Circuit(qubits=qubits, gates=qec_gates))

random_array = np.random.rand(repetitions, qec_rounds)
meas_outcome_index: NDArray[int] = np.zeros(
    shape=(
        repetitions,
        qec_rounds,
    ),
    dtype=int,
)

# Populate measurement outcome indices
for i in tqdm(range(repetitions), desc="Iterate Repetitions", disable=False):
    outcomes = []
    state: State = State(
        qubits, dim=state_dim, force=True
    )  # initialization in ground state

    initial_circuit @ state

    # Iterate through qec-cycles
    for j in range(qec_rounds):
        qec_circuit @ state

        # extract ancilla outcome probabilities from the trace
        for anc_qubit in anc_qubits:
            anc_qubit_idx = state.qubits.index(anc_qubit)
            probs = state.pauli_vector.meas_prob(anc_qubit_idx)
            # avoid problems with float, e.g. -1e-16 is negative so 'choice' is complaining
            probs = [0 if (-1e-10 <= p <= 0.0) else p for p in probs]
            probs = [1 if (1.0 <= p <= 1.0 + 1e-10) else p for p in probs]
            print(probs)
            outcome = np.random.choice(len(probs), p=probs)
            outcomes.append(outcome)

        # update state from sampled outcomes
        meas_outcome = "".join(map(str, outcomes[-len(anc_qubits) :]))
        print(meas_outcome)
        meas_project_circuit = state_projector.meas_outcome_to_circuit[meas_outcome]
        meas_project_circuit @ state

        # Execute data readout only at last qec-round
        last_qec_round: bool = j == qec_rounds - 1
        if last_qec_round:
            pass

    print(outcomes)
