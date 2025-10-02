import warnings
import os
from itertools import chain
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from quantumsim.circuits import Circuit
from quantumsim import Setup
from quantumsim.states.state import State
from quantumsim.models import DiCarloLabModel
from quantumsim.operations.state_projector import (
    StateProjector,
    fast_choice,
)
from quantumsim.utilities.process_state_projection import process_state_projection
from quantumsim.utilities.process_error_identifier import (
    error_identifier_to_dataset,
    tensor_to_error_identifier,
)
from quantumsim.utilities.process_initial_circuit import (
    RepetitionCodeDistance3,
    RepetitionCodeDistance5,
    construct_initial_state,
    construct_initial_circuit,
    InitialStateType,
)
from quantumsim.repetition_code.circuits import (
    distance_3_bitflip_gates,
    distance_5_bitflip_gates,
    distance_3_prep_initial_state_gates,
    distance_5_prep_initial_state_gates,
)
from qce_circuit.connectivity.intrf_channel_identifier import QubitIDObj
from qce_interp.visualization.plot_defect_rate import (
    plot_all_defect_rate,
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)

noise_file = "setup_noisy.yaml"
# noise_file = "setup_perfect.yaml"

DISTANCE = 3

qec_rounds: int = 10
repetitions: int = 10  # 10_000
state_dim: int = 3
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
model = DiCarloLabModel(
    setup=setup,
    dim_lookup=dim_lookup,
)

state_projector = StateProjector(
    ordered_qubit_ids=qubits,
    meas_qubit_ids=anc_qubits,
    dim_per_qubit={anc: state_dim for anc in anc_qubits},
    model=model,
)
final_state_projector = StateProjector(
    ordered_qubit_ids=qubits,
    meas_qubit_ids=data_qubits,
    dim_per_qubit={data: state_dim for data in data_qubits},
    model=model,
)

if DISTANCE == 3:
    init_gates = distance_3_prep_initial_state_gates(model)
    qec_gates = distance_3_bitflip_gates(model)
    code_layout = RepetitionCodeDistance3()
elif DISTANCE == 5:
    init_gates = distance_5_prep_initial_state_gates(model)
    qec_gates = distance_5_bitflip_gates(model)
    code_layout = RepetitionCodeDistance5()
else:
    raise ValueError(f"Distance {DISTANCE} not implemented.")

"""
initial_state = construct_initial_state(
    initial_state_type=InitialStateType.ODD,
    ordered_qubits=qubits,
    code_layout=code_layout,
)
initial_circuit = construct_initial_circuit(
    initial_state_type=InitialStateType.ODD,
    ordered_qubits=qubits,
    code_layout=code_layout,
    noise_model=model,
)
"""

print("THE ORDER OF APPLYING THE GATES IS THE ORDERING IN THE LIST")
print("IT DOES NOT USE THE t_start TO CHECK WHICH OPERATIONS ARE FIRST")
# qec_circuit = model.finalize(Circuit(qubits=qubits, gates=qec_gates))

# flatten gate list
init_gates = list(chain(*init_gates))
qec_gates = list(chain(*qec_gates))

meas_outcome_indices: NDArray[int] = np.zeros(
    shape=(
        repetitions,
        qec_rounds + 1,  # +1 for data-qubit measurement storage
    ),
    dtype=int,
)

# Populate measurement outcome indices
for i in tqdm(range(repetitions), desc="Iterate Repetitions", disable=False):
    state: State = State(
        qubits, dim_lookup=dim_lookup, force=True
    )  # initialization in ground state

    # apply init gates
    for indexedop in init_gates:
        indices = [qubits.index(q) for q in indexedop.indices]
        indexedop.operation(state.pauli_vector, *indices)

    # Iterate through qec-cycles
    for j in range(qec_rounds + 1):
        outcomes = []
        last_qec_round: bool = j == qec_rounds

        # Determine state projector (round dependent)
        current_state_projector = final_state_projector
        if not last_qec_round:
            # apply QEC cycle gates
            for indexedop in qec_gates:
                indices = [qubits.index(q) for q in indexedop.indices]
                indexedop.operation(state.pauli_vector, *indices)
            current_state_projector = state_projector

        # extract ancilla outcome probabilities from the trace
        for qubit_id in current_state_projector.ordered_measured_qubit_ids:
            qubit_idx = state.qubits.index(qubit_id)
            probs = state.pauli_vector.meas_prob(qubit_idx)
            # avoid problems with float, e.g. -1e-16 is negative so 'choice' is complaining
            probs = np.clip(probs, 0.0, 1.0) / np.sum(probs)
            # outcome = fast_choice(p=probs, rng=rng)
            outcome = np.random.choice(len(probs), p=probs)
            outcomes.append(outcome)

        qubit_indices = [
            current_state_projector.ordered_qubit_ids.index(qubit_name)
            for qubit_name in current_state_projector.ordered_measured_qubit_ids
        ]

        # update state from sampled outcomes
        meas_outcome = "".join(
            map(
                str,
                outcomes[-len(current_state_projector.ordered_measured_qubit_ids) :],
            )
        )
        meas_outcome_index: int = (
            current_state_projector.output_state_vector_keys.index(meas_outcome)
        )
        meas_outcome_indices[i, j] = meas_outcome_index
        if not last_qec_round:
            meas_project_circuit = current_state_projector.meas_outcome_to_circuit[
                meas_outcome
            ]
            meas_project_circuit @ state
            state.pauli_vector.renormalize()

kdjflasjfklsjklfjsdklfj

# print(meas_outcome_indices)
result_tensor = process_state_projection(
    state_projection_array=meas_outcome_indices,
    state_projector=state_projector,
    final_state_projector=final_state_projector,
)
# print(result_tensor)
error_identifier = tensor_to_error_identifier(
    result_tensor=result_tensor,
    depth=qec_rounds,
    repetitions=repetitions,
    code_layout=code_layout,
    initial_state=initial_state,
    involved_qubits_qubit_ids=[QubitIDObj(qubit_name) for qubit_name in qubits],
)
dataset = error_identifier_to_dataset(
    error_identifier=error_identifier,
    initial_state=initial_state,
    code_layout=code_layout,
    noise_config_string="",
)
# print(dataset)
# common_path: Path = ROOT_DIR / "simulated_datasets"
# save_dataset_to_netcdf(dataset, common_path / "defect_rate_d3.nc", overwrite=True)

fig, ax = plot_all_defect_rate(
    error_identifier=error_identifier,
    included_rounds=qec_rounds,
)
plt.show()
