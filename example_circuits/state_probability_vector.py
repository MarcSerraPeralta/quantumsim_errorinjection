import warnings

warnings.filterwarnings("ignore")

import os
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from quantumsim.circuits import Circuit, FinalizedCircuit
from quantumsim import Setup
from quantumsim.circuits.plotter import plot
from quantumsim.states.state import State
from quantumsim.models import DiCarloLabModel

np.set_printoptions(precision=4, suppress=True)

###################
# Parameters

noise_file = "setup_noisy.yaml"

qubit = "Z3"

PLOT_CIRCUITS = False

###################

current_dir = os.path.dirname(os.path.abspath(__file__))
setup = Setup.from_file(os.path.join(current_dir, "..", noise_file))
model = DiCarloLabModel(setup=setup)


def get_output_state_vector(circuit: FinalizedCircuit) -> np.ndarray:
    state = State([qubit], dim=3)  # initialization in ground state
    prob_init_state = np.trace(state.pauli_vector.to_dm())

    circuit @ state

    prob_out_state = np.trace(state.pauli_vector.to_dm())

    prob_outcome = prob_out_state / prob_init_state
    return prob_outcome


circuit = Circuit(
    qubits=[qubit],
    gates=[
        model.rotate_x90(qubit, t_start=0),
        model.measure_project_1(qubit, t_start=20)
    ],
)
finalized_circuit: FinalizedCircuit = model.finalize(circuit)

print(get_output_state_vector(circuit=finalized_circuit))
