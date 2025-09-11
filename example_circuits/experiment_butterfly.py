import warnings

warnings.filterwarnings("ignore")

import os
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from quantumsim.circuits import Circuit
from quantumsim import Setup
from quantumsim.circuits.plotter import plot
from quantumsim.states.state import State
from quantumsim.models import DiCarloLabModel

np.set_printoptions(precision=4, suppress=True)

###################
# Parameters

noise_file = "setup_noisy.yaml"
# noise_file = "setup_perfect.yaml"

qubit = "Z3"

PLOT_CIRCUITS = False

###################

current_dir = os.path.dirname(os.path.abspath(__file__))
setup = Setup.from_file(os.path.join(current_dir, "..", noise_file))
model = DiCarloLabModel(setup=setup)

meas_map = [model.measure_project_0, model.measure_project_1, model.measure_project_2]

ideal_outcome_probs = np.zeros((3, 3, 3))
for init_state in [0, 1, 2]:
    for outcomes in product([0, 1, 2], repeat=2):

        if init_state == 0:
            state_preparation = []
            t_current = 0
        elif init_state == 1:
            state_preparation = [model.rotate_x(qubit, t_start=0)]
            t_current = 20
        elif init_state == 2:
            state_preparation = [
                model.rotate_x(qubit, t_start=0),
                model.rotate_x12(qubit, t_start=20),
            ]
            t_current = 40
        else:
            raise ValueError

        circuit = Circuit(
            qubits=[qubit],
            gates=[
                *state_preparation,
                meas_map[outcomes[0]](qubit, t_start=t_current),
                meas_map[outcomes[1]](qubit, t_start=t_current + 300),
            ],
        )

        if PLOT_CIRCUITS:
            plot(circuit)
            plt.show()

        finalized_circuit = model.finalize(circuit)
        finalized_circuit = finalized_circuit(theta=180)

        state = State([qubit], dim=3)  # initialization in ground state
        prob_init_state = np.trace(state.pauli_vector.to_dm())

        finalized_circuit @ state

        prob_out_state = np.trace(state.pauli_vector.to_dm())

        prob_outcome = prob_out_state / prob_init_state
        ideal_outcome_probs[init_state, outcomes[0], outcomes[1]] = prob_outcome


noisy_outcome_probs = np.zeros((3, 3, 3))
assign_matrix = model.assign_matrix(qubit)
for init_state in [0, 1, 2]:
    p_ideal = ideal_outcome_probs[init_state]
    for i, j in product([0, 1, 2], repeat=2):
        for k, l in product([0, 1, 2], repeat=2):
            noisy_outcome_probs[init_state, i, j] += (
                assign_matrix[k, i] * assign_matrix[l, j] * p_ideal[k, l]
            )


# solve system of equation from butterfly experiment
# https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.130.250602/MarquesAli_SOM_230514.pdf
def butterfly_equations(vars, probs=noisy_outcome_probs):
    vars = vars.reshape(3, 3, 3)
    eqs = [
        0,
    ] * 27
    for k, (i, mk, ml) in enumerate(product([0, 1, 2], repeat=3)):
        # eq S13
        for s in [0, 1, 2]:
            for j in [0, 1, 2]:
                eqs[k] += vars[i, mk, s] * vars[s, ml, j]
        eqs[k] -= probs[i, mk, ml]

    return eqs


# intial guess (solution is very dependent on the initial guess)
x0 = np.zeros(27)
x0 = x0.reshape(3, 3, 3)
for i in [0, 1, 2]:
    x0[i, i, i] = 1
x0 = x0.reshape(-1)

solution = fsolve(butterfly_equations, x0)
eps = solution.reshape(3, 3, 3)

assign_matrix = np.zeros((3, 3))
for i, m in product([0, 1, 2], repeat=2):
    assign_matrix[i, m] = eps[i, m].sum()

qnd_matrix = np.zeros((3, 3))
for i, j in product([0, 1, 2], repeat=2):
    qnd_matrix[i, j] = eps[i, :, j].sum()

print("ASSIGN")
print(assign_matrix)
print("QND")
print(qnd_matrix)
