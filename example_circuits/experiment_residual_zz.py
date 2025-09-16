import warnings

warnings.filterwarnings("ignore")

import os
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit
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

NUM_IDLING_GATES = np.arange(0, 200, 10)  # each idling gate is 20ns

qubit_1 = "Z3"
qubit_2 = "D1"

PLOT_CIRCUITS = False

###################

current_dir = os.path.dirname(os.path.abspath(__file__))
setup = Setup.from_file(os.path.join(current_dir, "..", noise_file))
model = DiCarloLabModel(setup=setup)

prob_outcome_1 = np.zeros(
    (2, len(NUM_IDLING_GATES), 2)
)  # conditional X, idling time, measured qubit
for i, conditional_x in enumerate([0, 1]):
    for j, num_i_gates in enumerate(NUM_IDLING_GATES):
        for k, meas_qubit in enumerate([qubit_1, qubit_2]):
            first_idling = []
            second_idling = []
            for n in range(num_i_gates // 2):
                first_idling.append(model.idle_sq_gate(qubit_1, t_start=n * 20 + 20))
                first_idling.append(model.idle_sq_gate(qubit_2, t_start=n * 20 + 20))
                second_idling.append(
                    model.idle_sq_gate(
                        qubit_1, t_start=n * 20 + 20 + 20 + 20 * (num_i_gates // 2)
                    )
                )
                second_idling.append(
                    model.idle_sq_gate(
                        qubit_2, t_start=n * 20 + 20 + 20 + 20 * (num_i_gates // 2)
                    )
                )

            gates = [
                model.rotate_x_90(qubit_1, t_start=0),
                (
                    model.rotate_x_180(qubit_2, t_start=0)
                    if conditional_x
                    else model.idle_sq_gate(qubit_2, t_start=0)
                ),
                *first_idling,
                model.rotate_y_180(qubit_1, t_start=20 + 20 * (num_i_gates // 2)),
                (
                    model.rotate_x_180(qubit_2, t_start=20 + 20 * (num_i_gates // 2))
                    if conditional_x
                    else model.idle_sq_gate(
                        qubit_2, t_start=20 + 20 * (num_i_gates // 2)
                    )
                ),
                *second_idling,
                model.rotate_y_90(
                    qubit_1, t_start=20 + 20 * 2 * (num_i_gates // 2) + 20
                ),
                model.idle_sq_gate(
                    qubit_2, t_start=20 + 20 * 2 * (num_i_gates // 2) + 20
                ),
                model.measure_project_1(
                    meas_qubit, t_start=20 + 20 * 2 * (num_i_gates // 2) + 20 + 20
                ),
            ]
            circuit = Circuit(qubits=[qubit_1, qubit_2], gates=gates)

            if PLOT_CIRCUITS:
                plot(circuit)
                plt.show()

            finalized_circuit = model.finalize(circuit)

            state = State([qubit_1, qubit_2], dim=3)  # initialization in ground state
            prob_init_state = np.trace(state.pauli_vector.to_dm())

            finalized_circuit @ state

            prob_out_state = np.trace(state.pauli_vector.to_dm())

            prob_outcome_1[i, j, k] = np.real(prob_out_state / prob_init_state)

for i, conditional_x in enumerate([0, 1]):
    plt.plot(
        NUM_IDLING_GATES * 20,
        prob_outcome_1[i, :, 0],
        ".-",
        label=f"{qubit_1}" + (" X" if conditional_x else ""),
    )
for i, conditional_x in enumerate([0, 1]):
    plt.plot(
        NUM_IDLING_GATES * 20,
        prob_outcome_1[i, :, 1],
        ".-",
        label=f"{qubit_2}" + (" X" if conditional_x else ""),
    )

plt.xlabel("idling time")
plt.ylabel("p(m=1)")
plt.legend(loc="best")
plt.show()
