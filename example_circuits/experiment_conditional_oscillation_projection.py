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
noise_file = "setup_conditional_oscillation.yaml"
# noise_file = "setup_perfect.yaml"

CONDITIONAL_ANGLES = np.linspace(0, 360, 25)

qubit_1 = "D1"
qubit_2 = "Z3"

PLOT_CIRCUITS = False

###################

current_dir = os.path.dirname(os.path.abspath(__file__))
setup = Setup.from_file(os.path.join(current_dir, "..", noise_file))
model = DiCarloLabModel(setup=setup)

measurements = [
    model.measure_project_0,
    model.measure_project_1,
    model.measure_project_2,
]

prob_outcome_1 = np.zeros(
    (2, len(CONDITIONAL_ANGLES), 2)
)  # conditional X, cond angle, measured qubit
for i, conditional_x in enumerate([0, 1]):
    for j, phi_angle in enumerate(CONDITIONAL_ANGLES):
        for k, meas_qubit in enumerate([qubit_1, qubit_2]):
            gates = [
                model.rotate_x_90(qubit_1, t_start=0),
                (
                    model.rotate_x_180(qubit_2, t_start=0)
                    if conditional_x
                    else model.idle_sq_gate(qubit_2, t_start=0)
                ),
                model.cphase(
                    qubit_1, qubit_2, t_start=20
                ),  # static qubit, fluxed qubit
                model.rotate(qubit_1, t_start=20 + 60),
                (
                    model.rotate_x_180(qubit_2, t_start=20 + 60)
                    if conditional_x
                    else model.idle_sq_gate(qubit_2, t_start=20 + 60)
                ),
            ]
            circuit = Circuit(qubits=[qubit_1, qubit_2], gates=gates)

            if PLOT_CIRCUITS and (i == 0 and j == 0 and k == 0):
                plot(circuit)
                plt.show()

            finalized_circuit = model.finalize(circuit)
            finalized_circuit = finalized_circuit(
                theta=90, phi=phi_angle, angle=180
            )  # angle = CZ angle

            state = State([qubit_1, qubit_2], dim=3)  # initialization in ground state

            finalized_circuit @ state

            probs = []
            for m in [0, 1, 2]:
                trial_state = state.copy()
                gates = [
                    measurements[m](meas_qubit, t_start=20 + 60 + 20),
                    model.idle_meas_gate(
                        qubit_1 if meas_qubit == qubit_2 else qubit_2,
                        t_start=20 + 60 + 20,
                    ),
                ]
                circuit = Circuit(qubits=[qubit_1, qubit_2], gates=gates)
                finalized_circuit = model.finalize(circuit)

                finalized_circuit @ trial_state

                probs.append(trial_state.trace())

            index = np.random.randint(3, dtype=int)
            gates = [
                measurements[index](meas_qubit, t_start=20 + 60 + 20),
                model.idle_meas_gate(
                    qubit_1 if meas_qubit == qubit_2 else qubit_2, t_start=20 + 60 + 20
                ),
            ]
            circuit = Circuit(qubits=[qubit_1, qubit_2], gates=gates)
            finalized_circuit = model.finalize(circuit)

            finalized_circuit @ state

            if index == 1:
                prob_outcome_1[i, j, k] = np.real(state.trace())

for i, conditional_x in enumerate([0, 1]):
    plt.plot(
        CONDITIONAL_ANGLES,
        prob_outcome_1[i, :, 0],
        ".-",
        label=f"{qubit_1}" + (" X" if conditional_x else ""),
    )
for i, conditional_x in enumerate([0, 1]):
    plt.plot(
        CONDITIONAL_ANGLES,
        prob_outcome_1[i, :, 1],
        ".-",
        label=f"{qubit_2}" + (" X" if conditional_x else ""),
    )

plt.xlabel("phi [degrees]")
plt.ylabel("p(m=1)")
plt.legend(loc="best")
plt.show()
