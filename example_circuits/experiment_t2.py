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

num_idling_gates = np.arange(0, 2000, 100)

qubit = "X2"

PLOT_CIRCUITS = False

###################

current_dir = os.path.dirname(os.path.abspath(__file__))
setup = Setup.from_file(os.path.join(current_dir, "..", noise_file))
model = DiCarloLabModel(setup=setup)

T2_true = model.p("T2", qubit)

prob_outcome_0 = []

for k, n in enumerate(num_idling_gates):
    print(k, end=" ", flush=True)
    idling = []
    for j in range(n):
        idling.append(model.idle_sq_gate(qubit, t_start=j * 20))

    circuit = Circuit(
        qubits=[qubit],
        gates=[
            model.rotate_y_90(qubit, t_start=0),
            *idling,
            model.rotate_y_m90(qubit, t_start=20 + n * 20),
            model.measure_project_0(qubit, t_start=20 + 20 + n * 20),
        ],
    )

    if PLOT_CIRCUITS:
        plot(circuit)
        plt.show()

    finalized_circuit = model.finalize(circuit)

    state = State([qubit], dim=3)  # initialization in ground state
    prob_init_state = np.trace(state.pauli_vector.to_dm())

    finalized_circuit @ state

    prob_out_state = np.trace(state.pauli_vector.to_dm())

    # prob_outcome_0[k] = float(prob_out_state / prob_init_state)
    prob_outcome_0.append(np.real(prob_out_state / prob_init_state))

print()


def exponential_decay(t, T2_inv, a, b):
    return a * np.exp(-t * T2_inv) + b


opt_params, cov_params = curve_fit(
    exponential_decay, num_idling_gates * 20, prob_outcome_0, p0=[0.001, 1, 0.5]
)
T2_inv, a, b = opt_params

plt.plot(num_idling_gates * 20, prob_outcome_0, ".")
plt.plot(
    num_idling_gates * 20,
    exponential_decay(num_idling_gates * 20, T2_inv, a, b),
    "-",
    label=f"T2={1/T2_inv:0.0f}\nT2(true)={T2_true:0.0f}",
)
plt.xlabel("idling time [ns]")
plt.ylabel("p(m=0)")
plt.legend(loc="upper right")
plt.show()
