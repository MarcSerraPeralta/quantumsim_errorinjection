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
from quantumsim.models.model_dclab import DiCarloLabModel, DiCarloLabModelT2Ratio
from quantumsim.utilities import apply_gates

np.set_printoptions(precision=4, suppress=True)

###################
# Parameters

noise_file = "setup_noisy.yaml"
# noise_file = "setup_perfect.yaml"
noise_file = "setup_noisy_d5_t1t2_readout.yaml"

num_idling_gates = np.arange(0, 1000, 30)

qubit = "Z3"

PLOT_CIRCUITS = False

###################

current_dir = os.path.dirname(os.path.abspath(__file__))
setup = Setup.from_file(os.path.join(current_dir, "..", noise_file))
dim_lookup = {qubit: 3}
model = DiCarloLabModelT2Ratio(setup=setup, dim_lookup=dim_lookup, angle_drift=0.0)

T1_true = model.p("T1", qubit)
T1_T2_ratio = model.p("T1_T2_ratio", qubit)
T1_true *= T1_T2_ratio
duration_sq = model.p("time_1qubit")

prob_outcome_1 = []

for k, n in enumerate(num_idling_gates):
    print(k, end=" ", flush=True)
    idling = []
    for j in range(n):
        idling.append(model.idle_sq_gate(qubit, t_start=(j + 1) * 20))

    gates=[
        model.rotate_x_180(qubit, t_start=0),
        model.rotate_y_90(qubit, t_start=0),
        *idling,
        model.rotate_y_m90(qubit, t_start=0),
        model.measure_project_1(qubit, t_start=20 + n * 20),
    ]

    state = State([qubit], dim_lookup=dim_lookup)  # initialization in ground state
    prob_init_state = np.trace(state.pauli_vector.to_dm())

    apply_gates(state=state, gates=gates)

    prob_out_state = np.trace(state.pauli_vector.to_dm())

    # prob_outcome_1[k] = float(prob_out_state / prob_init_state)
    prob_outcome_1.append(np.real(prob_out_state / prob_init_state))

print()


def exponential_decay(t, T1_inv, a, b):
    return a * np.exp(-t * T1_inv) + b


opt_params, cov_params = curve_fit(
    exponential_decay, num_idling_gates * duration_sq, prob_outcome_1, p0=[0.001, 1, 0]
)
T1_inv, a, b = opt_params

plt.plot(num_idling_gates * duration_sq, prob_outcome_1, ".")
plt.plot(
    num_idling_gates * duration_sq,
    exponential_decay(num_idling_gates * duration_sq, T1_inv, a, b),
    "-",
    label=f"T2={1/T1_inv:0.0f}\nT2(true)={T1_true:0.0f}",
)
plt.xlabel("idling time [ns]")
plt.ylabel("p(m=1)")
plt.legend(loc="upper right")
plt.show()
