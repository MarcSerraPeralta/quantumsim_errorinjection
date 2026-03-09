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

#noise_file = "setup_noisy.yaml"
noise_file = "setup_perfect.yaml"
#noise_file = "setup_noisy_d5_t1t2_readout.yaml"

NUM_SHOTS = 10_000

###################

current_dir = os.path.dirname(os.path.abspath(__file__))
setup = Setup.from_file(os.path.join(current_dir, "..", noise_file))
dim_lookup = {"D1": 2, "D2": 2, "A1": 2}
model = DiCarloLabModelT2Ratio(setup=setup, dim_lookup=dim_lookup, angle_drift=1e-3)

list_defect_probs = []
for _ in range(NUM_SHOTS):
    gates=[
        model.rotate_y_m90("A1"),
        model.cphase("D1", "A1"),
        model.cphase("D2", "A1"),
        model.rotate_y_90("A1"),
        model.measure_project_1("A1"),
    ]
    state = State(list(dim_lookup), dim_lookup=dim_lookup)  # initialization in ground state
    prob_init_state = np.trace(state.pauli_vector.to_dm())

    apply_gates(state=state, gates=gates)

    prob_out_state = np.trace(state.pauli_vector.to_dm())
     
    print(prob_out_state / prob_init_state)


