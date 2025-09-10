import os
import numpy as np
from quantumsim.circuits import Circuit
from quantumsim import Setup
from quantumsim.circuits.circuit import Gate
from quantumsim.circuits.plotter import plot
from quantumsim.operations import Operation
from quantumsim.states.state import State
from quantumsim.models import DiCarloLabModel
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
setup = Setup.from_file(os.path.join(current_dir, "..", "setup_noisy.yaml"))
model = DiCarloLabModel(setup=setup)
print(model)

circuit = Circuit(
    qubits=["D1"],
    gates=[
        model.rotate_x("D1", t_start=0),
        model.rotate_x("D1", t_start=100),
        # model.measure_project("D1", t_start=100),
    ],
)

plot(circuit)
plt.show()


finalized_circuit = model.finalize(circuit)
print(finalized_circuit)
print(finalized_circuit._params)

finalized_circuit = finalized_circuit(result=1, theta=90)
print(finalized_circuit)
print(finalized_circuit._params)

# generate initial state
state = State(["D1"], dim=3)
print(state.pauli_vector.to_dm())
print(finalized_circuit @ state)
print(state.pauli_vector.to_dm())
