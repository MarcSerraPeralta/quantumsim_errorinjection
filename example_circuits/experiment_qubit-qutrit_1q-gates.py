import numpy as np
from quantumsim.states.state import State
import quantumsim.operations.qubits as ops2
import quantumsim.operations.qutrits as ops3
from quantumsim import bases

# system = qubit + qutrit
qubits = ["QB", "QT"]
dim_lookup = {"QB": 2, "QT": 3}
state: State = State(
    qubits, dim_lookup=dim_lookup, force=True
)  # initialization in ground state

# had to change some hard-coded dimensions from 'PauliVector.to_dm'
print(state.pauli_vector.to_dm())

# create operations
qubit_basis = (bases.general(2),)
qutrit_basis = (bases.general(3),)
rotate_qb = (
    ops2.rotate_x(np.deg2rad(90))
    .set_bases(qubit_basis, qubit_basis)
    .at(qubits.index("QB"))
)
rotate_qt = (
    ops3.rotate_x(np.deg2rad(180))
    .set_bases(qutrit_basis, qutrit_basis)
    .at(qubits.index("QT"))
)

#print(dir(rotate_qt.operation))
print(np.round(rotate_qt.operation._ptm, decimals=4))
a = rotate_qt.operation.compile()
print(a)
print(np.round(rotate_qt.operation._ptm, decimals=4))

print(rotate_qb)
print(rotate_qt)

# apply operation for qubit
print(state.pauli_vector.to_dm())
rotate_qb.operation(state.pauli_vector, *rotate_qb.indices)
print(state.pauli_vector.to_dm())

# apply operation for qutrit
print(state.pauli_vector.to_dm())
rotate_qt.operation(state.pauli_vector, *rotate_qt.indices)
print(np.round(state.pauli_vector.to_dm(), decimals=4))
