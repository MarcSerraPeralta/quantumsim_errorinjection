import numpy as np
from itertools import product
from quantumsim.states.state import State
import quantumsim.operations.qubits as ops2
import quantumsim.operations.qutrits as ops3
from quantumsim.models.cphase import cphase
from quantumsim import bases

# system = qubit + qutrit
qubits = ["QB", "QT"]
dim_lookup = {"QB": 2, "QT": 3}
state: State = State(
    qubits, dim_lookup=dim_lookup, force=True
)  # initialization in ground state

# had to change some hard-coded dimensions from 'PauliVector.to_dm'
print(state.pauli_vector.to_dm())

# create single-qubit operations
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

# create CZ operation
cz = cphase(
    np.deg2rad(180),
    leakage_rate=0,
    leakage_phase=0,
    leakage_mobility_rate=0,
    leakage_mobility_phase=0,
    phase_22=0,
    phase_diff_02_12=0,
    phase_diff_20_21=0,
    phase_corr_error_01=0,
    phase_corr_error_10=0,
    phase_corr_error_2q=0,
    basis=qubit_basis + qutrit_basis,
).at(qubits.index("QB"), qubits.index("QT"))

# create measurement operation
ptm_project_2 = [
    ops2.rotate_x(0)
    .set_bases((bases.general(2).subbasis([i]),), (bases.general(2).subbasis([i]),))
    .at(qubits.index("QB"))
    for i in range(2)
]
ptm_project_3 = [
    ops3.rotate_x(0)
    .set_bases((bases.general(3).subbasis([i]),), (bases.general(3).subbasis([i]),))
    .at(qubits.index("QT"))
    for i in range(3)
]

for i in range(3):
    print("outcome", i)
    print(bases.general(3).subbasis([i]).vectors)
    print(ptm_project_3[i].operation._ptm)

print("RUNNING SIMULATION")
print("initial density matrix")
print(np.round(state.pauli_vector.to_dm(), decimals=4))

# apply operation for qubit
print("apply operation to qubit")
rotate_qb.operation(state.pauli_vector, *rotate_qb.indices)
print(np.round(state.pauli_vector.to_dm(), decimals=4))

# apply operation for qutrit
print("apply operation to qutrit")
rotate_qt.operation(state.pauli_vector, *rotate_qt.indices)
print(np.round(state.pauli_vector.to_dm(), decimals=4))

# apply CZ operation between qubit and qutrit
print("apply CZ operation")
cz.operation(state.pauli_vector, *cz.indices)
print(np.round(state.pauli_vector.to_dm(), decimals=4))

# get diagonal of density matrix
print("extract density matrix diagonal")
diagonal = state.pauli_vector.diagonal()
outcomes = list(product(*[list(range(dim_lookup[q])) for q in qubits]))
diagonal = np.round(diagonal, decimals=4)
for o, p in zip(outcomes, diagonal):
    print(o, p)

# apply measurement projection
m_qb, m_qt = 1, 2
print("apply projection (QB, QT):", m_qb, m_qt)
proj = ptm_project_2[m_qb]
proj.operation(state.pauli_vector, *proj.indices)
proj = ptm_project_3[m_qt]
proj.operation(state.pauli_vector, *proj.indices)
print(np.round(state.pauli_vector.to_dm(), decimals=4))
