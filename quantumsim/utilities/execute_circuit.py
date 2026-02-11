from itertools import chain
import numpy as np


def apply_gates(gates, state):
    """Applies gates in-place to the given state"""
    for indexedop in chain(*gates):
        indices = [state.qubits.index(q) for q in indexedop.indices]
        indexedop.operation(state.pauli_vector, *indices)
    return


def add_classical_error(ideal_outcome, assign_matrix):
    return np.random.choice([0, 1, 2], p=assign_matrix[ideal_outcome])
