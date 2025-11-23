import numpy as np
from .. import bases
from .operation import Operation

_PAULI = dict(zip(["I", "X", "Y", "Z"], bases.gell_mann(2).vectors))

bases1_default = (bases.general(2),)
bases2_default = bases1_default * 2


def rotate_euler(phi, theta, lamda):
    """A perfect single qubit rotation described by three Euler angles.

    Unitary operation, that corresponds to this rotation, is:

    .. math::

         U = R_Z(\\phi) \\cdot R_X(\\theta) \\cdot R_Z(\\lambda)

    Parameters
    ----------
    phi, theta, lamda: float
        Euler rotation angles in radians.

    Returns
    -------
    Operation
        An operation, that corresponds to the rotation.
    """
    exp_phi, exp_lambda = np.exp(1j * phi), np.exp(1j * lamda)
    sin_theta, cos_theta = np.sin(theta / 2), np.cos(theta / 2)
    matrix = np.array(
        [
            [cos_theta, -1j * exp_lambda * sin_theta],
            [-1j * exp_phi * sin_theta, exp_phi * exp_lambda * cos_theta],
        ]
    )
    return Operation.from_kraus(matrix, bases1_default)


def rotate_x(angle=np.pi):
    """A perfect single qubit rotation around :math:`Ox` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle / 2), np.cos(angle / 2)
    matrix = np.array([[cos, -1j * sin], [-1j * sin, cos]])
    return Operation.from_kraus(matrix, bases1_default)


def random_rotate_x(angle=np.pi, prob=1):
    """A random single qubit rotation around :math:`Ox` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.
    prob: float, optional
        Probability that the X rotation is applied.

    Returns
    -------
    Operation
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle / 2), np.cos(angle / 2)
    matrix = np.array([[cos, -1j * sin], [-1j * sin, cos]])
    k1 = np.sqrt(1 - prob) * np.eye(2)
    k2 = np.sqrt(prob) * matrix
    return Operation.from_kraus((k1, k2), bases1_default)


def depolarizing_channel(px=0, py=0, pz=0):
    """A depolarizing channel.

    Parameters
    ----------
    px: float, optional
        Probability of an X error
    py: float, optional
        Probability of an Y error
    pz: float, optional
        Probability of an Z error

    Returns
    -------
    Operation
        An operation, that corresponds to the depolarizing channel.
    """
    pi = 1 - px - py - pz
    ki = np.sqrt(pi) * np.eye(2)

    sigma_x = np.array([[0, 1], [1, 0]])
    kx = np.sqrt(px) * sigma_x

    sigma_y = np.array([[0, -1j], [1j, 0]])
    ky = np.sqrt(py) * sigma_y

    sigma_z = np.array([[1, 0], [0, -1]])
    kz = np.sqrt(pz) * sigma_z
    return Operation.from_kraus((ki, kx, ky, kz), bases1_default)


def amplitude_damping_01(t, t1=np.inf):
    """An amplitude damping channel from |1> to |0>.

    Parameters
    ----------
    t: float
        Duration of this channel.
    t1: float, optional
        T1 time.

    Returns
    -------
    Operation
        An operation, that corresponds to the amplitude damping channel.
    """
    p = 1 - np.exp(-t/t1)
    k0 = np.array([[1, 0], [0, np.sqrt(1-p)]])
    k1 = np.array([[0, np.sqrt(p)], [0, 0]])
    return Operation.from_kraus((k0, k1), bases1_default)


def rotate_y(angle=np.pi):
    """A perfect single qubit rotation around :math:`Oy` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle / 2), np.cos(angle / 2)
    matrix = np.array([[cos, -sin], [sin, cos]])
    return Operation.from_kraus(matrix, bases1_default)


def rotate_z(angle=np.pi):
    """A perfect single qubit rotation around :math:`Oz` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    exp = np.exp(-1j * angle / 2)
    matrix = np.diag([exp, exp.conj()])
    return Operation.from_kraus(matrix, bases1_default)


def phase_shift(angle):
    matrix = np.diag([1, np.exp(1j * angle)])
    return Operation.from_kraus(matrix, bases1_default)


def hadamard():
    """A perfect Hadamard operation.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    matrix = np.sqrt(0.5) * np.array([[1, 1], [1, -1]])
    return Operation.from_kraus(matrix, bases1_default)


def cphase(angle=np.pi):
    """A perfect controlled phase rotation.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    matrix = np.diag([1, 1, 1, np.exp(1j * angle)])
    return Operation.from_kraus(matrix, bases2_default)


def iswap(angle=np.pi / 2):
    """A perfect controlled phase rotation.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle), np.cos(angle)
    matrix = np.array(
        [[1, 0, 0, 0], [0, cos, 1j * sin, 0], [0, 1j * sin, cos, 0], [0, 0, 0, 1]]
    )
    return Operation.from_kraus(matrix, bases2_default)


def cnot():
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    return Operation.from_kraus(matrix, bases2_default)


def controlled_unitary(unitary):
    dim_hilbert = unitary.shape[0]
    if unitary.shape != (dim_hilbert, dim_hilbert):
        raise ValueError("Unitary matrix must be square")
    control_block = np.eye(2)
    off_diag_block_0 = np.zeros((2, dim_hilbert))
    off_diag_block_1 = np.zeros((dim_hilbert, 2))
    matrix = np.array([[control_block, off_diag_block_0], [off_diag_block_1, unitary]])
    return Operation.from_kraus(matrix, bases2_default)


def controlled_rotation(angle=np.pi, axis="z"):
    if axis == "x":
        sin, cos = np.sin(angle / 2), np.cos(angle / 2)
        matrix = np.array([[cos, -1j * sin], [-1j * sin, cos]])
    elif axis == "y":
        sin, cos = np.sin(angle / 2), np.cos(angle / 2)
        matrix = np.array([[cos, -sin], [sin, cos]])
    elif axis == "z":
        exp = np.exp(-1j * angle / 2)
        matrix = np.array([[exp, 0], [0, exp.conj()]])
    else:
        raise ValueError("Please provide a valid axis, got {}".format(axis))
    return controlled_unitary(matrix)


def amp_damping(total_rate=None, *, exc_rate=None, damp_rate=None):
    if total_rate is not None:
        kraus = np.array(
            [[[1, 0], [0, np.sqrt(1 - total_rate)]], [[0, np.sqrt(total_rate)], [0, 0]]]
        )
        return Operation.from_kraus(kraus, bases1_default)
    else:
        if None in (exc_rate, damp_rate):
            raise ValueError(
                "Either the total_rate or both the exc_rate and damp_rate "
                "must be provided"
            )
        comb_rate = exc_rate + damp_rate
        ptm = np.array(
            [
                [1, 0, 0, 0],
                [0, np.sqrt((1 - comb_rate)), 0, 0],
                [0, 0, np.sqrt((1 - comb_rate)), 0],
                [2 * damp_rate - comb_rate, 0, 0, 1 - comb_rate],
            ]
        )
        return Operation.from_ptm(ptm, (bases.gell_mann(2),))


def phase_damping(
    total_rate=None, *, x_deph_rate=None, y_deph_rate=None, z_deph_rate=None
):
    if total_rate is not None:
        kraus = np.array(
            [[[1, 0], [0, np.sqrt(1 - total_rate)]], [[0, 0], [0, np.sqrt(total_rate)]]]
        )
        return Operation.from_kraus(kraus, bases1_default)
    else:
        if None in (x_deph_rate, y_deph_rate, z_deph_rate):
            raise ValueError(
                "Either the total_rate or the dephasing rates along each of "
                "the three axis must be provided"
            )
        ptm = np.diag([1, 1 - x_deph_rate, 1 - y_deph_rate, 1 - z_deph_rate])
        return Operation.from_ptm(ptm, (bases.gell_mann(2),))


def amp_phase_damping(damp_rate, deph_rate):
    amp_damp = amp_damping(damp_rate)
    phase_damp = phase_damping(deph_rate)
    return Operation.from_sequence(amp_damp.at(0), phase_damp.at(0))


def bit_flipping(flip_rate):
    matrix = np.array(
        [np.sqrt(flip_rate) * _PAULI["I"], np.sqrt(1 - flip_rate) * _PAULI["X"]]
    )
    return Operation.from_kraus(matrix, bases1_default)


def phase_flipping(flip_rate):
    # This is actually equivalent to the phase damping
    matrix = np.array(
        [np.sqrt(flip_rate) * _PAULI["I"], np.sqrt(1 - flip_rate) * _PAULI["Z"]]
    )
    return Operation.from_kraus(matrix, bases1_default)


def bit_phase_flipping(flip_rate):
    matrix = np.array(
        [np.sqrt(flip_rate) * _PAULI["I"], np.sqrt(1 - flip_rate) * _PAULI["Y"]]
    )
    return Operation.from_kraus(matrix, bases1_default)


def depolarization(rate):
    rate = rate / 2
    sqrt = np.sqrt(rate)
    matrix = np.array(
        [
            np.sqrt(2 - (3 * rate)) * _PAULI["I"],
            sqrt * _PAULI["X"],
            sqrt * _PAULI["Y"],
            sqrt * _PAULI["Z"],
        ]
    )
    return Operation.from_kraus(matrix, bases1_default)


def amp_damping_qubit(p0_up, p1_down):
    """
    A gate, that excites or relaxes a qubit with a certain probability.

    Parameters
    ----------
    p0_up : float
        Probability to excite to state 1, being in the state 0
    p1_down : float
        Probability to relax to state 0, being in the state 1

    Returns
    -------
        quantumsim.operation._PTMOperation
    """
    ptm = np.identity(4, dtype=float)
    ptm[:2, :2] = [
        [1.0 - p0_up, p1_down],
        [p0_up, 1.0 - p1_down],
    ]
    basis = (bases.general(2),)
    return Operation.from_ptm(ptm, basis, basis)


def meas_butterfly(p0_up, p1_up, p1_down, p2_down):
    """
    Returns a gate, that corresponds to measurement-induced excitations.
    Each measurement should be sandwiched by two of these gates (before
    and after projection. This operation dephases the qubit immediately.

    Note: if measurement-induced leakage is reported by RB, p1_up should
    be twice larger, since RB would report average probabllity for both 0
    and 1 state.

    Parameters
    ----------
    p0_up : float
        Probability to excite to state 1, being in the state 0
    p1_up : float
        Probability to excite to state 2, being in the state 1
    p1_down : float
        Probability to relax to state 0, being in the state 1
    p2_down : float
        Probability to relax to state 1, being in the state 2

    Returns
    -------
        quantumsim.operation._PTMOperation
    """
    basis = (bases.general(2).computational_subbasis(),)
    return amp_damping_qubit(p0_up, p1_down).set_bases(bases_in=basis, bases_out=basis)
