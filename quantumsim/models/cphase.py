import numpy as np
from scipy.linalg import expm

from quantumsim import bases
from quantumsim.algebra.tools import verify_kraus_unitarity
from quantumsim.operations import Operation


def cphase(
    angle,
    leakage_rate,
    leakage_phase,
    leakage_mobility_rate,
    leakage_mobility_phase,
    phase_22,
    phase_diff_02_12,
    phase_diff_20_21,
    phase_corr_error_01,
    phase_corr_error_10,
    phase_corr_error_2q,
    basis,
):
    """

    Parameters
    ----------
    angle : float
        Conditional phase of a CPhase gate, default is :math:`\\pi`.
    quasistatic_flux : float
    time_int : float
    leakage_rate : float
    leakage_phase : float
    leakage_mobility_rate : float
    leakage_mobility_phase : float
    flux_sensitivity : float
    phase_22 : float
    phase_diff_02_12 : float
    phase_diff_20_21 : float
    phase_corr_error : float

    Returns
    -------
    Operation
        Resulting CPhase operation. First qubit is static (low-frequency)
        qubit,
    """
    qstatic_interf_leakage = 0
    rot_angle = (
        angle
        + 1.5 * 0
        + phase_corr_error_2q
        + (phase_corr_error_01 + phase_corr_error_10)
    )

    dim1, dim2 = [b.dim_hilbert for b in basis]
    if dim1 == 2 and dim2 == 2:
        ideal_unitary = expm(
            1j
            * _ideal_generator_basis22(
                phase_10=phase_corr_error_10,
                phase_01=phase_corr_error_01,
                phase_11=rot_angle,
            )
        )
        cz_unitary = ideal_unitary
    elif dim1 == 2 and dim2 == 3:
        ideal_unitary = expm(
            1j
            * _ideal_generator_basis23(
                phase_10=phase_corr_error_10,
                phase_01=phase_corr_error_01,
                phase_11=angle,  # rot_angle,
                phase_02=angle,  # rot_angle,
                phase_12=phase_diff_02_12,  # - rot_angle,
            )
        )
        print("MISSING NOISE AND INCLUDE ROT_ANGLE!!!!")
        cz_unitary = ideal_unitary
        print(np.round(cz_unitary, decimals=4))
    elif dim1 == 3 and dim2 == 2:
        ideal_unitary = expm(
            1j
            * _ideal_generator_basis32(
                phase_10=phase_corr_error_10,
                phase_01=phase_corr_error_01,
                phase_11=rot_angle,
                phase_20=0,
                phase_21=phase_diff_20_21,
            )
        )
        cz_unitary = ideal_unitary
    elif dim1 == 3 and dim2 == 3:
        ideal_unitary = expm(
            1j
            * _ideal_generator_basis33(
                phase_10=phase_corr_error_10,
                phase_01=phase_corr_error_01,
                phase_11=rot_angle,
                phase_02=rot_angle,
                phase_12=phase_diff_02_12 - rot_angle,
                phase_20=0,
                phase_21=phase_diff_20_21,
                phase_22=phase_22,
            )
        )
        noisy_unitary = expm(
            1j
            * _exchange_generator(
                leakage=4 * leakage_rate + qstatic_interf_leakage,
                leakage_phase=leakage_phase,
                leakage_mobility_rate=leakage_mobility_rate,
                leakage_mobility_phase=leakage_mobility_phase,
            )
        )
        cz_unitary = ideal_unitary @ noisy_unitary
    else:
        raise NotImplementedError(
            f"Dimensionality ({dim1}, {dim2}) is not supported. Please specify custom cz unitary."
        )

    if not verify_kraus_unitarity(cz_unitary):
        raise RuntimeError("CPhase gate is not unitary, " "verify provided parameters.")
    cz_op = Operation.from_kraus(cz_unitary, basis)
    return cz_op


def _ideal_generator_basis33(
    phase_01, phase_02, phase_10, phase_11, phase_12, phase_20, phase_21, phase_22
):
    phases = np.array(
        [
            0,
            phase_01,
            phase_02,
            phase_10,
            phase_11,
            phase_12,
            phase_20,
            phase_21,
            phase_22,
        ]
    )
    generator = np.diag(phases).astype(complex)
    return generator


def _ideal_generator_basis32(phase_01, phase_10, phase_11, phase_20, phase_21):
    phases = np.array(
        [
            0,
            phase_01,
            phase_10,
            phase_11,
            phase_20,
            phase_21,
        ]
    )
    generator = np.diag(phases).astype(complex)
    return generator


def _ideal_generator_basis23(phase_01, phase_02, phase_10, phase_11, phase_12):
    phases = np.array(
        [
            0,
            phase_01,
            phase_02,
            phase_10,
            phase_11,
            phase_12,
        ]
    )
    generator = np.diag(phases).astype(complex)
    return generator


def _ideal_generator_basis22(phase_01, phase_10, phase_11):
    phases = np.array(
        [
            0,
            phase_01,
            phase_10,
            phase_11,
        ]
    )
    generator = np.diag(phases).astype(complex)
    return generator


def _exchange_generator(
    leakage, leakage_phase, leakage_mobility_rate, leakage_mobility_phase
):
    generator = np.zeros((9, 9), dtype=complex)
    generator[2][4] = 1j * np.arcsin(np.sqrt(leakage)) * np.exp(1j * leakage_phase)
    generator[4][2] = -1j * np.arcsin(np.sqrt(leakage)) * np.exp(-1j * leakage_phase)
    generator[5][7] = (
        1j
        * np.arcsin(np.sqrt(leakage_mobility_rate))
        * np.exp(1j * leakage_mobility_phase)
    )
    generator[7][5] = (
        -1j
        * np.arcsin(np.sqrt(leakage_mobility_rate))
        * np.exp(-1j * leakage_mobility_phase)
    )
    return generator


def zz_coupling(
    zz_coupling_11,
    basis,
):

    dimensionality = basis[0].dim_hilbert
    if dimensionality == 2:
        zz_unitary = expm(
            1j
            * _ideal_generator_basis2(
                phase_10=0,
                phase_01=0,
                phase_11=zz_coupling_11,
            )
        )
    elif dimensionality == 3:
        zz_unitary = expm(
            1j
            * _ideal_generator_basis3(
                phase_10=0,
                phase_01=0,
                phase_11=zz_coupling_11,
                phase_02=0,
                phase_12=0,
                phase_20=0,
                phase_21=0,
                phase_22=0,
            )
        )
    else:
        raise NotImplementedError(
            f"Dimensionality ({dimensionality}) is not supported. Please specify custom zz unitary."
        )

    if not verify_kraus_unitarity(zz_unitary):
        raise RuntimeError(
            "ZZ-coupling gate is not unitary, " "verify provided parameters."
        )
    zz_op = Operation.from_kraus(zz_unitary, basis)
    return zz_op
