import numpy as np
from scipy.linalg import expm

from quantumsim import bases
from quantumsim.algebra.tools import verify_kraus_unitarity
from quantumsim.operations import Operation


def cphase(
    angle,
    quasistatic_flux,
    time_int,
    leakage_rate,
    leakage_phase,
    leakage_mobility_rate,
    leakage_mobility_phase,
    flux_sensitivity,
    phase_22,
    phase_diff_02_12,
    phase_diff_20_21,
    phase_corr_error,
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
    qstatic_deviation = time_int * np.pi * flux_sensitivity * quasistatic_flux**2
    qstatic_interf_leakage = (0.5 - (2 * leakage_rate)) * (
        1 - np.cos(1.5 * qstatic_deviation)
    )

    rot_angle = angle + 1.5 * qstatic_deviation + 2 * phase_corr_error

    ideal_unitary = expm(
        1j
        * _ideal_generator(
            phase_10=phase_corr_error,
            phase_01=phase_corr_error + qstatic_deviation,
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
    if not verify_kraus_unitarity(cz_unitary):
        raise RuntimeError("CPhase gate is not unitary, " "verify provided parameters.")
    cz_op = Operation.from_kraus(cz_unitary, (bases.general(3),) * 2)
    return cz_op


def _ideal_generator(
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
