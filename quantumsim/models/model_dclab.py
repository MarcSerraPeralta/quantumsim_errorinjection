# ------------------------
# Reference: https://arxiv.org/pdf/1210.5799
# ------------------------
import math
import numpy as np
import quantumsim.operations.qubits as ops2
import quantumsim.operations.qutrits as ops3
from quantumsim import bases
from quantumsim.operations import Operation, ParametrizedOperation
from .cphase import cphase, zz_coupling
from quantumsim.models import Model
from typing import Dict, Tuple, List, Iterable, Optional

qubit_basis = (bases.general(2),)
qutrit_basis = (bases.general(3),)
qubit_basis_classical = (bases.general(2).subbasis([0, 1]),)
qutrit_basis_classical = (bases.general(3).subbasis([0, 1, 2]),)

rad = np.deg2rad


class DiCarloLabModel(Model):
    dim = 3

    def __init__(
        self, setup, dim_lookup: Dict[str, int], *, measurement_full_projection=False
    ):
        super().__init__(setup)
        self.qubit_to_dim = dim_lookup
        self.dim_to_basis = {
            2: qubit_basis,
            3: qutrit_basis,
        }
        self.dim_to_basis_classical = {
            2: qubit_basis_classical,
            3: qutrit_basis_classical,
        }
        self.basis_lookup: Dict[str, Tuple] = {
            qubit_name: self.dim_to_basis[dim] for qubit_name, dim in dim_lookup.items()
        }
        self.basis_classical_lookup: Dict[str, Tuple] = {
            qubit_name: self.dim_to_basis_classical[dim]
            for qubit_name, dim in dim_lookup.items()
        }
        dim_to_ops = {
            2: ops2,
            3: ops3,
        }
        self.ops_lookup = {
            qubit_name: dim_to_ops[dim] for qubit_name, dim in dim_lookup.items()
        }
        if measurement_full_projection:
            self.measure = self.measure_project
        else:
            self.measure = self.measure_partial

    def get_ptm_dephase(self, qubit):
        dim = self.qubit_to_dim[qubit]
        basis = self.basis_classical_lookup[qubit]
        ptm_dephase = self.ops_lookup[qubit].rotate_x(0).set_bases(basis, basis)
        return ptm_dephase

    def get_ptm_project(self, qubit):
        dim = self.qubit_to_dim[qubit]
        ptm_project = [
            self.ops_lookup[qubit]
            .rotate_x(0)
            .set_bases(
                (bases.general(dim).subbasis([i]),), (bases.general(dim).subbasis([i]),)
            )
            for i in range(dim)
        ]
        return ptm_project

    def zz_coupling_1q(self, qubit_static, qubit_fluxed, t_start=None):
        time = self.p("time_1qubit", qubit_static, qubit_fluxed)
        input_output_basis = (
            self.basis_lookup[qubit_static] + self.basis_lookup[qubit_fluxed]
        )
        zz_coupling_ = (
            zz_coupling(
                self.p("zz_coupling_wfreq", qubit_static, qubit_fluxed) * time,
                input_output_basis,
            )
            .set_bases(input_output_basis, input_output_basis)
            .at(qubit_static, qubit_fluxed),
        )
        return zz_coupling_

    def zz_coupling_2q(self, qubit_static, qubit_fluxed, t_start=None):
        time = self.p("time_2qubit", qubit_static, qubit_fluxed)
        input_output_basis = (
            self.basis_lookup[qubit_static] + self.basis_lookup[qubit_fluxed]
        )
        zz_coupling_ = (
            zz_coupling(
                self.p("zz_coupling_wfreq", qubit_static, qubit_fluxed) * time,
                input_output_basis,
            )
            .set_bases(input_output_basis, input_output_basis)
            .at(qubit_static, qubit_fluxed),
        )
        return zz_coupling_

    def zz_coupling_measurement(self, qubit_static, qubit_fluxed, t_start=None):
        time = self.p("time_measure", qubit_static, qubit_fluxed)
        input_output_basis = (
            self.basis_lookup[qubit_static] + self.basis_lookup[qubit_fluxed]
        )
        zz_coupling_ = (
            zz_coupling(
                self.p("zz_coupling_wfreq", qubit_static, qubit_fluxed) * time,
                input_output_basis,
            )
            .set_bases(input_output_basis, input_output_basis)
            .at(qubit_static, qubit_fluxed),
        )
        return zz_coupling_

    def cphase(self, qubit_static, qubit_fluxed, t_start=None):
        """CPhase gate. Parameters: angle (degrees)"""
        time_int = self.p("cphase_time_int", qubit_static, qubit_fluxed)
        time_rise = self.p("cphase_time_rise", qubit_static, qubit_fluxed)
        time_int_point = time_int - 4 * time_rise
        time_phase_corr = self.p("cphase_time_phase_corr")
        q0_t1 = self.p("T1", qubit_static)
        q0_t2 = self.p("T2", qubit_static)
        q1_t1 = self.p("T1", qubit_fluxed)
        q1_t2 = self.p("T2", qubit_fluxed)
        q1_t2_int = self.p("T2_int", qubit_fluxed)
        t2_rise = (q1_t2 + q1_t2_int) / 2
        idle_half = (
            self._idle(qubit_static, 0.5 * time_int, q0_t1, q0_t2),
            self._idle(qubit_fluxed, time_rise, q1_t1, t2_rise),
            self._idle(qubit_fluxed, 0.5 * time_int_point, q1_t1, q1_t2_int),
            self._idle(qubit_fluxed, time_rise, q1_t1, t2_rise),
        )
        phase_corr = (
            self._idle(qubit_static, time_phase_corr, q0_t1, q0_t2),
            self._idle(qubit_fluxed, time_phase_corr, q1_t1, q1_t2),
        )
        input_output_basis = (
            self.basis_lookup[qubit_static] + self.basis_lookup[qubit_fluxed]
        )
        cphase_ = (
            cphase(
                rad(180),
                self.p("cphase_leakage_rate", qubit_static, qubit_fluxed),
                self.p("cphase_leakage_phase", qubit_static, qubit_fluxed),
                self.p("cphase_leakage_mobility_rate", qubit_static, qubit_fluxed),
                self.p("cphase_leakage_mobility_phase", qubit_static, qubit_fluxed),
                self.p("cphase_phase_22", qubit_static, qubit_fluxed),
                self.p("cphase_phase_diff_02_12", qubit_static, qubit_fluxed),
                self.p("cphase_phase_diff_20_21", qubit_static, qubit_fluxed),
                self.p("cphase_phase_corr_error_01", qubit_static, qubit_fluxed),
                self.p("cphase_phase_corr_error_10", qubit_static, qubit_fluxed),
                self.p("cphase_phase_corr_error_2q", qubit_static, qubit_fluxed),
                input_output_basis,
            )
            .set_bases(input_output_basis, input_output_basis)
            .at(qubit_static, qubit_fluxed),
        )
        return idle_half + cphase_ + idle_half + phase_corr

    def rotate(self, qubit, t_start=None):
        """Generic microwave rotation. Parameters: `phi`, `theta` (degrees).
        :math:`\\phi` is an angle, that specifies a rotation axis
        (:math:`\\phi=0` is X rotation and :math:`\\phi=90` is Y rotation).
        :math:`\\theta` specifies rotation angle.
        """
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            ParametrizedOperation(
                lambda phi, theta: self.ops_lookup[qubit].rotate_euler(
                    rad(phi), rad(theta), -rad(phi)
                ),
                self.basis_lookup[qubit],
            ),
            self._idle(qubit, half_duration, t1, t2),
        )

    def rotate_x(self, qubit, theta: float = 0.0, t_start=None, *args, **kwargs):
        """Microwave rotation around X axis. Parameters: `theta` (degrees)."""
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            self.ops_lookup[qubit]
            .rotate_x(rad(theta))
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit),
            # ParametrizedOperation(
            #     lambda theta: self.ops_lookup[qubit].rotate_x(rad(theta)),
            #     self.basis_lookup[qubit],
            # ),
            self._idle(qubit, half_duration, t1, t2),
        )

    def rotate_x_90(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            self.ops_lookup[qubit]
            .rotate_x(rad(90))
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit),
            self._idle(qubit, half_duration, t1, t2),
        )

    def rotate_x_180(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            self.ops_lookup[qubit]
            .rotate_x(rad(180))
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit),
            self._idle(qubit, half_duration, t1, t2),
        )

    def random_rotate_x_180(self, qubit, prob=1, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            self.ops_lookup[qubit]
            .random_rotate_x(rad(180), prob=prob)
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit),
            # ParametrizedOperation(
            #     lambda prob: self.ops_lookup[qubit].random_rotate_x(rad(180), prob=prob),
            #     self.basis_lookup[qubit],
            # ),
            self._idle(qubit, half_duration, t1, t2),
        )

    def depolarizing_channel(self, qubit, t=0, t_start=0):
        t1 = self.p("T1", qubit)
        duration = t
        p = (1 - np.exp(-duration / t1))
        px = p / 4
        py = p / 4
        pz = (2 - p - 2 * np.sqrt(1-p)) / 4
        return (
            self.ops_lookup[qubit]
            .depolarizing_channel(px=px, py=py, pz=pz)
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit),
        )

    def amplitude_damping_01(self, qubit, t=0, t_start=0):
        t1 = self.p("T1", qubit)
        return (
            self.ops_lookup[qubit]
            .amplitude_damping_01(t=t, t1=t1)
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit),
        )

    def phase_damping_01(self, qubit, t=0, t_start=0):
        t1 = self.p("T1", qubit)
        factor_t1_to_t2 = self.p("T1_T2_ratio", qubit)
        t2 = t1 * factor_t1_to_t2
        return (
            self.ops_lookup[qubit]
            .phase_damping_01(t=t, t2=t2)
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit),
        )

    def random_rotate_x_180_t1like(self, qubit, t=0, t_start=0):
        t1 = self.p("T1", qubit)
        duration = t
        p = (1 - np.exp(-duration / t1))
        px = p / 2
        return (
            self.ops_lookup[qubit]
            .random_rotate_x(rad(180), prob=px)
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit),
        )

    def rotate_x12(self, qubit, t_start=None):
        """Microwave rotation around X12 axis. Parameters: `theta` (degrees)."""
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            ParametrizedOperation(
                lambda theta: self.ops_lookup[qubit].rotate_x12(rad(theta)),
                self.basis_lookup[qubit],
            ),
            self._idle(qubit, half_duration, t1, t2),
        )

    def rotate_x12_180(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            self.ops_lookup[qubit]
            .rotate_x12(rad(180))
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit),
            self._idle(qubit, half_duration, t1, t2),
        )

    def rotate_y(self, qubit, t_start=None):
        """Microwave rotation around Y axis. Parameters: `theta` (degrees)."""
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            ParametrizedOperation(
                lambda theta: self.ops_lookup[qubit].rotate_y(rad(theta)),
                self.basis_lookup[qubit],
            ),
            self._idle(qubit, half_duration, t1, t2),
        )

    def rotate_y_180(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            self.ops_lookup[qubit]
            .rotate_y(rad(180))
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit),
            self._idle(qubit, half_duration, t1, t2),
        )

    def rotate_y_90(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            self.ops_lookup[qubit]
            .rotate_y(rad(90))
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit),
            self._idle(qubit, half_duration, t1, t2),
        )

    def rotate_y_m90(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            self.ops_lookup[qubit]
            .rotate_y(rad(-90))
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit),
            self._idle(qubit, half_duration, t1, t2),
        )

    def _meas_butterfly(self, qubit):
        args = [
            self.p(name, qubit)
            for name in ("meas_prob01", "meas_prob12", "meas_prob10", "meas_prob21")
        ]
        return self.ops_lookup[qubit].meas_butterfly(*args).at(qubit)

    def _project(self, qubit, state):
        if state == -1:
            return self.get_ptm_dephase(qubit)
        else:
            return self.get_ptm_project(qubit)[state]

    def measure_partial(self, qubit, t_start=None):
        t1_meas = self.p("T1_meas", qubit)
        half_time = 0.5 * self.p("time_measure", qubit)

        def project(result):
            overlap = self.p("meas_overlap_01", qubit)
            if result == 0:
                return Operation.from_ptm(
                    np.diag([1 - overlap, overlap, 0]), qutrit_basis_classical
                )
            elif result == 1:
                return Operation.from_ptm(
                    np.diag([overlap, 1 - overlap, 1]), qutrit_basis_classical
                )
            elif result == -1:
                return self._ptm_dephase
            else:
                raise ValueError("Unknown measurement result: {}".format(result))

        return (
            self._idle(qubit, half_time, t1_meas, t1_meas),
            self._meas_butterfly(qubit),
            ParametrizedOperation(project, qutrit_basis_classical).at(qubit),
            self._meas_butterfly(qubit),
            self._idle(qubit, half_time, t1_meas, t1_meas),
        )

    def measure_project(self, qubit, t_start=0):
        """Classical measurement noise needs to be added afterwards."""
        t1_meas = self.p("T1_meas", qubit)
        half_time = 0.5 * self.p("time_measure", qubit)

        def project(result):
            if result in (0, 1, 2):
                return self.get_ptm_project(qubit)[result]
            elif result == -1:
                return self.get_ptm_dephase(qubit)
            else:
                raise ValueError("Unknown measurement result: {}".format(result))

        return (
            # self._idle(qubit, half_time, t1_meas, t1_meas),
            # self._meas_butterfly(qubit),
            ParametrizedOperation(project, qutrit_basis_classical).at(qubit),
            self._meas_butterfly(qubit),
            # self._idle(qubit, 2*half_time, t1_meas, t1_meas),
        )

    def measure_project_0(self, qubit, t_start=0):
        """Classical measurement noise needs to be added afterwards."""
        basis = self.basis_classical_lookup[qubit]
        return (
            self.get_ptm_project(qubit)[0].set_bases(basis, basis).at(qubit),
            self._meas_butterfly(qubit),
        )

    def measure_project_1(self, qubit, t_start=0):
        """Classical measurement noise needs to be added afterwards."""
        basis = self.basis_classical_lookup[qubit]
        return (
            self.get_ptm_project(qubit)[1].set_bases(basis, basis).at(qubit),
            self._meas_butterfly(qubit),
        )

    def measure_project_2(self, qubit, t_start=0):
        """Classical measurement noise needs to be added afterwards."""
        basis = self.basis_classical_lookup[qubit]
        return (
            self.get_ptm_project(qubit)[2].set_bases(basis, basis).at(qubit),
            self._meas_butterfly(qubit),
        )

    def assign_matrix(self, qubit):
        return np.array(self.p("assign_matrix", qubit))

    def idle_measurement(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        idling_time = self.p("time_measure", qubit)

        return (self._idle(qubit, idling_time, t1, t2),)

    def idle_sq_gate(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        idling_time = self.p("time_1qubit", qubit)

        return (self._idle(qubit, idling_time, t1, t2),)

    def idle_tq_gate(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        idling_time = self.p("time_2qubit", qubit)

        return (self._idle(qubit, idling_time, t1, t2),)

    def idle_meas_gate(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        idling_time = self.p("time_measure", qubit)

        return (self._idle(qubit, idling_time, t1, t2),)

    def _idle(self, qubit, duration, t1, t2):
        if np.isfinite(t1) and np.isfinite(t2):
            t_phi = 1.0 / (1.0 / t2 - 0.5 / t1)
            if t_phi < 0:
                raise ValueError("t2 must be less than 2*t1")
            elif np.allclose(t_phi, 0):
                ops_t2 = []
            else:
                ops_t2 = [
                    (8.0 / (9 * t_phi)) ** 0.5
                    * np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]),
                    (2.0 / (9 * t_phi)) ** 0.5
                    * np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
                    (2.0 / (9 * t_phi)) ** 0.5
                    * np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]]),
                ]
        else:
            ops_t2 = []

        op_t1 = t1**-0.5 * np.array([
            [0, 1, 0],
            [0, 0, np.sqrt(2)],
            [0, 0, 0]
        ])

        anharmonicity = self.p("anharmonicity", qubit)
        if not np.allclose(anharmonicity, 0.0):
            ham = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, anharmonicity],
                ]
            )
        else:
            ham = None

        (dim,) = self.basis_lookup[qubit]
        if dim.dim_hilbert == 2:  # Slice matrix dimensions to work with qubit-state
            op_t1 = op_t1[:2, :2]
            ops_t2 = [op[:2, :2] for op in ops_t2]
            ham = None

        return Operation.from_lindblad_form(
            duration,
            self.basis_lookup[qubit],
            hamiltonian=ham,
            lindblad_ops=[op_t1, *ops_t2],
        ).at(qubit)

    def get_ordered_qubits_to_basis(self, qubits: List[str]) -> Iterable:
        return [self.basis_lookup[qubit_name][0] for qubit_name in qubits]


class DiCarloLabModelNoIdle(DiCarloLabModel):

    def _idle(self, qubit: str, duration: float, t1: float, t2: float):
        """
        Models idle time decoherence.

        The standard T1 relaxation is replaced by a symmetric bit-flip
        channel (|0> <-> |1>) with a total probability p=0.1, while
        T2 dephasing is preserved.
        """
        return (
            self.ops_lookup[qubit]
            .rotate_x(rad(0))
            .set_bases(self.basis_lookup[qubit], self.basis_lookup[qubit])
            .at(qubit)
        )


class DiCarloLabModelT2Ratio(DiCarloLabModel):

    def _idle(self, qubit, duration, t1, t2):
        factor_t1_to_t2 = self.p("T1_T2_ratio", qubit)
        t2 = t1 * factor_t1_to_t2

        if np.isfinite(t1) and np.isfinite(t2):
            t_phi = 1.0 / (1.0 / t2 - 0.5 / t1)
            if t_phi < 0:
                raise ValueError("t2 must be less than 2*t1")
            elif np.allclose(t_phi, 0):
                ops_t2 = []
            else:
                ops_t2 = [
                    (8.0 / (9 * t_phi)) ** 0.5
                    * np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]),
                    (2.0 / (9 * t_phi)) ** 0.5
                    * np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
                    (2.0 / (9 * t_phi)) ** 0.5
                    * np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]]),
                ]
        else:
            ops_t2 = []

        op_t1 = t1**-0.5 * np.array([
            [0, 1, 0],
            [0, 0, np.sqrt(2)],
            [0, 0, 0]
        ])

        anharmonicity = self.p("anharmonicity", qubit)
        if not np.allclose(anharmonicity, 0.0):
            ham = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, anharmonicity],
                ]
            )
        else:
            ham = None

        (dim,) = self.basis_lookup[qubit]
        if dim.dim_hilbert == 2:  # Slice matrix dimensions to work with qubit-state
            op_t1 = op_t1[:2, :2]
            ops_t2 = [op[:2, :2] for op in ops_t2]
            ham = None

        return Operation.from_lindblad_form(
            duration,
            self.basis_lookup[qubit],
            hamiltonian=ham,
            lindblad_ops=[op_t1, *ops_t2],
        ).at(qubit)
