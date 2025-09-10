import numpy as np
import quantumsim.operations.qutrits as ops
from quantumsim import bases
from quantumsim.operations import Operation, ParametrizedOperation
from .cphase import cphase
from quantumsim.models import Model

basis = (bases.general(3),)
basis_classical = (bases.general(3).subbasis([0, 1, 2]),)

rad = np.deg2rad


class DiCarloLabModel(Model):
    dim = 3
    _ptm_dephase = ops.rotate_x(0).set_bases(basis_classical, basis_classical)
    _ptm_project = [
        ops.rotate_x(0).set_bases(
            (bases.general(3).subbasis([i]),), (bases.general(3).subbasis([i]),)
        )
        for i in range(3)
    ]

    def __init__(self, setup, *, measurement_full_projection=False):
        super().__init__(setup)
        if measurement_full_projection:
            self.measure = self.measure_project
        else:
            self.measure = self.measure_partial

    @Model.gate(
        duration="time_2qubit",
        plot_metadata={
            "style": "line",
            "markers": [
                {"style": "marker", "label": "o"},
                {"style": "marker", "label": "o"},
            ],
        },
    )
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
        cphase_ = (
            ParametrizedOperation(
                lambda angle: cphase(
                    rad(angle),
                    self.p("quasistatic_flux", qubit_static, qubit_fluxed),
                    time_int,
                    self.p("cphase_leakage_rate", qubit_static, qubit_fluxed),
                    self.p("cphase_leakage_phase", qubit_static, qubit_fluxed),
                    self.p("cphase_leakage_mobility_rate", qubit_static, qubit_fluxed),
                    self.p("cphase_leakage_mobility_phase", qubit_static, qubit_fluxed),
                    self.p("cphase_flux_sensitivity", qubit_static, qubit_fluxed),
                    self.p("cphase_phase_22", qubit_static, qubit_fluxed),
                    self.p("cphase_phase_diff_02_12", qubit_static, qubit_fluxed),
                    self.p("cphase_phase_diff_20_21", qubit_static, qubit_fluxed),
                    self.p("cphase_phase_corr_error", qubit_static, qubit_fluxed),
                ),
                basis * 2,
            ).at(qubit_static, qubit_fluxed),
        )
        return idle_half + cphase_ + idle_half + phase_corr

    @Model.gate(
        duration="time_1qubit",
        plot_metadata={"style": "box", "label": "$R_{{{phi}}}({theta})$"},
    )
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
                lambda phi, theta: ops.rotate_euler(rad(phi), rad(theta), -rad(phi)),
                basis,
            ),
            self._idle(qubit, half_duration, t1, t2),
        )

    @Model.gate(
        duration="time_1qubit", plot_metadata={"style": "box", "label": "$X({theta})$"}
    )
    def rotate_x(self, qubit, t_start=None):
        """Microwave rotation around X axis. Parameters: `theta` (degrees)."""
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            ParametrizedOperation(lambda theta: ops.rotate_x(rad(theta)), basis),
            self._idle(qubit, half_duration, t1, t2),
        )

    @Model.gate(
        duration="time_1qubit", plot_metadata={"style": "box", "label": "$Y({theta})$"}
    )
    def rotate_y(self, qubit, t_start=None):
        """Microwave rotation around Y axis. Parameters: `theta` (degrees)."""
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        half_duration = 0.5 * self.p("time_1qubit", qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            ParametrizedOperation(lambda theta: ops.rotate_y(rad(theta)), basis),
            self._idle(qubit, half_duration, t1, t2),
        )

    def _meas_butterfly(self, qubit):
        args = [
            self.p(name, qubit)
            for name in ("meas_prob01", "meas_prob12", "meas_prob10", "meas_prob21")
        ]
        return ops.meas_butterfly(*args).at(qubit)

    def _project(self, qubit, state):
        if state == -1:
            return self._ptm_dephase
        else:
            return self._ptm_project[state]

    @Model.gate(
        duration="time_measure",
        plot_metadata={"style": "box", "label": r"$\circ\!\!\!\!\!\!\!\nearrow$"},
    )
    def measure_partial(self, qubit, t_start=None):
        t1_meas = self.p("T1_meas", qubit)
        half_time = 0.5 * self.p("time_measure", qubit)

        def project(result):
            overlap = self.p("meas_overlap_01", qubit)
            if result == 0:
                return Operation.from_ptm(
                    np.diag([1 - overlap, overlap, 0]), basis_classical
                )
            elif result == 1:
                return Operation.from_ptm(
                    np.diag([overlap, 1 - overlap, 1]), basis_classical
                )
            elif result == -1:
                return self._ptm_dephase
            else:
                raise ValueError("Unknown measurement result: {}".format(result))

        return (
            self._idle(qubit, half_time, t1_meas, t1_meas),
            self._meas_butterfly(qubit),
            ParametrizedOperation(project, basis_classical).at(qubit),
            self._meas_butterfly(qubit),
            self._idle(qubit, half_time, t1_meas, t1_meas),
        )

    @Model.gate(
        duration="time_measure",
        plot_metadata={"style": "box", "label": r"$\circ\!\!\!\!\!\!\!\nearrow$"},
    )
    def measure_project(self, qubit, t_start=0):
        t1_meas = self.p("T1_meas", qubit)
        half_time = 0.5 * self.p("time_measure", qubit)

        def project(result):
            if result in (0, 1, 2):
                return self._ptm_project[result]
            elif result == -1:
                return self._ptm_dephase
            else:
                raise ValueError("Unknown measurement result: {}".format(result))

        return (
            self._idle(qubit, half_time, t1_meas, t1_meas),
            self._meas_butterfly(qubit),
            ParametrizedOperation(project, basis_classical).at(qubit),
            self._meas_butterfly(qubit),
            self._idle(qubit, half_time, t1_meas, t1_meas),
        )

    @Model.gate(
        duration="time_measure",
        plot_metadata={"style": "box", "label": "$Idle$"},
    )
    def idle_measurement(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        idling_time = self.p("time_measure", qubit)

        return (self._idle(qubit, idling_time, t1, t2),)

    @Model.gate(
        duration="time_1qubit",
        plot_metadata={"style": "box", "label": "$Idle$"},
    )
    def idle_sq_gate(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        idling_time = self.p("time_1qubit", qubit)

        return (self._idle(qubit, idling_time, t1, t2),)

    @Model.gate(
        duration="time_2qubit",
        plot_metadata={"style": "box", "label": "$Idle$"},
    )
    def idle_tq_gate(self, qubit, t_start=0):
        t1 = self.p("T1", qubit)
        t2 = self.p("T2", qubit)
        idling_time = self.p("time_2qubit", qubit)

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

        op_t1 = t1**-0.5 * np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]])
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
        return Operation.from_lindblad_form(
            duration,
            (bases.general(3),),
            hamiltonian=ham,
            lindblad_ops=[op_t1, *ops_t2],
        ).at(qubit)
