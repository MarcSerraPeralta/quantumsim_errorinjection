import os
from dataclasses import dataclass
from typing import List, Optional, Any
import xarray as xr
import numpy as np
from pathlib import Path
from enum import Enum, unique, auto
from quantumsim.circuits import Circuit, FinalizedCircuit
from quantumsim import Setup
from quantumsim.states.state import State
from quantumsim.models import DiCarloLabModel
from quantumsim.operations.state_projector import StateProjector
from qce_circuit.language.intrf_declarative_circuit import (
    InitialStateEnum,
    InitialStateContainer,
)
from qce_circuit.structure.acquisition_indexing.kernel_repetition_code import (
    SimulatedRepetitionExperimentKernel,
)
from qce_circuit.utilities.singleton_base import SingletonABCMeta
from qce_circuit.connectivity.intrf_channel_identifier import (
    IQubitID,
    QubitIDObj,
    IEdgeID,
    EdgeIDObj,
)
from qce_circuit.connectivity.intrf_connectivity_surface_code import (
    IParityGroup,
    StabilizerType,
)
from qce_circuit.connectivity.intrf_connectivity_gate_sequence import (
    GateSequenceLayer,
    Operation,
)
from qce_circuit.connectivity.connectivity_surface_code import (
    ParityGroup,
    Surface17Layer,
    get_higher_frequency_qubit_id,
    get_lower_frequency_qubit_id,
)
from qce_circuit.connectivity.generic_gate_sequence import (
    IGenericSurfaceCodeLayer,
    GenericSurfaceCode,
)
from qce_circuit.connectivity.intrf_connectivity_surface_code import (
    ISurfaceCodeLayer,
    StabilizerType,
)
from qce_interp.utilities.expected_parities import initial_state_to_expected_parity
from qce_interp.utilities.serialize_error_identifier import construct_processed_dataset
from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier
from qce_interp.interface_definitions.intrf_state_classification import (
    StateClassifierContainer,
    ParityType,
)
from quantumsim.operations.repetition_tree import LazyTreeCircuitEvaluation, RepeatedCircuitEvaluation


@unique
class RepetitionCodeType(Enum):
    DISTANCE_3 = auto()
    DISTANCE_5 = auto()


@unique
class InitialStateType(Enum):
    ALL_ZERO = auto()
    ALL_ONES = auto()
    EVEN = auto()
    ODD = auto()


class RepetitionCodeDistance3(
    GenericSurfaceCode, IGenericSurfaceCodeLayer, metaclass=SingletonABCMeta
):

    # region Class Constructor
    def __init__(self):
        super().__init__(
            gate_sequences=[
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj("D1")),
                    ],
                    _gate_operations=[
                        Operation.type_gate(
                            EdgeIDObj(QubitIDObj("X1"), QubitIDObj("D2"))
                        ),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj("D2")),
                    ],
                    _gate_operations=[
                        Operation.type_gate(
                            EdgeIDObj(QubitIDObj("X1"), QubitIDObj("D1"))
                        ),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj("D2")),
                    ],
                    _gate_operations=[
                        Operation.type_gate(
                            EdgeIDObj(QubitIDObj("Z1"), QubitIDObj("D1"))
                        ),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj("Z3")),
                        Operation.type_park(QubitIDObj("X3")),
                    ],
                    _gate_operations=[
                        Operation.type_gate(
                            EdgeIDObj(QubitIDObj("D4"), QubitIDObj("Z1"))
                        ),
                    ],
                ),
            ],
            parity_group_z=[
                ParityGroup(
                    _parity_type=StabilizerType.STABILIZER_Z,
                    _ancilla_qubit=QubitIDObj("X1"),
                    _data_qubits=[QubitIDObj("D1"), QubitIDObj("D2")],
                ),
                ParityGroup(
                    _parity_type=StabilizerType.STABILIZER_Z,
                    _ancilla_qubit=QubitIDObj("Z1"),
                    _data_qubits=[QubitIDObj("D4"), QubitIDObj("D1")],
                ),
            ],
            parity_group_x=[],
        )

    # endregion


class RepetitionCodeDistance5(
    GenericSurfaceCode, IGenericSurfaceCodeLayer, metaclass=SingletonABCMeta
):
    # region Class Constructor
    def __init__(self):
        super().__init__(
            gate_sequences=[
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj('D1')),
                        Operation.type_park(QubitIDObj('Z1')),
                        Operation.type_park(QubitIDObj('X3')),
                    ],
                    _gate_operations=[
                        Operation.type_gate(EdgeIDObj(QubitIDObj('X1'), QubitIDObj('D2'))),
                        Operation.type_gate(EdgeIDObj(QubitIDObj('Z3'), QubitIDObj('D4'))),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj('D2')),
                    ],
                    _gate_operations=[
                        Operation.type_gate(EdgeIDObj(QubitIDObj('X1'), QubitIDObj('D1'))),
                        Operation.type_gate(EdgeIDObj(QubitIDObj('Z3'), QubitIDObj('D7'))),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj('Z3')),
                        Operation.type_park(QubitIDObj('X3')),
                        Operation.type_park(QubitIDObj('D3')),
                    ],
                    _gate_operations=[
                        Operation.type_gate(EdgeIDObj(QubitIDObj('D4'), QubitIDObj('Z1'))),
                        Operation.type_gate(EdgeIDObj(QubitIDObj('X2'), QubitIDObj('D2'))),
                    ],
                ),
                GateSequenceLayer(
                    _park_operations=[
                        Operation.type_park(QubitIDObj('D2')),
                    ],
                    _gate_operations=[
                        Operation.type_gate(EdgeIDObj(QubitIDObj('Z1'), QubitIDObj('D1'))),
                        Operation.type_gate(EdgeIDObj(QubitIDObj('X2'), QubitIDObj('D3'))),
                    ],
                ),
            ],
            parity_group_z=[
                ParityGroup(
                    _parity_type=StabilizerType.STABILIZER_Z,
                    _ancilla_qubit=QubitIDObj('X2'),
                    _data_qubits=[QubitIDObj('D2'), QubitIDObj('D3')]
                ),
                ParityGroup(
                    _parity_type=StabilizerType.STABILIZER_Z,
                    _ancilla_qubit=QubitIDObj('X1'),
                    _data_qubits=[QubitIDObj('D1'), QubitIDObj('D2')]
                ),
                ParityGroup(
                    _parity_type=StabilizerType.STABILIZER_Z,
                    _ancilla_qubit=QubitIDObj('Z1'),
                    _data_qubits=[QubitIDObj('D4'), QubitIDObj('D1')]
                ),
                ParityGroup(
                    _parity_type=StabilizerType.STABILIZER_Z,
                    _ancilla_qubit=QubitIDObj('Z3'),
                    _data_qubits=[QubitIDObj('D4'), QubitIDObj('D7')]
                ),
            ],
            parity_group_x=[],
        )
    # endregion


@dataclass(frozen=True)
class CodeSettings:
    qubits: List[str]
    code_layout: ISurfaceCodeLayer

    # region Class Methods
    @classmethod
    def as_distance_3(cls) -> 'CodeSettings':
        return CodeSettings(
            qubits=["D4", "Z1", "D1", "X1", "D2"],
            code_layout=RepetitionCodeDistance3(),
        )

    @classmethod
    def as_distance_5(cls) -> 'CodeSettings':
        return CodeSettings(
            qubits=["D7", "Z3", "D4", "Z1", "D1", "X1", "D2", "X2", "D3"],
            code_layout=RepetitionCodeDistance5(),
        )

    @classmethod
    def as_single_stabilizer(cls, parity_group: IParityGroup) -> 'CodeSettings':
        return CodeSettings(
            qubits=[qubit_id.id for qubit_id in [parity_group.ancilla_id] + parity_group.data_ids],
            code_layout=GenericSurfaceCode(
                gate_sequences=[],
                parity_group_z=[] if parity_group.parity_type == StabilizerType.STABILIZER_X else [parity_group],
                parity_group_x=[] if parity_group.parity_type == StabilizerType.STABILIZER_Z else [parity_group],
                surface_code_layer=Surface17Layer(),
            ),
        )

    @classmethod
    def from_code_type(cls, repetition_code_type: RepetitionCodeType) -> Optional['CodeSettings']:
        if repetition_code_type == RepetitionCodeType.DISTANCE_3:
            return cls.as_distance_3()
        elif repetition_code_type == RepetitionCodeType.DISTANCE_5:
            return cls.as_distance_5()
        raise NotImplementedError(f"Code type ({repetition_code_type.name}) is not supported yet.")
    # endregion


class MeasurementControl:
    """
    Behaviour class, builds a specific measurement environment with user-friendly inputs.
    """

    # region Class Properties
    @property
    def state_projector(self) -> StateProjector:
        return self._state_projector

    @property
    def final_state_projector(self) -> StateProjector:
        return self._final_state_projector

    @property
    def initial_circuit(self) -> FinalizedCircuit:
        return self.construct_initial_circuit(self._initial_state_type)

    @property
    def initial_state(self) -> InitialStateContainer:
        return self.construct_initial_state(self._initial_state_type)
    # endregion

    # region Class Constructor
    def __init__(self, noise_model: DiCarloLabModel, code_settings: CodeSettings, initial_state_type: InitialStateType = InitialStateType.ODD, state_dim: int = 3):
        self._code_settings: CodeSettings = code_settings
        self._noise_model: DiCarloLabModel = noise_model
        self._initial_state_type: InitialStateType = initial_state_type
        ancilla_qubits: List[str] = [
            qubit_name
            for qubit_name in self._code_settings.qubits
            if QubitIDObj(qubit_name) in self._code_settings.code_layout.ancilla_qubit_ids
        ]
        self._state_projector = StateProjector(
            ordered_qubit_ids=self._code_settings.qubits,
            meas_qubit_ids=ancilla_qubits,
            dim_per_qubit={anc: 3 for anc in ancilla_qubits},
            model=self._noise_model,
        )
        data_qubits: List[str] = [
            qubit_name
            for qubit_name in self._code_settings.qubits
            if QubitIDObj(qubit_name) in self._code_settings.code_layout.data_qubit_ids
        ]
        self._final_state_projector = StateProjector(
            ordered_qubit_ids=self._code_settings.qubits,
            meas_qubit_ids=data_qubits,
            dim_per_qubit={data: 3 for data in data_qubits},
            model=self._noise_model,
        )
        self.state_dim = state_dim
    # endregion

    # region Class Methods
    def construct_initial_circuit(self, initial_state_type: InitialStateType) -> FinalizedCircuit:
        gates = []
        t_ref: float = 0
        # All zero (Even) initial state
        if initial_state_type in [InitialStateType.ALL_ZERO, InitialStateType.EVEN]:
            for qubit in self._code_settings.qubits:
                gates.append(self._noise_model.idle_sq_gate(qubit, t_start=t_ref))
            return self._noise_model.finalize(Circuit(
                qubits=self._code_settings.qubits,
                gates=gates,
            ))
        # All ones initial state
        if initial_state_type == InitialStateType.ALL_ONES:
            for qubit in self._code_settings.qubits:
                if QubitIDObj(qubit) in self._code_settings.code_layout.data_qubit_ids:
                    gates.append(self._noise_model.rotate_x_180(qubit, t_start=t_ref))
                else:
                    gates.append(self._noise_model.idle_sq_gate(qubit, t_start=t_ref))

            return self._noise_model.finalize(Circuit(
                qubits=self._code_settings.qubits,
                gates=gates,
            ))
        # Odd initial state
        if initial_state_type == InitialStateType.ODD:
            data_qubit_count: int = 0
            for qubit in self._code_settings.qubits:
                if QubitIDObj(qubit) in self._code_settings.code_layout.data_qubit_ids:
                    if data_qubit_count % 2 == 0:
                        gates.append(self._noise_model.idle_sq_gate(qubit, t_start=t_ref))
                    else:
                        gates.append(self._noise_model.rotate_x_180(qubit, t_start=t_ref))
                    data_qubit_count += 1
                else:
                    gates.append(self._noise_model.idle_sq_gate(qubit, t_start=t_ref))
            return self._noise_model.finalize(Circuit(
                qubits=self._code_settings.qubits,
                gates=gates,
            ))
        raise NotImplementedError(f"Initial state type ({initial_state_type.name}) not yet supported.")

    def construct_initial_state(self, initial_state_type: InitialStateType) -> InitialStateContainer:
        data_qubit_names: List[str] = [
            qubit_name
            for qubit_name in self._code_settings.qubits
            if QubitIDObj(qubit_name) in self._code_settings.code_layout.data_qubit_ids
        ]
        # All zero (Even) initial state
        if initial_state_type in [InitialStateType.ALL_ZERO, InitialStateType.EVEN]:
            return InitialStateContainer.from_ordered_list(
                initial_states=[
                    InitialStateEnum.ZERO
                    for i, qubit_id in enumerate(data_qubit_names)
                ]
            )
        # All ones initial state
        if initial_state_type == InitialStateType.ALL_ONES:
            return InitialStateContainer.from_ordered_list(
                initial_states=[
                    InitialStateEnum.ONE
                    for i, qubit_id in enumerate(data_qubit_names)
                ]
            )
        # Odd initial state
        if initial_state_type == InitialStateType.ODD:
            return InitialStateContainer.from_ordered_list(
                initial_states=[
                    InitialStateEnum.ONE if (i % 2 == 1) else InitialStateEnum.ZERO
                    for i, qubit_id in enumerate(data_qubit_names)
                ]
            )
        raise NotImplementedError(f"Initial state type ({initial_state_type.name}) not yet supported.")

    def construct_error_identifier(self, cycle_circuit: Circuit, depth: int = 10, repetitions: int = 10, **error_identifier_construction_kwargs) -> ErrorDetectionIdentifier:
        # Construct initial state
        state = State(self._code_settings.qubits, dim=self.state_dim, force=False)  # initialization in ground state
        _ = self.initial_circuit @ state

        repetition_tree = LazyTreeCircuitEvaluation(
            initial_state=state,
            circuit=self._noise_model.finalize(cycle_circuit),
            state_projector=self.state_projector,
            final_state_projector=self.final_state_projector,
            max_depth=depth,
        )
        result_tensor: np.ndarray = repetition_tree.evaluate_as_array(
            depth=depth,
            repetitions=repetitions,
            verbose=True,
        )
        error_identifier: ErrorDetectionIdentifier = self.tensor_to_error_identifier(
            result_tensor=result_tensor,
            depth=depth,
            repetitions=repetitions,
            code_layout=self._code_settings.code_layout,
            initial_state=self.initial_state,
            involved_qubits_qubit_ids=[QubitIDObj(qubit_name) for qubit_name in self._code_settings.qubits],
            **error_identifier_construction_kwargs,
        )
        return error_identifier

    def construct_dataset(self, error_identifier: ErrorDetectionIdentifier, noise_file_path: Path) -> xr.Dataset:
        with open(noise_file_path, 'r') as file:
            noise_config_string = file.read()
            dataset: xr.Dataset = self.error_identifier_to_dataset(
                error_identifier=error_identifier,
                initial_state=self.initial_state,
                code_layout=self._code_settings.code_layout,
                noise_config_string=noise_config_string,
            )
        return dataset

    def construct_distance_3_bitflip_circuit(self, error_probability: float = 0.0, error_coherence: bool = True) -> Circuit:
        gates = []
        t_sqg: float = 20
        t_tqg: float = 60
        t_ref: float = 0
        qubits: List[str] = self._code_settings.qubits
        # Random flips
        if error_coherence:
            theta: float = np.rad2deg(2 * np.arcsin(np.sqrt(error_probability)))
            gates.extend(
                [
                    self._noise_model.rotate_x("D4", t_start=t_ref, theta=theta),
                    self._noise_model.rotate_x("Z1", t_start=t_ref, theta=theta),
                    self._noise_model.rotate_x("D1", t_start=t_ref, theta=theta),
                    self._noise_model.rotate_x("X1", t_start=t_ref, theta=theta),
                    self._noise_model.rotate_x("D2", t_start=t_ref, theta=theta),
                ]
            )
        else:
            pass
        t_ref += t_sqg
        # Start of Parities
        gates.extend(
            [
                self._noise_model.idle_sq_gate("D4", t_start=t_ref),
                self._noise_model.idle_sq_gate("Z1", t_start=t_ref),
                self._noise_model.idle_sq_gate("D1", t_start=t_ref),
                self._noise_model.rotate_y_90("X1", t_start=t_ref),
                self._noise_model.idle_sq_gate("D2", t_start=t_ref),
            ]
        )
        t_ref += t_sqg
        # X1-D2
        gates.extend(
            [
                self._noise_model.idle_tq_gate("D4", t_start=t_ref),
                self._noise_model.idle_tq_gate("Z1", t_start=t_ref),
                self._noise_model.idle_tq_gate("D1", t_start=t_ref),
                self._noise_model.cphase("D2", "X1", t_start=t_ref),
                self._noise_model.zz_coupling_2q("D1", "X1", t_start=t_ref),
            ]
        )
        t_ref += t_tqg
        # X1-D1
        gates.extend(
            [
                self._noise_model.idle_tq_gate("D4", t_start=t_ref),
                self._noise_model.idle_tq_gate("Z1", t_start=t_ref),
                self._noise_model.idle_tq_gate("D2", t_start=t_ref),
                self._noise_model.cphase("D1", "X1", t_start=t_ref),
                self._noise_model.zz_coupling_2q("D2", "X1", t_start=t_ref),
            ]
        )
        t_ref += t_tqg
        # Middle of Parities
        gates.extend(
            [
                self._noise_model.idle_sq_gate("D4", t_start=t_ref),
                self._noise_model.rotate_y_90("Z1", t_start=t_ref),
                self._noise_model.idle_sq_gate("D1", t_start=t_ref),
                self._noise_model.rotate_y_m90("X1", t_start=t_ref),
                self._noise_model.idle_sq_gate("D2", t_start=t_ref),
            ]
        )
        t_ref += t_sqg
        # Z1-D1
        gates.extend(
            [
                self._noise_model.idle_tq_gate("D4", t_start=t_ref),
                self._noise_model.idle_tq_gate("X1", t_start=t_ref),
                self._noise_model.idle_tq_gate("D2", t_start=t_ref),
                self._noise_model.cphase("D1", "Z1", t_start=t_ref),
                self._noise_model.zz_coupling_2q("Z1", "D4", t_start=t_ref),
                self._noise_model.zz_coupling_2q("D2", "Z1", t_start=t_ref),
            ]
        )
        t_ref += t_tqg
        # Z1-D4
        gates.extend(
            [
                self._noise_model.idle_tq_gate("D1", t_start=t_ref),
                self._noise_model.idle_tq_gate("X1", t_start=t_ref),
                self._noise_model.idle_tq_gate("D2", t_start=t_ref),
                self._noise_model.cphase("Z1", "D4", t_start=t_ref),
                self._noise_model.zz_coupling_2q("D1", "Z1", t_start=t_ref),
                self._noise_model.zz_coupling_2q("D2", "Z1", t_start=t_ref),
            ]
        )
        t_ref += t_tqg
        # End of Parities
        gates.extend(
            [
                self._noise_model.idle_sq_gate("D4", t_start=t_ref),
                self._noise_model.rotate_y_m90("Z1", t_start=t_ref),
                self._noise_model.idle_sq_gate("D1", t_start=t_ref),
                self._noise_model.idle_sq_gate("X1", t_start=t_ref),
                self._noise_model.idle_sq_gate("D2", t_start=t_ref),
            ]
        )
        t_ref += t_sqg
        # Refocusing operations
        gates.extend(
            [
                self._noise_model.rotate_x_180("D4", t_start=t_ref),
                self._noise_model.idle_sq_gate("Z1", t_start=t_ref),
                self._noise_model.rotate_x_180("D1", t_start=t_ref),
                self._noise_model.idle_sq_gate("X1", t_start=t_ref),
                self._noise_model.rotate_x_180("D2", t_start=t_ref),
            ]
        )
        return Circuit(
            qubits=self._code_settings.qubits,
            gates=gates,
        )
    # endregion

    # region Static Class Methods
    @staticmethod
    def construct_detector_only_circuit(noise_model: DiCarloLabModel, parity_group: IParityGroup, code_layout: ISurfaceCodeLayer) -> Circuit:
        gates = []
        t_sqg: float = 20
        t_tqg: float = 60
        t_ref: float = 0
        # Start of Parities
        gates.extend(
            [
                noise_model.rotate_y_90(parity_group.ancilla_id.id, t_start=t_ref),
            ] + [
                noise_model.idle_sq_gate(data_qubit_id.id, t_start=t_ref)
                for data_qubit_id in parity_group.data_ids
            ]
        )
        t_ref += t_sqg
        # End of Parities
        gates.extend(
            [
                noise_model.rotate_y_m90(parity_group.ancilla_id.id, t_start=t_ref),
            ] + [
                noise_model.idle_sq_gate(data_qubit_id.id, t_start=t_ref)
                for data_qubit_id in parity_group.data_ids
            ]
        )
        t_ref += t_sqg
        # Buffer period
        for i, edge_id in enumerate(parity_group.edge_ids):
            zz_gates = []
            for zz_edge in parity_group.edge_ids:
                zz_msq: IQubitID = get_higher_frequency_qubit_id(zz_edge, code_layout)
                zz_lsq: IQubitID = get_lower_frequency_qubit_id(zz_edge, code_layout)
                zz_gates.append(noise_model.zz_coupling_2q(zz_lsq.id, zz_msq.id, t_start=t_ref))
            for qubit_id in [parity_group.ancilla_id] + parity_group.data_ids:
                zz_gates.append(noise_model.idle_tq_gate(qubit_id.id, t_start=t_ref))
            gates.extend(zz_gates)
            t_ref += t_tqg
        return Circuit(
            qubits=[parity_group.ancilla_id.id] + [qubit_id.id for qubit_id in parity_group.data_ids],
            gates=gates,
        )

    @staticmethod
    def construct_detector_inverse_parity_circuit(noise_model: DiCarloLabModel, parity_group: IParityGroup, code_layout: ISurfaceCodeLayer) -> Circuit:
        gates = []
        t_sqg: float = 20
        t_tqg: float = 60
        t_ref: float = 0
        # Start of Parities
        gates.extend(
            [
                noise_model.rotate_y_90(parity_group.ancilla_id.id, t_start=t_ref),
            ] + [
                noise_model.idle_sq_gate(data_qubit_id.id, t_start=t_ref)
                for data_qubit_id in parity_group.data_ids
            ]
        )
        t_ref += t_sqg
        # End of Parities
        gates.extend(
            [
                noise_model.rotate_y_90(parity_group.ancilla_id.id, t_start=t_ref),
            ] + [
                noise_model.idle_sq_gate(data_qubit_id.id, t_start=t_ref)
                for data_qubit_id in parity_group.data_ids
            ]
        )
        t_ref += t_sqg
        # Buffer period
        for i, edge_id in enumerate(parity_group.edge_ids):
            zz_gates = []
            for zz_edge in parity_group.edge_ids:
                zz_msq: IQubitID = get_higher_frequency_qubit_id(zz_edge, code_layout)
                zz_lsq: IQubitID = get_lower_frequency_qubit_id(zz_edge, code_layout)
                zz_gates.append(noise_model.zz_coupling_2q(zz_lsq.id, zz_msq.id, t_start=t_ref))
            for qubit_id in [parity_group.ancilla_id] + parity_group.data_ids:
                zz_gates.append(noise_model.idle_tq_gate(qubit_id.id, t_start=t_ref))
            gates.extend(zz_gates)
            t_ref += t_tqg
        return Circuit(
            qubits=[parity_group.ancilla_id.id] + [qubit_id.id for qubit_id in parity_group.data_ids],
            gates=gates,
        )

    @staticmethod
    def construct_decoherence_only_circuit(noise_model: DiCarloLabModel, parity_group: IParityGroup, code_layout: ISurfaceCodeLayer) -> Circuit:
        gates = []
        t_sqg: float = 20
        t_tqg: float = 60
        t_ref: float = 0
        # Start of Parities
        gates.extend(
            [
                noise_model.rotate_y_90(parity_group.ancilla_id.id, t_start=t_ref),
            ] + [
                noise_model.idle_sq_gate(data_qubit_id.id, t_start=t_ref)
                for data_qubit_id in parity_group.data_ids
            ]
        )
        t_ref += t_sqg
        # Middle of Parities
        for i, edge_id in enumerate(parity_group.edge_ids):
            zz_gates = []
            for zz_edge in parity_group.edge_ids:
                zz_msq: IQubitID = get_higher_frequency_qubit_id(zz_edge, code_layout)
                zz_lsq: IQubitID = get_lower_frequency_qubit_id(zz_edge, code_layout)
                zz_gates.append(noise_model.zz_coupling_2q(zz_lsq.id, zz_msq.id, t_start=t_ref))
            for qubit_id in [parity_group.ancilla_id] + parity_group.data_ids:
                zz_gates.append(noise_model.idle_tq_gate(qubit_id.id, t_start=t_ref))
            gates.extend(zz_gates)
            t_ref += t_tqg
        # End of Parities
        gates.extend(
            [
                noise_model.rotate_y_m90(parity_group.ancilla_id.id, t_start=t_ref),
            ] + [
                noise_model.idle_sq_gate(data_qubit_id.id, t_start=t_ref)
                for data_qubit_id in parity_group.data_ids
            ]
        )
        t_ref += t_sqg
        return Circuit(
            qubits=[parity_group.ancilla_id.id] + [qubit_id.id for qubit_id in parity_group.data_ids],
            gates=gates,
        )

    @staticmethod
    def construct_relaxation_data_constant_circuit(noise_model: DiCarloLabModel, parity_group: IParityGroup, code_layout: ISurfaceCodeLayer) -> Circuit:
        gates = []
        t_sqg: float = 20
        t_tqg: float = 60
        t_ref: float = 0
        # Start of Parities
        gates.extend(
            [
                noise_model.rotate_y_90(parity_group.ancilla_id.id, t_start=t_ref),
            ] + [
                noise_model.idle_sq_gate(data_qubit_id.id, t_start=t_ref)
                for data_qubit_id in parity_group.data_ids
            ]
        )
        t_ref += t_sqg
        # Middle of Parities
        for i, edge_id in enumerate(parity_group.edge_ids):
            zz_gates = []
            for zz_edge in parity_group.edge_ids:
                zz_msq: IQubitID = get_higher_frequency_qubit_id(zz_edge, code_layout)
                zz_lsq: IQubitID = get_lower_frequency_qubit_id(zz_edge, code_layout)
                zz_gates.append(noise_model.zz_coupling_2q(zz_lsq.id, zz_msq.id, t_start=t_ref))
            for qubit_id in [parity_group.ancilla_id] + parity_group.data_ids:
                zz_gates.append(noise_model.idle_tq_gate(qubit_id.id, t_start=t_ref))
            gates.extend(zz_gates)
            t_ref += t_tqg
        # End of Parities
        gates.extend(
            [
                noise_model.rotate_y_m90(parity_group.ancilla_id.id, t_start=t_ref),
            ] + [
                noise_model.rotate_x_180(data_qubit_id.id, t_start=t_ref)
                for data_qubit_id in parity_group.data_ids
            ]
        )
        t_ref += t_sqg
        return Circuit(
            qubits=[parity_group.ancilla_id.id] + [qubit_id.id for qubit_id in parity_group.data_ids],
            gates=gates,
        )

    @staticmethod
    def construct_parity_circuit(noise_model: DiCarloLabModel, parity_group: IParityGroup, code_layout: ISurfaceCodeLayer) -> Circuit:
        gates = []
        t_sqg: float = 20
        t_tqg: float = 60
        t_ref: float = 0
        # Start of Parities
        gates.extend(
            [
                noise_model.rotate_y_90(parity_group.ancilla_id.id, t_start=t_ref),
            ] + [
                noise_model.idle_sq_gate(data_qubit_id.id, t_start=t_ref)
                for data_qubit_id in parity_group.data_ids
            ]
        )
        t_ref += t_sqg
        # Middle of Parities
        for i, edge_id in enumerate(parity_group.edge_ids):
            msq: IQubitID = get_higher_frequency_qubit_id(edge_id, code_layout)
            lsq: IQubitID = get_lower_frequency_qubit_id(edge_id, code_layout)
            cphase_gate = [noise_model.cphase(lsq.id, msq.id, t_start=t_ref)]
            zz_edges: List[IEdgeID] = [_edge_id for _edge_id in parity_group.edge_ids if _edge_id is not edge_id]
            zz_gates = []
            for zz_edge in zz_edges:
                zz_msq: IQubitID = get_higher_frequency_qubit_id(zz_edge, code_layout)
                zz_lsq: IQubitID = get_lower_frequency_qubit_id(zz_edge, code_layout)
                zz_gates.append(noise_model.zz_coupling_2q(zz_lsq.id, zz_msq.id, t_start=t_ref))
            for qubit_id in [parity_group.ancilla_id] + parity_group.data_ids:
                zz_gates.append(noise_model.idle_tq_gate(qubit_id.id, t_start=t_ref))

            gates.extend(cphase_gate + zz_gates)
            t_ref += t_tqg
        # End of Parities
        gates.extend(
            [
                noise_model.rotate_y_m90(parity_group.ancilla_id.id, t_start=t_ref),
            ] + [
                noise_model.rotate_x_180(data_qubit_id.id, t_start=t_ref)
                for data_qubit_id in parity_group.data_ids
            ]
        )
        t_ref += t_sqg
        return Circuit(
            qubits=[parity_group.ancilla_id.id] + [qubit_id.id for qubit_id in parity_group.data_ids],
            gates=gates,
        )

    @staticmethod
    def tensor_to_error_identifier(
            result_tensor: np.ndarray,
            depth: int,
            repetitions: int,
            initial_state: InitialStateContainer,
            code_layout: ISurfaceCodeLayer,
            involved_qubits_qubit_ids: List[IQubitID],  # Ordered
            **kwargs,
    ) -> ErrorDetectionIdentifier:
        # Extract keyword arguments
        inverse_parity: bool = kwargs.get("inverse_parity", False)
        stabilizer_active: bool = kwargs.get("stabilizer_active", True)

        # Data allocation
        involved_data_qubit_ids: List[IQubitID] = [
            qubit_id
            for qubit_id in involved_qubits_qubit_ids
            if qubit_id in code_layout.data_qubit_ids
        ]
        involved_ancilla_qubit_ids: List[IQubitID] = [
            qubit_id
            for qubit_id in involved_qubits_qubit_ids
            if qubit_id in code_layout.ancilla_qubit_ids
        ]
        expected_parity = initial_state_to_expected_parity(
            initial_state=initial_state,
            parity_layout=code_layout,
            involved_data_qubit_ids=involved_data_qubit_ids,
            involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
            inverse_parity=inverse_parity,
            stabilizer_active=stabilizer_active,
        )

        classifier_lookup = {}
        for i, qubit_id in enumerate(involved_qubits_qubit_ids):
            state_classification = result_tensor[:, :, i].flatten()
            state_classification[state_classification == 2] = (
                1  # Map 0, 1, 2 outcomes to 0, 1
            )

            classifier_lookup[qubit_id] = StateClassifierContainer(
                state_classification=state_classification,
                _expected_parity=(
                    ParityType.EVEN
                    if qubit_id not in expected_parity
                    else expected_parity[qubit_id]
                ),
                _stabilizer_reset=False,
            )

        return ErrorDetectionIdentifier(
            classifier_lookup=classifier_lookup,
            index_kernel=SimulatedRepetitionExperimentKernel(
                rounds=[depth],
                involved_data_qubit_ids=involved_data_qubit_ids,
                involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
                experiment_repetitions=repetitions,
            ),
            involved_qubit_ids=involved_qubits_qubit_ids,
            device_layout=code_layout,
            qec_rounds=[depth],
            use_heralded_post_selection=False,
            use_projected_leakage_post_selection=False,
            use_stabilizer_leakage_post_selection=False,
            post_selection_qubits=None,
            use_computational_parity=True,
        )

    @staticmethod
    def error_identifier_to_dataset(
            error_identifier: ErrorDetectionIdentifier,
            initial_state: InitialStateContainer,
            code_layout: ISurfaceCodeLayer,
            noise_config_string: str,
    ) -> xr.Dataset:
        dataset: xr.Dataset = construct_processed_dataset(
            error_identifier=error_identifier,
            initial_state=initial_state,
            qec_rounds=error_identifier.qec_rounds,
            code_layout=code_layout,
        )
        dataset.attrs["noise_config"] = noise_config_string

        return dataset

    @staticmethod
    def save_dataset_to_netcdf(
            dataset: xr.Dataset,
            file_path: Path,
            overwrite: bool = False,
            **kwargs: Any
    ) -> None:
        """
        Saves an xarray.Dataset to a NetCDF file.

        :param dataset: The xarray.Dataset to be saved.
        :param file_path: The full path (including filename) for the output .nc file.
        :param overwrite: If True, it will overwrite the file if it already exists.
                          If False, it will raise a FileExistsError.
        :param kwargs: Additional keyword arguments to be passed directly to the
                       xarray.Dataset.to_netcdf() method (e.g., 'engine', 'encoding').
        :return: None. The function saves the file as a side effect.
        :raises FileExistsError: If the file already exists and overwrite is False.
        :raises IOError: For other file system-related errors.
        """
        # Check if the file exists and handle the overwrite logic
        if overwrite:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError(
                f"File '{file_path}' already exists. Set overwrite=True to replace it."
            )
        try:
            # Save the dataset to a NetCDF file
            dataset.to_netcdf(path=file_path, **kwargs)
            print(f"Dataset successfully saved to: {file_path}")
        except IOError as e:
            print(f"An I/O error occurred while saving the file: {e}")
            raise
    # endregion


if __name__ == '__main__':
    from qce_interp.visualization.plot_defect_rate import (
        plot_all_defect_rate,
    )
    import matplotlib.pyplot as plt

    noise_file = "setup_noisy.yaml"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = Path(os.path.join(current_dir, ".."))
    setup = Setup.from_file(os.path.join(ROOT_DIR, noise_file))
    model = DiCarloLabModel(setup=setup)

    meas_ctrl = MeasurementControl(
        noise_model=model,
        code_settings=CodeSettings.from_code_type(RepetitionCodeType.DISTANCE_3)
    )

    common_path: Path = ROOT_DIR / "simulated_datasets"
    qec_rounds: int = 10
    error_probabilities = np.linspace(0.0, 0.1, 5)
    list_of_datasets: List[xr.Dataset] = []
    for i, error_probability in enumerate(error_probabilities):
        error_identifier: ErrorDetectionIdentifier = meas_ctrl.construct_error_identifier(
            cycle_circuit=meas_ctrl.construct_distance_3_bitflip_circuit(
                error_probability=float(error_probability),
                error_coherence=True,
            ),
            depth=qec_rounds,
            repetitions=10_000,
        )

        dataset = meas_ctrl.construct_dataset(
            error_identifier=error_identifier,
            noise_file_path=ROOT_DIR / noise_file,
        )
        dataset.attrs["error_probability"] = error_probability
        dataset.attrs["error_coherence"] = True
        list_of_datasets.append(dataset)

        # print(dataset)
        # meas_ctrl.save_dataset_to_netcdf(dataset, common_path / f"defect_rate_d3_p{i}-{len(error_probabilities)}.nc", overwrite=True)
        # fig, ax = plot_all_defect_rate(
        #     error_identifier=error_identifier,
        #     included_rounds=qec_rounds,
        # )

    # Concatenate along a new, unnamed dimension
    combined_stepwise: xr.Dataset = xr.concat(list_of_datasets, dim="error_probability")
    # Assign the coordinate values to the new dimension
    combined_stepwise.coords["error_probability"] = error_probabilities
    meas_ctrl.save_dataset_to_netcdf(combined_stepwise, common_path / f"defect_rate_d3_stacked.nc", overwrite=True)

    # plt.show()
