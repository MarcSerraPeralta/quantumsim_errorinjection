import os
from dataclasses import dataclass
from itertools import chain, product
from typing import List, Optional, Any, Tuple, Dict
import xarray as xr
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from pathlib import Path
from enum import Enum, unique, auto
from quantumsim.circuits import Circuit, FinalizedCircuit
from quantumsim import Setup
from quantumsim.states.state import State
from quantumsim.models import DiCarloLabModel
from quantumsim.operations.state_projector import StateProjector
from quantumsim.utilities.process_state_projection import (
    process_state_projection,
    process_assignment_errors,
)
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
from quantumsim.utilities.process_initial_circuit import (
    RepetitionCodeDistance3,
    RepetitionCodeDistance5,
)


IQubitList = List[str]
IGateList = List[Any]
QEC_DURATION_D3: float = 600
QEC_DURATION_D5: float = 880


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
    def initial_circuit_gates(self) -> List[Any]:
        return self.construct_initial_circuit_gates(self._initial_state_type)

    @property
    def initial_state(self) -> InitialStateContainer:
        return self.construct_initial_state(self._initial_state_type)
    # endregion

    # region Class Constructor
    def __init__(self, noise_model: DiCarloLabModel, code_settings: CodeSettings, dim_lookup: Dict[str, int], initial_state_type: InitialStateType = InitialStateType.ODD):
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
        self._dim_lookup = dim_lookup  # {q: state_dim for q in self._code_settings.qubits}
    # endregion

    # region Class Methods
    def construct_initial_circuit_gates(self, initial_state_type: InitialStateType) -> List[Any]:
        gates = []
        t_ref: float = 0
        # All zero (Even) initial state
        if initial_state_type in [InitialStateType.ALL_ZERO, InitialStateType.EVEN]:
            for qubit in self._code_settings.qubits:
                gates.append(self._noise_model.idle_sq_gate(qubit, t_start=t_ref))
            return gates
        # All ones initial state
        if initial_state_type == InitialStateType.ALL_ONES:
            for qubit in self._code_settings.qubits:
                if QubitIDObj(qubit) in self._code_settings.code_layout.data_qubit_ids:
                    gates.append(self._noise_model.rotate_x_180(qubit, t_start=t_ref))
                else:
                    gates.append(self._noise_model.idle_sq_gate(qubit, t_start=t_ref))

            return gates
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
            return gates
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

    def execute_simulation_loop(self, cycle_circuit_gates: IGateList, qec_rounds: int = 10, repetitions: int = 10):
        # flatten gate list
        init_gates = list(chain(*self.initial_circuit_gates))
        qec_gates = list(chain(*cycle_circuit_gates))
        qubits: IQubitList = self._code_settings.qubits
        dim_lookup: Dict[str, int] = self._dim_lookup

        meas_outcome_indices: NDArray[int] = np.zeros(
            shape=(
                repetitions,
                qec_rounds + 1,  # +1 for data-qubit measurement storage
            ),
            dtype=int,
        )

        # Populate measurement outcome indices
        for i in tqdm(range(repetitions), desc="Iterate Repetitions", disable=False):
            state: State = State(
                qubits, dim_lookup=dim_lookup, force=True
            )  # initialization in ground state

            # apply init gates
            for indexedop in init_gates:
                indices = [qubits.index(q) for q in indexedop.indices]
                indexedop.operation(state.pauli_vector, *indices)

            # Iterate through qec-cycles
            for j in range(qec_rounds + 1):
                outcomes = []
                last_qec_round: bool = j == qec_rounds

                # Determine state projector (round dependent)
                current_state_projector = self.final_state_projector
                if not last_qec_round:
                    # apply QEC cycle gates
                    for indexedop in qec_gates:
                        indices = [qubits.index(q) for q in indexedop.indices]
                        indexedop.operation(state.pauli_vector, *indices)
                    current_state_projector = self.state_projector

                # extract ancilla outcome probabilities from the diagonal
                probs = state.pauli_vector.diagonal()
                probs = np.clip(probs, 0.0, 1.0)
                probs /= np.sum(probs)
                # fake outcomes, as not all qubits are measured
                list_outcomes = list(product(*[list(range(dim_lookup[q])) for q in qubits]))
                outcome_idx = np.random.choice(len(list_outcomes), p=probs)
                outcome = list_outcomes[outcome_idx]
                # 'trace out' all the non-measured qubits
                outcome = [
                    o
                    for q, o in zip(qubits, outcome)
                    if q in current_state_projector.ordered_measured_qubit_ids
                ]
                outcomes += outcome

                # update state from sampled outcomes
                meas_outcome = "".join(
                    map(
                        str,
                        outcomes[-len(current_state_projector.ordered_measured_qubit_ids):],
                    )
                )
                meas_outcome_index: int = (
                    current_state_projector.output_state_vector_keys.index(meas_outcome)
                )
                meas_outcome_indices[i, j] = meas_outcome_index
                if not last_qec_round:
                    meas_project_gates = (
                        current_state_projector.get_meas_outcome_to_circuit_gates(meas_outcome)
                    )
                    # apply measurement projection
                    for indexedop in meas_project_gates:
                        indices = [qubits.index(q) for q in indexedop.indices]
                        indexedop.operation(state.pauli_vector, *indices)
                    state.pauli_vector.renormalize()

        return meas_outcome_indices

    def construct_error_identifier(self, cycle_circuit_gates: IGateList, depth: int = 10, repetitions: int = 10, **error_identifier_construction_kwargs) -> Tuple[ErrorDetectionIdentifier, xr.DataArray]:
        # Construct initial state
        meas_outcome_indices: np.ndarray = self.execute_simulation_loop(
            cycle_circuit_gates=cycle_circuit_gates,
            qec_rounds=depth,
            repetitions=repetitions,
        )
        assigned_meas_outcome_indices = process_assignment_errors(
            meas_outcome_indices=meas_outcome_indices,
            state_projector=self.state_projector,
            final_state_projector=self.final_state_projector,
        )
        result_tensor = process_state_projection(
            state_projection_array=assigned_meas_outcome_indices,
            state_projector=self.state_projector,
            final_state_projector=self.final_state_projector,
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
        # Construct raw DataArray
        n_reps, n_cycles, n_qubits = result_tensor.shape
        simulated_result_dataset = xr.DataArray(
            data=result_tensor,
            dims=['repetition', 'qec_cycles_array', 'qubit_index'],
            coords={
                'repetition': np.arange(n_reps),
                'qec_cycles_array': np.arange(n_cycles),
                'qubit_index': np.arange(n_qubits)
            },
            name="raw_simulation_outcomes",
            attrs={
                "qubits_ordered": [q.id for q in error_identifier.involved_qubit_ids],
                "initial_state": self.initial_state.as_array,
            }
        )
        return error_identifier, simulated_result_dataset

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

    def construct_distance_3_bitflip_circuit_gates_global_phase_error(self, phi_rate: float = 0.0, error_probability: float = 0.0, error_coherence: bool = True, t_ref: float = 0) -> Tuple[IQubitList, IGateList]:
        combined_gates = []
        t_sqg: float = 0
        rotation_degree_per_qec_cycle: float = np.rad2deg(phi_rate * QEC_DURATION_D3 * 2 * np.pi)  # rotation rate [Hz] * QEC_DURATION [s] * 2pi * rad2deg
        combined_gates.extend([
            self._noise_model.rotate_z("D4", phi=rotation_degree_per_qec_cycle, t_start=t_ref),
            # self._noise_model.rotate_z("Z1", phi=rotation_degree_per_qec_cycle, t_start=t_ref),
            self._noise_model.rotate_z("D1", phi=rotation_degree_per_qec_cycle, t_start=t_ref),
            # self._noise_model.rotate_z("X1", phi=rotation_degree_per_qec_cycle, t_start=t_ref),
            self._noise_model.rotate_z("D2", phi=rotation_degree_per_qec_cycle, t_start=t_ref),
        ])
        t_ref += t_sqg
        qubits, gates = self.construct_distance_3_bitflip_circuit_gates(
            error_probability=error_probability,
            error_coherence=error_coherence,
            t_ref=t_ref,
        )
        combined_gates.extend(gates)
        return qubits, combined_gates

    def construct_distance_3_bitflip_circuit_gates_fermioniclike(self, error_probability: float = 0.0, error_coherence: bool = True, t_ref: float = 0) -> Tuple[IQubitList, IGateList]:
        combined_gates = []
        t_sqg: float = 0
        combined_gates.extend([
            self._noise_model.random_rotate_x_180_t1like("D4", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.random_rotate_x_180_t1like("Z1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.random_rotate_x_180_t1like("D1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.random_rotate_x_180_t1like("X1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.random_rotate_x_180_t1like("D2", t=QEC_DURATION_D3, t_start=t_ref),
        ])
        t_ref += t_sqg
        qubits, gates = self.construct_distance_3_bitflip_circuit_gates(
            error_probability=error_probability,
            error_coherence=error_coherence,
            t_ref=t_ref,
        )
        combined_gates.extend(gates)
        return qubits, combined_gates

    def construct_distance_3_bitflip_circuit_gates_depolarizing(self, error_probability: float = 0.0, error_coherence: bool = True, t_ref: float = 0) -> Tuple[IQubitList, IGateList]:
        combined_gates = []
        t_sqg: float = 0
        combined_gates.extend([
            self._noise_model.depolarizing_channel("D4", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.depolarizing_channel("Z1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.depolarizing_channel("D1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.depolarizing_channel("X1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.depolarizing_channel("D2", t=QEC_DURATION_D3, t_start=t_ref),
        ])
        t_ref += t_sqg
        qubits, gates = self.construct_distance_3_bitflip_circuit_gates(
            error_probability=error_probability,
            error_coherence=error_coherence,
            t_ref=t_ref,
        )
        combined_gates.extend(gates)
        return qubits, combined_gates

    def construct_distance_3_bitflip_circuit_gates_amplitude_damping(self, error_probability: float = 0.0, error_coherence: bool = True, t_ref: float = 0) -> Tuple[IQubitList, IGateList]:
        combined_gates = []
        t_sqg: float = 0
        combined_gates.extend([
            self._noise_model.amplitude_damping_01("D4", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.amplitude_damping_01("Z1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.amplitude_damping_01("D1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.amplitude_damping_01("X1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.amplitude_damping_01("D2", t=QEC_DURATION_D3, t_start=t_ref),
        ])
        t_ref += t_sqg
        qubits, gates = self.construct_distance_3_bitflip_circuit_gates(
            error_probability=error_probability,
            error_coherence=error_coherence,
            t_ref=t_ref,
        )
        combined_gates.extend(gates)
        return qubits, combined_gates

    def construct_distance_3_bitflip_circuit_gates_phase_damping(self, error_probability: float = 0.0, error_coherence: bool = True, t_ref: float = 0) -> Tuple[IQubitList, IGateList]:
        combined_gates = []
        t_sqg: float = 0
        combined_gates.extend([
            self._noise_model.amplitude_damping_01("D4", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.amplitude_damping_01("Z1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.amplitude_damping_01("D1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.amplitude_damping_01("X1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.amplitude_damping_01("D2", t=QEC_DURATION_D3, t_start=t_ref),
        ])
        combined_gates.extend([
            self._noise_model.phase_damping_01("D4", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.phase_damping_01("Z1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.phase_damping_01("D1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.phase_damping_01("X1", t=QEC_DURATION_D3, t_start=t_ref),
            self._noise_model.phase_damping_01("D2", t=QEC_DURATION_D3, t_start=t_ref),
        ])
        t_ref += t_sqg
        qubits, gates = self.construct_distance_3_bitflip_circuit_gates(
            error_probability=error_probability,
            error_coherence=error_coherence,
            t_ref=t_ref,
        )
        combined_gates.extend(gates)
        return qubits, combined_gates

    def construct_distance_3_bitflip_circuit_gates(self, error_probability: float = 0.0, error_coherence: bool = True, t_ref: float = 0) -> Tuple[IQubitList, IGateList]:
        gates = []
        t_sqg: float = 20
        t_tqg: float = 60
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
            gates.extend(
                [
                    self._noise_model.random_rotate_x_180("D4", t_start=t_ref, prob=error_probability),
                    self._noise_model.random_rotate_x_180("Z1", t_start=t_ref, prob=error_probability),
                    self._noise_model.random_rotate_x_180("D1", t_start=t_ref, prob=error_probability),
                    self._noise_model.random_rotate_x_180("X1", t_start=t_ref, prob=error_probability),
                    self._noise_model.random_rotate_x_180("D2", t_start=t_ref, prob=error_probability),
                ]
            )
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
        return self._code_settings.qubits, gates

    def construct_distance_5_bitflip_circuit_gates_fermioniclike(self, error_probability: float = 0.0, error_coherence: bool = True, t_ref: float = 0) -> Tuple[IQubitList, IGateList]:
        combined_gates = []
        t_sqg: float = 0
        combined_gates.extend([
            self._noise_model.random_rotate_x_180_t1like("D7", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.random_rotate_x_180_t1like("D4", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.random_rotate_x_180_t1like("D1", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.random_rotate_x_180_t1like("D2", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.random_rotate_x_180_t1like("D3", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.random_rotate_x_180_t1like("Z3", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.random_rotate_x_180_t1like("Z1", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.random_rotate_x_180_t1like("X1", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.random_rotate_x_180_t1like("X2", t=QEC_DURATION_D5, t_start=t_ref),
        ])
        t_ref += t_sqg
        qubits, gates = self.construct_distance_5_bitflip_circuit_gates(
            error_probability=error_probability,
            error_coherence=error_coherence,
            t_ref=t_ref,
        )
        combined_gates.extend(gates)
        return qubits, combined_gates

    def construct_distance_5_bitflip_circuit_gates_depolarizing(self, error_probability: float = 0.0, error_coherence: bool = True, t_ref: float = 0) -> Tuple[IQubitList, IGateList]:
        combined_gates = []
        t_sqg: float = 0
        combined_gates.extend([
            self._noise_model.depolarizing_channel("D7", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.depolarizing_channel("D4", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.depolarizing_channel("D1", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.depolarizing_channel("D2", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.depolarizing_channel("D3", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.depolarizing_channel("Z3", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.depolarizing_channel("Z1", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.depolarizing_channel("X1", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.depolarizing_channel("X2", t=QEC_DURATION_D5, t_start=t_ref),
        ])
        t_ref += t_sqg
        qubits, gates = self.construct_distance_5_bitflip_circuit_gates(
            error_probability=error_probability,
            error_coherence=error_coherence,
            t_ref=t_ref,
        )
        combined_gates.extend(gates)
        return qubits, combined_gates

    def construct_distance_5_bitflip_circuit_gates_amplitude_damping(self, error_probability: float = 0.0, error_coherence: bool = True, t_ref: float = 0) -> Tuple[IQubitList, IGateList]:
        combined_gates = []
        t_sqg: float = 0
        combined_gates.extend([
            self._noise_model.amplitude_damping_01("D7", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.amplitude_damping_01("D4", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.amplitude_damping_01("D1", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.amplitude_damping_01("D2", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.amplitude_damping_01("D3", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.amplitude_damping_01("Z3", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.amplitude_damping_01("Z1", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.amplitude_damping_01("X1", t=QEC_DURATION_D5, t_start=t_ref),
            self._noise_model.amplitude_damping_01("X2", t=QEC_DURATION_D5, t_start=t_ref),
        ])
        t_ref += t_sqg
        qubits, gates = self.construct_distance_5_bitflip_circuit_gates(
            error_probability=error_probability,
            error_coherence=error_coherence,
            t_ref=t_ref,
        )
        combined_gates.extend(gates)
        return qubits, combined_gates

    def construct_distance_5_bitflip_circuit_gates(self, error_probability: float = 0.0, error_coherence: bool = True, t_ref: float = 0) -> Tuple[IQubitList, IGateList]:
        gates = []
        t_sqg: float = 20
        t_tqg: float = 60
        if error_coherence:
            theta: float = np.rad2deg(2 * np.arcsin(np.sqrt(error_probability)))
            gates.extend(
                [
                    self._noise_model.rotate_x("D7", t_start=t_ref, theta=theta),
                    self._noise_model.rotate_x("D4", t_start=t_ref, theta=theta),
                    self._noise_model.rotate_x("D1", t_start=t_ref, theta=theta),
                    self._noise_model.rotate_x("D2", t_start=t_ref, theta=theta),
                    self._noise_model.rotate_x("D3", t_start=t_ref, theta=theta),
                    self._noise_model.rotate_x("Z3", t_start=t_ref, theta=theta),
                    self._noise_model.rotate_x("Z1", t_start=t_ref, theta=theta),
                    self._noise_model.rotate_x("X1", t_start=t_ref, theta=theta),
                    self._noise_model.rotate_x("X2", t_start=t_ref, theta=theta),
                ]
            )
        else:
            gates.extend(
                [
                    self._noise_model.random_rotate_x_180("D7", t_start=t_ref, prob=error_probability),
                    self._noise_model.random_rotate_x_180("D4", t_start=t_ref, prob=error_probability),
                    self._noise_model.random_rotate_x_180("D1", t_start=t_ref, prob=error_probability),
                    self._noise_model.random_rotate_x_180("D2", t_start=t_ref, prob=error_probability),
                    self._noise_model.random_rotate_x_180("D3", t_start=t_ref, prob=error_probability),
                    self._noise_model.random_rotate_x_180("Z3", t_start=t_ref, prob=error_probability),
                    self._noise_model.random_rotate_x_180("Z1", t_start=t_ref, prob=error_probability),
                    self._noise_model.random_rotate_x_180("X1", t_start=t_ref, prob=error_probability),
                    self._noise_model.random_rotate_x_180("X2", t_start=t_ref, prob=error_probability),
                ]
            )
            pass

        gates.extend(
            [
                self._noise_model.idle_sq_gate("D7", t_start=t_ref),
                self._noise_model.idle_sq_gate("D4", t_start=t_ref),
                self._noise_model.idle_sq_gate("D1", t_start=t_ref),
                self._noise_model.idle_sq_gate("D2", t_start=t_ref),
                self._noise_model.idle_sq_gate("D3", t_start=t_ref),
                self._noise_model.rotate_y_90("Z3", t_start=t_ref),
                self._noise_model.idle_sq_gate("Z1", t_start=t_ref),
                self._noise_model.rotate_y_90("X1", t_start=t_ref),
                self._noise_model.idle_sq_gate("X2", t_start=t_ref),
            ]
        )
        t_ref += t_sqg

        gates.extend(
            [
                self._noise_model.idle_tq_gate("D7", t_start=t_ref),
                self._noise_model.cphase("Z3", "D4", t_start=t_ref),
                self._noise_model.idle_tq_gate("D1", t_start=t_ref),
                self._noise_model.cphase("D2", "X1", t_start=t_ref),
                self._noise_model.idle_tq_gate("D3", t_start=t_ref),
                self._noise_model.idle_tq_gate("Z1", t_start=t_ref),
                self._noise_model.idle_tq_gate("X2", t_start=t_ref),
                self._noise_model.zz_coupling_2q("Z3", "D7", t_start=t_ref),
                self._noise_model.zz_coupling_2q("X1", "D1", t_start=t_ref),
            ]
        )
        t_ref += t_tqg

        gates.extend(
            [
                self._noise_model.cphase("D7", "Z3", t_start=t_ref),
                self._noise_model.idle_tq_gate("D4", t_start=t_ref),
                self._noise_model.cphase("D1", "X1", t_start=t_ref),
                self._noise_model.idle_tq_gate("D2", t_start=t_ref),
                self._noise_model.idle_tq_gate("D3", t_start=t_ref),
                self._noise_model.idle_tq_gate("Z1", t_start=t_ref),
                self._noise_model.idle_tq_gate("X2", t_start=t_ref),
                self._noise_model.zz_coupling_2q("Z3", "D4", t_start=t_ref),
                self._noise_model.zz_coupling_2q("X1", "D2", t_start=t_ref),
            ]
        )
        t_ref += t_tqg

        gates.extend(
            [
                self._noise_model.idle_sq_gate("D7", t_start=t_ref),
                self._noise_model.idle_sq_gate("D4", t_start=t_ref),
                self._noise_model.idle_sq_gate("D1", t_start=t_ref),
                self._noise_model.idle_sq_gate("D2", t_start=t_ref),
                self._noise_model.idle_sq_gate("D3", t_start=t_ref),
                self._noise_model.rotate_y_m90("Z3", t_start=t_ref),
                self._noise_model.rotate_y_90("Z1", t_start=t_ref),
                self._noise_model.rotate_y_m90("X1", t_start=t_ref),
                self._noise_model.rotate_y_90("X2", t_start=t_ref),
            ]
        )
        t_ref += t_sqg

        gates.extend(
            [
                self._noise_model.idle_tq_gate("D7", t_start=t_ref),
                self._noise_model.cphase("Z1", "D4", t_start=t_ref),
                self._noise_model.idle_tq_gate("D1", t_start=t_ref),
                self._noise_model.cphase("D2", "X2", t_start=t_ref),
                self._noise_model.idle_tq_gate("D3", t_start=t_ref),
                self._noise_model.idle_tq_gate("Z3", t_start=t_ref),
                self._noise_model.idle_tq_gate("X1", t_start=t_ref),
                self._noise_model.zz_coupling_2q("Z1", "D1", t_start=t_ref),
                self._noise_model.zz_coupling_2q("Z1", "D2", t_start=t_ref),
                self._noise_model.zz_coupling_2q("X2", "D3", t_start=t_ref),
            ]
        )
        t_ref += t_tqg

        gates.extend(
            [
                self._noise_model.idle_tq_gate("D7", t_start=t_ref),
                self._noise_model.idle_tq_gate("D4", t_start=t_ref),
                self._noise_model.cphase("D1", "Z1", t_start=t_ref),
                self._noise_model.idle_tq_gate("D2", t_start=t_ref),
                self._noise_model.cphase("D3", "X2", t_start=t_ref),
                self._noise_model.idle_tq_gate("Z3", t_start=t_ref),
                self._noise_model.idle_tq_gate("X1", t_start=t_ref),
                self._noise_model.zz_coupling_2q("Z1", "D4", t_start=t_ref),
                self._noise_model.zz_coupling_2q("Z1", "D2", t_start=t_ref),
                self._noise_model.zz_coupling_2q("X2", "D2", t_start=t_ref),
            ]
        )
        t_ref += t_tqg

        gates.extend(
            [
                self._noise_model.idle_sq_gate("D7", t_start=t_ref),
                self._noise_model.idle_sq_gate("D4", t_start=t_ref),
                self._noise_model.idle_sq_gate("D1", t_start=t_ref),
                self._noise_model.idle_sq_gate("D2", t_start=t_ref),
                self._noise_model.idle_sq_gate("D3", t_start=t_ref),
                self._noise_model.idle_sq_gate("Z3", t_start=t_ref),
                self._noise_model.rotate_y_m90("Z1", t_start=t_ref),
                self._noise_model.idle_sq_gate("X1", t_start=t_ref),
                self._noise_model.rotate_y_m90("X2", t_start=t_ref),
            ]
        )
        t_ref += t_sqg

        gates.extend(
            [
                self._noise_model.rotate_x_180("D7", t_start=t_ref),
                self._noise_model.rotate_x_180("D4", t_start=t_ref),
                self._noise_model.rotate_x_180("D1", t_start=t_ref),
                self._noise_model.rotate_x_180("D2", t_start=t_ref),
                self._noise_model.rotate_x_180("D3", t_start=t_ref),
                self._noise_model.idle_sq_gate("Z3", t_start=t_ref),
                self._noise_model.idle_sq_gate("Z1", t_start=t_ref),
                self._noise_model.idle_sq_gate("X1", t_start=t_ref),
                self._noise_model.idle_sq_gate("X2", t_start=t_ref),
            ]
        )

        return self._code_settings.qubits, gates
    # endregion

    # region Static Class Methods
    @staticmethod
    def construct_detector_only_circuit_gates(noise_model: DiCarloLabModel, parity_group: IParityGroup, code_layout: ISurfaceCodeLayer) -> Tuple[IQubitList, IGateList]:
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
        return [parity_group.ancilla_id.id] + [qubit_id.id for qubit_id in parity_group.data_ids], gates

    @staticmethod
    def construct_detector_inverse_parity_circuit_gates(noise_model: DiCarloLabModel, parity_group: IParityGroup, code_layout: ISurfaceCodeLayer) -> Tuple[IQubitList, IGateList]:
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
        return [parity_group.ancilla_id.id] + [qubit_id.id for qubit_id in parity_group.data_ids], gates

    @staticmethod
    def construct_decoherence_only_circuit_gates(noise_model: DiCarloLabModel, parity_group: IParityGroup, code_layout: ISurfaceCodeLayer) -> Tuple[IQubitList, IGateList]:
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
        return [parity_group.ancilla_id.id] + [qubit_id.id for qubit_id in parity_group.data_ids], gates

    @staticmethod
    def construct_relaxation_data_constant_circuit_gates(noise_model: DiCarloLabModel, parity_group: IParityGroup, code_layout: ISurfaceCodeLayer) -> Tuple[IQubitList, IGateList]:
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
        return [parity_group.ancilla_id.id] + [qubit_id.id for qubit_id in parity_group.data_ids], gates

    @staticmethod
    def construct_parity_circuit_args(noise_model: DiCarloLabModel, parity_group: IParityGroup, code_layout: ISurfaceCodeLayer) -> Tuple[IQubitList, IGateList]:
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
        return [parity_group.ancilla_id.id] + [qubit_id.id for qubit_id in parity_group.data_ids], gates

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
            #include_partitioned_fidelities=True,
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
    state_dim: int = 3
    dim_lookup = {
        "D7": 2,
        "Z3": state_dim,
        "D4": state_dim,
        "Z1": state_dim,
        "D1": 2,
        "X1": state_dim,
        "D2": 2,
        "X2": state_dim,
        "D3": 2,
    }
    model = DiCarloLabModel(
        setup=setup,
        dim_lookup=dim_lookup,
    )

    meas_ctrl = MeasurementControl(
        noise_model=model,
        code_settings=CodeSettings.from_code_type(RepetitionCodeType.DISTANCE_5),
        dim_lookup=dim_lookup,
        initial_state_type=InitialStateType.ODD,
    )

    common_path: Path = ROOT_DIR / "simulated_datasets"
    qec_rounds: int = 10
    repetitions = 10_000
    error_probabilities = np.linspace(0.0, 0.1, 3)
    list_of_datasets: List[xr.Dataset] = []
    for i, error_probability in enumerate(error_probabilities):
        involved_qubits, gates = meas_ctrl.construct_distance_5_bitflip_circuit_gates(
            error_probability=float(error_probability),
            error_coherence=True,
        )
        error_identifier: ErrorDetectionIdentifier = meas_ctrl.construct_error_identifier(
            cycle_circuit_gates=gates,
            depth=qec_rounds,
            repetitions=repetitions,
        )

        dataset = meas_ctrl.construct_dataset(
            error_identifier=error_identifier,
            noise_file_path=ROOT_DIR / noise_file,
        )
        dataset.attrs["error_probability"] = error_probability
        dataset.attrs["error_coherence"] = True
        list_of_datasets.append(dataset)

        # print(dataset)
        meas_ctrl.save_dataset_to_netcdf(dataset, common_path / f"defect_rate_d5_p{i}-{len(error_probabilities)}_coherent.nc", overwrite=True)
        # fig, ax = plot_all_defect_rate(
        #     error_identifier=error_identifier,
        #     included_rounds=qec_rounds,
        # )

    # Concatenate along a new, unnamed dimension
    combined_stepwise: xr.Dataset = xr.concat(list_of_datasets, dim="error_probability")
    # Assign the coordinate values to the new dimension
    combined_stepwise.coords["error_probability"] = error_probabilities
    meas_ctrl.save_dataset_to_netcdf(combined_stepwise, common_path / f"defect_rate_d5_coherent_stacked.nc", overwrite=True)

    list_of_datasets: List[xr.Dataset] = []
    for i, error_probability in enumerate(error_probabilities):
        involved_qubits, gates = meas_ctrl.construct_distance_5_bitflip_circuit_gates(
            error_probability=float(error_probability),
            error_coherence=False,
        )
        error_identifier: ErrorDetectionIdentifier = meas_ctrl.construct_error_identifier(
            cycle_circuit_gates=gates,
            depth=qec_rounds,
            repetitions=repetitions,
        )

        dataset = meas_ctrl.construct_dataset(
            error_identifier=error_identifier,
            noise_file_path=ROOT_DIR / noise_file,
        )
        dataset.attrs["error_probability"] = error_probability
        dataset.attrs["error_coherence"] = False
        list_of_datasets.append(dataset)

        # print(dataset)
        meas_ctrl.save_dataset_to_netcdf(dataset, common_path / f"defect_rate_d5_p{i}-{len(error_probabilities)}_stochastic.nc", overwrite=True)
        # fig, ax = plot_all_defect_rate(
        #     error_identifier=error_identifier,
        #     included_rounds=qec_rounds,
        # )

    # Concatenate along a new, unnamed dimension
    combined_stepwise: xr.Dataset = xr.concat(list_of_datasets, dim="error_probability")
    # Assign the coordinate values to the new dimension
    combined_stepwise.coords["error_probability"] = error_probabilities
    meas_ctrl.save_dataset_to_netcdf(combined_stepwise, common_path / f"defect_rate_d5_stochastic_stacked.nc", overwrite=True)
