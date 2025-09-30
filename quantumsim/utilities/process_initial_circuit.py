from typing import List
from enum import Enum, unique, auto
from quantumsim.circuits import Circuit, FinalizedCircuit
from quantumsim.models.model_dclab import DiCarloLabModel
from qce_circuit.language.intrf_declarative_circuit import (
    InitialStateEnum,
    InitialStateContainer,
)
from qce_circuit.utilities.singleton_base import SingletonABCMeta
from qce_circuit.connectivity.intrf_channel_identifier import (
    QubitIDObj,
    EdgeIDObj,
)
from qce_circuit.connectivity.intrf_connectivity_gate_sequence import (
    GateSequenceLayer,
    Operation,
)
from qce_circuit.connectivity.intrf_connectivity_surface_code import (
    StabilizerType,
    ISurfaceCodeLayer,
)
from qce_circuit.connectivity.connectivity_surface_code import ParityGroup
from qce_circuit.connectivity.generic_gate_sequence import (
    IGenericSurfaceCodeLayer,
    GenericSurfaceCode,
)


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


def construct_initial_circuit(initial_state_type: InitialStateType, ordered_qubits: List[str], code_layout: ISurfaceCodeLayer, noise_model: DiCarloLabModel) -> FinalizedCircuit:
    gates = []
    t_ref: float = 0
    # All zero (Even) initial state
    if initial_state_type in [InitialStateType.ALL_ZERO, InitialStateType.EVEN]:
        for qubit in ordered_qubits:
            gates.append(noise_model.idle_sq_gate(qubit, t_start=t_ref))
        return noise_model.finalize(Circuit(
            qubits=ordered_qubits,
            gates=gates,
        ))
    # All ones initial state
    if initial_state_type == InitialStateType.ALL_ONES:
        for qubit in ordered_qubits:
            if QubitIDObj(qubit) in code_layout.data_qubit_ids:
                gates.append(noise_model.rotate_x_180(qubit, t_start=t_ref))
            else:
                gates.append(noise_model.idle_sq_gate(qubit, t_start=t_ref))

        return noise_model.finalize(Circuit(
            qubits=ordered_qubits,
            gates=gates,
        ))
    # Odd initial state
    if initial_state_type == InitialStateType.ODD:
        data_qubit_count: int = 0
        for qubit in ordered_qubits:
            if QubitIDObj(qubit) in code_layout.data_qubit_ids:
                if data_qubit_count % 2 == 0:
                    gates.append(noise_model.idle_sq_gate(qubit, t_start=t_ref))
                else:
                    gates.append(noise_model.rotate_x_180(qubit, t_start=t_ref))
                data_qubit_count += 1
            else:
                gates.append(noise_model.idle_sq_gate(qubit, t_start=t_ref))
        return noise_model.finalize(Circuit(
            qubits=ordered_qubits,
            gates=gates,
        ))
    raise NotImplementedError(f"Initial state type ({initial_state_type.name}) not yet supported.")


def construct_initial_state(initial_state_type: InitialStateType, ordered_qubits: List[str], code_layout: ISurfaceCodeLayer) -> InitialStateContainer:
    data_qubit_names: List[str] = [
        qubit_name
        for qubit_name in ordered_qubits
        if QubitIDObj(qubit_name) in code_layout.data_qubit_ids
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
