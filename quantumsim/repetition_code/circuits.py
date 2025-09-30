t_sqg: float = 20
t_tqg: float = 60


def distance_3_bitflip_gates(model):
    gates = []
    t_ref: float = 0
    # Start of Parities
    gates.extend(
        [
            model.idle_sq_gate("D4", t_start=t_ref),
            model.idle_sq_gate("Z1", t_start=t_ref),
            model.idle_sq_gate("D1", t_start=t_ref),
            model.rotate_y_90("X1", t_start=t_ref),
            model.idle_sq_gate("D2", t_start=t_ref),
        ]
    )
    t_ref += t_sqg
    # X1-D2
    gates.extend(
        [
            model.idle_tq_gate("D4", t_start=t_ref),
            model.idle_tq_gate("Z1", t_start=t_ref),
            model.idle_tq_gate("D1", t_start=t_ref),
            model.cphase("D2", "X1", t_start=t_ref),
            model.zz_coupling_2q("D1", "X1", t_start=t_ref),
        ]
    )
    t_ref += t_tqg
    # X1-D1
    gates.extend(
        [
            model.idle_tq_gate("D4", t_start=t_ref),
            model.idle_tq_gate("Z1", t_start=t_ref),
            model.idle_tq_gate("D2", t_start=t_ref),
            model.cphase("D1", "X1", t_start=t_ref),
            model.zz_coupling_2q("D2", "X1", t_start=t_ref),
        ]
    )
    t_ref += t_tqg
    # Middle of Parities
    gates.extend(
        [
            model.idle_sq_gate("D4", t_start=t_ref),
            model.rotate_y_90("Z1", t_start=t_ref),
            model.idle_sq_gate("D1", t_start=t_ref),
            model.rotate_y_m90("X1", t_start=t_ref),
            model.idle_sq_gate("D2", t_start=t_ref),
        ]
    )
    t_ref += t_sqg
    # Z1-D1
    gates.extend(
        [
            model.idle_tq_gate("D4", t_start=t_ref),
            model.idle_tq_gate("X1", t_start=t_ref),
            model.idle_tq_gate("D2", t_start=t_ref),
            model.cphase("D1", "Z1", t_start=t_ref),
            model.zz_coupling_2q("Z1", "D4", t_start=t_ref),
            model.zz_coupling_2q("D2", "Z1", t_start=t_ref),
        ]
    )
    t_ref += t_tqg
    # Z1-D4
    gates.extend(
        [
            model.idle_tq_gate("D1", t_start=t_ref),
            model.idle_tq_gate("X1", t_start=t_ref),
            model.idle_tq_gate("D2", t_start=t_ref),
            model.cphase("Z1", "D4", t_start=t_ref),
            model.zz_coupling_2q("D1", "Z1", t_start=t_ref),
            model.zz_coupling_2q("D2", "Z1", t_start=t_ref),
        ]
    )
    t_ref += t_tqg
    # End of Parities
    gates.extend(
        [
            model.idle_sq_gate("D4", t_start=t_ref),
            model.rotate_y_m90("Z1", t_start=t_ref),
            model.idle_sq_gate("D1", t_start=t_ref),
            model.idle_sq_gate("X1", t_start=t_ref),
            model.idle_sq_gate("D2", t_start=t_ref),
        ]
    )
    t_ref += t_sqg
    # Refocusing operations
    gates.extend(
        [
            model.rotate_x_180("D4", t_start=t_ref),
            model.idle_sq_gate("Z1", t_start=t_ref),
            model.rotate_x_180("D1", t_start=t_ref),
            model.idle_sq_gate("X1", t_start=t_ref),
            model.rotate_x_180("D2", t_start=t_ref),
        ]
    )

    return gates


def distance_5_bitflip_gates(model):
    # picture from Sean's WhatsApp conversation
    gates = []
    t_ref: float = 0
    gates.extend(
        [
            model.idle_sq_gate("D7", t_start=t_ref),
            model.idle_sq_gate("D4", t_start=t_ref),
            model.idle_sq_gate("D1", t_start=t_ref),
            model.idle_sq_gate("D2", t_start=t_ref),
            model.idle_sq_gate("D3", t_start=t_ref),
            model.rotate_y_90("Z3", t_start=t_ref),
            model.idle_sq_gate("Z1", t_start=t_ref),
            model.rotate_y_90("X1", t_start=t_ref),
            model.idle_sq_gate("X2", t_start=t_ref),
        ]
    )
    t_ref += t_sqg

    gates.extend(
        [
            model.idle_tq_gate("D7", t_start=t_ref),
            model.cphase("Z3", "D4", t_start=t_ref),
            model.idle_tq_gate("D1", t_start=t_ref),
            model.cphase("D2", "X1", t_start=t_ref),
            model.idle_tq_gate("D3", t_start=t_ref),
            model.idle_tq_gate("Z1", t_start=t_ref),
            model.idle_tq_gate("X2", t_start=t_ref),
            model.zz_coupling_2q("Z3", "D7", t_start=t_ref),
            model.zz_coupling_2q("X1", "D1", t_start=t_ref),
        ]
    )
    t_ref += t_tqg

    gates.extend(
        [
            model.cphase("D7", "Z3", t_start=t_ref),
            model.idle_tq_gate("D4", t_start=t_ref),
            model.cphase("D1", "X1", t_start=t_ref),
            model.idle_tq_gate("D2", t_start=t_ref),
            model.idle_tq_gate("D3", t_start=t_ref),
            model.idle_tq_gate("Z1", t_start=t_ref),
            model.idle_tq_gate("X2", t_start=t_ref),
            model.zz_coupling_2q("Z3", "D4", t_start=t_ref),
            model.zz_coupling_2q("X1", "D2", t_start=t_ref),
        ]
    )
    t_ref += t_tqg

    gates.extend(
        [
            model.idle_sq_gate("D7", t_start=t_ref),
            model.idle_sq_gate("D4", t_start=t_ref),
            model.idle_sq_gate("D1", t_start=t_ref),
            model.idle_sq_gate("D2", t_start=t_ref),
            model.idle_sq_gate("D3", t_start=t_ref),
            model.rotate_y_m90("Z3", t_start=t_ref),
            model.rotate_y_90("Z1", t_start=t_ref),
            model.rotate_y_m90("X1", t_start=t_ref),
            model.rotate_y_90("X2", t_start=t_ref),
        ]
    )
    t_ref += t_sqg

    gates.extend(
        [
            model.idle_tq_gate("D7", t_start=t_ref),
            model.cphase("Z1", "D4", t_start=t_ref),
            model.idle_tq_gate("D1", t_start=t_ref),
            model.cphase("D2", "X2", t_start=t_ref),
            model.idle_tq_gate("D3", t_start=t_ref),
            model.idle_tq_gate("Z3", t_start=t_ref),
            model.idle_tq_gate("X1", t_start=t_ref),
            model.zz_coupling_2q("Z1", "D1", t_start=t_ref),
            model.zz_coupling_2q("X2", "D3", t_start=t_ref),
        ]
    )
    t_ref += t_tqg

    gates.extend(
        [
            model.idle_tq_gate("D7", t_start=t_ref),
            model.idle_tq_gate("D4", t_start=t_ref),
            model.cphase("D1", "Z1", t_start=t_ref),
            model.idle_tq_gate("D2", t_start=t_ref),
            model.cphase("D3", "X2", t_start=t_ref),
            model.idle_tq_gate("Z3", t_start=t_ref),
            model.idle_tq_gate("X1", t_start=t_ref),
            model.zz_coupling_2q("Z1", "D4", t_start=t_ref),
            model.zz_coupling_2q("X2", "D2", t_start=t_ref),
        ]
    )
    t_ref += t_tqg

    gates.extend(
        [
            model.idle_sq_gate("D7", t_start=t_ref),
            model.idle_sq_gate("D4", t_start=t_ref),
            model.idle_sq_gate("D1", t_start=t_ref),
            model.idle_sq_gate("D2", t_start=t_ref),
            model.idle_sq_gate("D3", t_start=t_ref),
            model.idle_sq_gate("Z3", t_start=t_ref),
            model.rotate_y_m90("Z1", t_start=t_ref),
            model.idle_sq_gate("X1", t_start=t_ref),
            model.rotate_y_m90("X2", t_start=t_ref),
        ]
    )
    t_ref += t_sqg

    gates.extend(
        [
            model.rotate_x_180("D7", t_start=t_ref),
            model.rotate_x_180("D4", t_start=t_ref),
            model.rotate_x_180("D1", t_start=t_ref),
            model.rotate_x_180("D2", t_start=t_ref),
            model.rotate_x_180("D3", t_start=t_ref),
            model.idle_sq_gate("Z3", t_start=t_ref),
            model.idle_sq_gate("Z1", t_start=t_ref),
            model.idle_sq_gate("X1", t_start=t_ref),
            model.idle_sq_gate("X2", t_start=t_ref),
        ]
    )

    return gates


def distance_3_prep_initial_state_gates(model):
    return [
        model.idle_sq_gate("D4", t_start=0),
        model.idle_sq_gate("Z1", t_start=0),
        model.rotate_x_180("D1", t_start=0),
        model.idle_sq_gate("X1", t_start=0),
        model.idle_sq_gate("D2", t_start=0),
    ]


def distance_5_prep_initial_state_gates(model):
    qubits = ["D7", "Z3", "D4", "Z1", "D1", "X1", "D2", "X2", "D3"]
    return [model.idle_sq_gate(qubit, t_start=0) for qubit in qubits]
