from typing import Any

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ==========================================
# QUANTUM STATUS
# ==========================================

def quantum_status() -> dict[str, Any]:

    try:

        simulator = AerSimulator()

        try:

            devices = (
                simulator
                .available_devices()
            )

        except Exception:

            devices = []


        return {

            "status": "ready",

            "backend": (
                "Qiskit AerSimulator"
            ),

            "devices": list(devices)

        }


    except Exception as exc:

        return {

            "status": "error",

            "backend": (
                "Qiskit AerSimulator"
            ),

            "error": (
                f"{type(exc).__name__}: {exc}"
            )

        }


# ==========================================
# RUN QUANTUM
# ==========================================

def run_quantum(
    num_qubits: int = 20,
    shots: int = 1
):

    # --------------------------------------
    # Validation
    # --------------------------------------

    if not 1 <= num_qubits <= 30:

        raise ValueError(
            "num_qubits must be between 1 and 30."
        )


    if not 1 <= shots <= 10000:

        raise ValueError(
            "shots must be between 1 and 10000."
        )


    # --------------------------------------
    # CREATE CIRCUIT
    # --------------------------------------

    circuit = QuantumCircuit(
        num_qubits,
        num_qubits
    )


    # --------------------------------------
    # SUPERPOSITION
    # --------------------------------------

    for qubit in range(num_qubits):

        circuit.h(qubit)


    # --------------------------------------
    # ENTANGLEMENT
    # --------------------------------------

    for qubit in range(
        num_qubits - 1
    ):

        circuit.cx(
            qubit,
            qubit + 1
        )


    # --------------------------------------
    # MEASUREMENT
    # --------------------------------------

    circuit.measure(
        range(num_qubits),
        range(num_qubits)
    )


    # --------------------------------------
    # AER SIMULATOR
    # --------------------------------------

    simulator = AerSimulator()


    # --------------------------------------
    # RUN
    # --------------------------------------

    job = simulator.run(
        circuit,
        shots=shots
    )


    result = job.result()


    counts = (
        result.get_counts(
            circuit
        )
    )


    # --------------------------------------
    # GET MEASURED STATE
    # --------------------------------------

    binary = next(
        iter(counts)
    )


    # --------------------------------------
    # BINARY → INTEGER
    # --------------------------------------

    integer_value = int(
        binary,
        2
    )


    # --------------------------------------
    # NORMALIZE
    # --------------------------------------

    max_value = (
        2 ** num_qubits
    ) - 1


    factor = (
        integer_value / max_value
        if max_value > 0
        else 0.0
    )


    # --------------------------------------
    # CIRCUIT INFO
    # --------------------------------------

    operations = dict(
        circuit.count_ops()
    )


    return {

        "backend":
            "Qiskit AerSimulator",

        "qubits":
            num_qubits,

        "shots":
            shots,

        "binary":
            binary,

        "integer":
            integer_value,

        "factor":
            factor,

        "depth":
            circuit.depth(),

        "operations":
            operations,

        "counts":
            counts
    }