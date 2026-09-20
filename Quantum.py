from typing import Any

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def quantum_status() -> dict[str, Any]:
    try:
        simulator = AerSimulator()

        try:
            devices = list(simulator.available_devices())
        except Exception:
            devices = []

        return {
            "status": "ready",
            "backend": "Qiskit AerSimulator",
            "devices": devices,
        }

    except Exception as exc:
        return {
            "status": "error",
            "backend": "Qiskit AerSimulator",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_quantum(
    num_qubits: int = 20,
    shots: int = 1,
) -> dict[str, Any]:

    if not 1 <= num_qubits <= 30:
        raise ValueError("num_qubits must be between 1 and 30.")

    if not 1 <= shots <= 10000:
        raise ValueError("shots must be between 1 and 10000.")

    circuit = QuantumCircuit(
        num_qubits,
        num_qubits,
    )

    for qubit in range(num_qubits):
        circuit.h(qubit)

    for qubit in range(num_qubits - 1):
        circuit.cx(qubit, qubit + 1)

    circuit.measure(
        range(num_qubits),
        range(num_qubits),
    )

    simulator = AerSimulator()

    job = simulator.run(
        circuit,
        shots=shots,
    )

    result = job.result()
    counts = result.get_counts(circuit)

    if not counts:
        raise RuntimeError(
            "Quantum simulation returned no measurement results."
        )

    measurements = []

    for state, count in sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        clean_state = state.replace(" ", "")
        value = int(clean_state, 2)

        measurements.append({
            "binary": clean_state,
            "integer": value,
            "count": count,
            "probability": count / shots,
        })

    most_common = measurements[0]

    max_value = (2 ** num_qubits) - 1

    normalized_value = (
        most_common["integer"] / max_value
        if max_value > 0
        else 0.0
    )

    return {
        "backend": "Qiskit AerSimulator",
        "result_type": "measurement",
        "qubits": num_qubits,
        "shots": shots,

        "binary": most_common["binary"],
        "integer": most_common["integer"],
        "factor": normalized_value,

        "count": most_common["count"],
        "probability": most_common["probability"],

        "counts": counts,
        "measurements": measurements,

        "depth": circuit.depth(),
        "operations": dict(circuit.count_ops()),

        "circuit": {
            "qubits": num_qubits,
            "classical_bits": num_qubits,
            "gates": {
                "h": num_qubits,
                "cx": max(0, num_qubits - 1),
                "measure": num_qubits,
            },
        },

        "meaning": (
            "This is a measured computational-basis state "
            "from the quantum circuit. It is not automatically "
            "the solution to the user's problem."
        ),
    }