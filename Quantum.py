from typing import Any

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


MIN_QUBITS = 1
MAX_QUBITS = 30

MIN_SHOTS = 1
MAX_SHOTS = 10000


def quantum_status() -> dict[str, Any]:

    try:
        simulator = AerSimulator()

        try:
            devices = list(
                simulator.available_devices()
            )
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
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


def validate_parameters(
    num_qubits: int,
    shots: int,
) -> None:

    if not (
        MIN_QUBITS
        <= num_qubits
        <= MAX_QUBITS
    ):
        raise ValueError(
            f"Qubits must be between "
            f"{MIN_QUBITS} and "
            f"{MAX_QUBITS}."
        )

    if not (
        MIN_SHOTS
        <= shots
        <= MAX_SHOTS
    ):
        raise ValueError(
            f"Shots must be between "
            f"{MIN_SHOTS} and "
            f"{MAX_SHOTS}."
        )


def build_demo_circuit(
    num_qubits: int,
) -> QuantumCircuit:

    circuit = QuantumCircuit(
        num_qubits,
        num_qubits,
    )

    circuit.h(
        range(num_qubits)
    )

    for index in range(
        num_qubits - 1
    ):
        circuit.cx(
            index,
            index + 1,
        )

    circuit.measure(
        range(num_qubits),
        range(num_qubits),
    )

    return circuit


def normalize_state(
    state: str,
) -> str:

    return (
        str(state)
        .replace(" ", "")
        .strip()
    )


def run_quantum(
    num_qubits: int = 20,
    shots: int = 1024,
) -> dict[str, Any]:

    validate_parameters(
        num_qubits,
        shots,
    )

    circuit = build_demo_circuit(
        num_qubits
    )

    simulator = AerSimulator()

    job = simulator.run(
        circuit,
        shots=shots,
    )

    result = job.result()

    counts = result.get_counts(
        circuit
    )

    if not counts:
        raise RuntimeError(
            "Quantum simulation returned no results."
        )

    measurements = []

    for raw_state, raw_count in sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        state = normalize_state(
            raw_state
        )

        count = int(raw_count)

        integer_value = int(
            state,
            2,
        )

        probability = (
            count / shots
        )

        measurements.append(
            {
                "binary": state,
                "integer": integer_value,
                "count": count,
                "probability": probability,
            }
        )

    dominant = measurements[0]

    max_value = (
        2 ** num_qubits
    ) - 1

    normalized_value = (
        dominant["integer"]
        / max_value
        if max_value
        else 0.0
    )

    return {
        "backend": "Qiskit AerSimulator",
        "result_type": "measurement",
        "qubits": num_qubits,
        "shots": shots,
        "binary": dominant["binary"],
        "integer": dominant["integer"],
        "probability": dominant["probability"],
        "factor": normalized_value,
        "count": dominant["count"],
        "counts": counts,
        "measurements": measurements,
        "depth": circuit.depth(),
        "operations": dict(
            circuit.count_ops()
        ),
        "circuit": {
            "qubits": num_qubits,
            "classical_bits": num_qubits,
            "gates": {
                "h": num_qubits,
                "cx": max(
                    0,
                    num_qubits - 1,
                ),
                "measure": num_qubits,
            },
        },
        "meaning": (
            "This is a measurement outcome "
            "from the executed quantum circuit. "
            "It is not automatically a solution "
            "to the user's problem."
        ),
    }