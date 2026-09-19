from typing import Any
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def quantum_status() -> dict[str, Any]:
    try:
        simulator = AerSimulator()
        try:
            devices = simulator.available_devices()
        except Exception:
            devices = []
        return {
            "status": "ready",
            "backend": "Qiskit AerSimulator",
            "devices": list(devices)
        }
    except Exception as exc:
        return {
            "status": "error",
            "backend": "Qiskit AerSimulator",
            "error": f"{type(exc).__name__}: {exc}"
        }

def run_quantum(num_qubits: int = 20, shots: int = 1):
    if not 1 <= num_qubits <= 30:
        raise ValueError("num_qubits must be between 1 and 30.")
    if not 1 <= shots <= 10000:
        raise ValueError("shots must be between 1 and 10000.")

    circuit = QuantumCircuit(num_qubits, num_qubits)

    for qubit in range(num_qubits):
        circuit.h(qubit)

    for qubit in range(num_qubits - 1):
        circuit.cx(qubit, qubit + 1)

    circuit.measure(range(num_qubits), range(num_qubits))

    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)
    binary = next(iter(counts))
    integer_value = int(binary, 2)
    max_value = (2 ** num_qubits) - 1
    factor = integer_value / max_value if max_value > 0 else 0.0
    operations = dict(circuit.count_ops())

    return {
        "backend": "Qiskit AerSimulator",
        "qubits": num_qubits,
        "shots": shots,
        "binary": binary,
        "integer": integer_value,
        "factor": factor,
        "depth": circuit.depth(),
        "operations": operations,
        "counts": counts
    }
