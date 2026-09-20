from typing import Any
import math


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _state(value: Any) -> str:
    return str(value).replace(" ", "").strip()


def _probability(count: int, shots: int) -> float:
    if shots <= 0:
        return 0.0
    return max(0.0, min(1.0, count / shots))


def _entropy(probabilities: list[float]) -> float:
    return -sum(
        p * math.log2(p)
        for p in probabilities
        if p > 0
    )


def _normalize_measurements(
    measurements: Any,
    counts: Any,
    shots: int,
) -> list[dict[str, Any]]:

    normalized = []

    if isinstance(measurements, list):
        for item in measurements:
            if not isinstance(item, dict):
                continue

            binary = _state(
                item.get("binary", "")
            )

            if not binary:
                continue

            count = _int(
                item.get("count", 0)
            )

            probability = _float(
                item.get("probability", 0.0)
            )

            if shots > 0:
                probability = _probability(
                    count,
                    shots,
                )

            normalized.append({
                "binary": binary,
                "integer": _int(
                    item.get("integer", 0)
                ),
                "count": count,
                "probability": probability,
            })

    elif isinstance(counts, dict):

        for raw_state, raw_count in counts.items():

            binary = _state(raw_state)

            if not binary:
                continue

            count = _int(raw_count)

            normalized.append({
                "binary": binary,
                "integer": int(binary, 2),
                "count": count,
                "probability": _probability(
                    count,
                    shots,
                ),
            })

    normalized.sort(
        key=lambda item: (
            item["probability"],
            item["count"],
        ),
        reverse=True,
    )

    return normalized


def _build_measurement_analysis(
    measurements: list[dict[str, Any]],
    shots: int,
) -> dict[str, Any]:

    probabilities = [
        item["probability"]
        for item in measurements
    ]

    total_probability = sum(
        probabilities
    )

    dominant = (
        measurements[0]
        if measurements
        else None
    )

    return {
        "unique_states": len(measurements),
        "total_probability": total_probability,
        "dominant_state": (
            dominant["binary"]
            if dominant
            else None
        ),
        "dominant_integer": (
            dominant["integer"]
            if dominant
            else None
        ),
        "dominant_probability": (
            dominant["probability"]
            if dominant
            else 0.0
        ),
        "dominant_count": (
            dominant["count"]
            if dominant
            else 0
        ),
        "entropy_bits": _entropy(
            probabilities
        ),
        "shots": shots,
    }


def _build_summary(
    qubits: int,
    shots: int,
    dominant: dict[str, Any] | None,
    entropy_bits: float,
    unique_states: int,
) -> str:

    if not dominant:
        return (
            "The quantum engine completed, "
            "but no valid measurement state "
            "was available."
        )

    return (
        f"The quantum circuit used {qubits} qubits "
        f"and {shots} measurement shot(s). "
        f"It produced {unique_states} unique "
        f"observed state(s). "
        f"The most frequently observed state was "
        f"{dominant['binary']}, corresponding to "
        f"decimal value {dominant['integer']}, "
        f"with an observed probability of "
        f"{dominant['probability']:.2%}. "
        f"The measurement distribution has an "
        f"entropy of {entropy_bits:.4f} bits. "
        f"This result is a measurement outcome, "
        f"not automatically a solution to the "
        f"user's problem."
    )


def _build_ai_context(
    qubits: int,
    shots: int,
    analysis: dict[str, Any],
    normalized_value: float,
    circuit_depth: int,
    operations: dict[str, Any],
) -> str:

    dominant = analysis.get(
        "dominant_state"
    )

    dominant_integer = analysis.get(
        "dominant_integer"
    )

    dominant_probability = analysis.get(
        "dominant_probability",
        0.0,
    )

    return (
        "Quantum computation summary:\n"
        f"Qubits: {qubits}\n"
        f"Shots: {shots}\n"
        f"Unique measured states: "
        f"{analysis.get('unique_states', 0)}\n"
        f"Dominant measured state: "
        f"{dominant}\n"
        f"Dominant decimal value: "
        f"{dominant_integer}\n"
        f"Dominant probability: "
        f"{dominant_probability:.2%}\n"
        f"Normalized value: "
        f"{normalized_value:.6f}\n"
        f"Measurement entropy: "
        f"{analysis.get('entropy_bits', 0.0):.4f} bits\n"
        f"Circuit depth: {circuit_depth}\n"
        f"Operations: {operations}\n\n"
        "Interpretation rule: the measured state is "
        "quantum-circuit output. It must not be "
        "described as the user's problem solution "
        "unless the circuit was explicitly designed "
        "to solve that problem."
    )


def interpret_quantum_result(
    result: dict[str, Any],
) -> dict[str, Any]:

    if not isinstance(result, dict):
        raise TypeError(
            "Quantum result must be a dictionary."
        )

    qubits = max(
        0,
        _int(result.get("qubits", 0)),
    )

    shots = max(
        0,
        _int(result.get("shots", 0)),
    )

    normalized_value = max(
        0.0,
        min(
            1.0,
            _float(
                result.get("factor", 0.0)
            ),
        ),
    )

    circuit_depth = max(
        0,
        _int(result.get("depth", 0)),
    )

    operations = result.get(
        "operations",
        {},
    )

    if not isinstance(operations, dict):
        operations = {}

    measurements = _normalize_measurements(
        result.get("measurements"),
        result.get("counts"),
        shots,
    )

    analysis = _build_measurement_analysis(
        measurements,
        shots,
    )

    dominant = (
        measurements[0]
        if measurements
        else None
    )

    summary = _build_summary(
        qubits=qubits,
        shots=shots,
        dominant=dominant,
        entropy_bits=analysis["entropy_bits"],
        unique_states=analysis["unique_states"],
    )

    ai_context = _build_ai_context(
        qubits=qubits,
        shots=shots,
        analysis=analysis,
        normalized_value=normalized_value,
        circuit_depth=circuit_depth,
        operations=operations,
    )

    return {
        "result_type": "quantum_measurement",
        "status": "interpreted",
        "has_result": bool(measurements),
        "summary": summary,
        "ai_context": ai_context,
        "qubits": qubits,
        "shots": shots,
        "normalized_value": normalized_value,
        "circuit_depth": circuit_depth,
        "operations": operations,
        "analysis": analysis,
        "measurements": measurements,
        "solution_status": {
            "is_solution": False,
            "reason": (
                "A measurement result alone does not "
                "establish that the user's problem "
                "has been solved."
            ),
        },
    }