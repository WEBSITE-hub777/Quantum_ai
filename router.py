import re
from typing import Any

from queen import ask_queen
from quantum import run_quantum
from quantum_interpreter import interpret_quantum_result


APP_NAME = "Quantum Queen AI"
MAX_HISTORY = 20
MAX_MESSAGE_LENGTH = 12000
QUANTUM_QUBITS = 20
QUANTUM_SHOTS = 1

QUANTUM_TERMS = {
    "quantum",
    "qubit",
    "qubits",
    "qiskit",
    "quantum circuit",
    "quantum computing",
    "quantum computer",
    "quantum mechanics",
    "quantum state",
    "quantum states",
    "superposition",
    "entanglement",
    "entangled",
    "bell state",
    "bell states",
    "quantum gate",
    "quantum gates",
    "quantum algorithm",
    "quantum algorithms",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_quantum_question(text: str) -> bool:
    normalized = normalize_text(text)

    return any(
        term in normalized
        for term in sorted(
            QUANTUM_TERMS,
            key=len,
            reverse=True,
        )
    )


def clean_history(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    if not history:
        return []

    cleaned = []

    for item in history[-MAX_HISTORY:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        cleaned.append({
            "role": role,
            "content": content[:MAX_MESSAGE_LENGTH],
        })

    return cleaned


def validate_message(message: str) -> str:
    message = message.strip()

    if not message:
        raise ValueError("Please enter a message.")

    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            "Your message is too long. Please send a shorter message."
        )

    return message


def build_quantum_prompt(
    user_message: str,
    interpretation: dict[str, Any],
) -> str:
    return f"""
You are {APP_NAME}.

Answer the user's question naturally, accurately, and clearly.

USER QUESTION:
{user_message}

QUANTUM ENGINE INTERPRETATION:
{interpretation.get("summary", "No interpretation available.")}

QUANTUM DATA:
Result type: {interpretation.get("result_type", "unknown")}
Qubits: {interpretation.get("qubits", 0)}
Shots: {interpretation.get("shots", 0)}
Measured state: {interpretation.get("measured_state", "unknown")}
Decimal value: {interpretation.get("decimal_value", 0)}
Normalized value: {interpretation.get("normalized_value", 0.0):.6f}
Probability: {interpretation.get("probability", 0.0):.2%}
Circuit depth: {interpretation.get("circuit_depth", 0)}
Operations: {interpretation.get("operations", {})}

RESPONSE RULES:
- Explain the quantum result in natural language.
- Do not simply repeat the binary number.
- Do not present the measured number as the solution unless the quantum computation actually solves the user's problem.
- Do not claim that 20 qubits means 1,048,576 separate problems were individually solved.
- Do not claim that Qiskit Aer is an IBM Quantum QPU.
- Distinguish between a measurement result and an actual computed solution.
- If the quantum result is not relevant to the user's question, say so clearly.
- Use the quantum information as computational context.
- Prefer a useful explanation over a raw data dump.
- Answer in the same language as the user whenever practical.
""".strip()


def process_quantum_request(
    message: str,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        quantum_result = run_quantum(
            num_qubits=QUANTUM_QUBITS,
            shots=QUANTUM_SHOTS,
        )

        interpretation = interpret_quantum_result(
            quantum_result,
        )

        prompt = build_quantum_prompt(
            message,
            interpretation,
        )

        answer = ask_queen(
            prompt,
            history,
        )

        return {
            "type": "quantum",
            "status": "completed",
            "answer": answer,
            "quantum": {
                "backend": quantum_result.get(
                    "backend",
                    "Qiskit AerSimulator",
                ),
                "result_type": interpretation.get(
                    "result_type",
                    "quantum_measurement",
                ),
                "qubits": interpretation.get(
                    "qubits",
                    QUANTUM_QUBITS,
                ),
                "shots": interpretation.get(
                    "shots",
                    QUANTUM_SHOTS,
                ),
                "interpretation": interpretation,
            },
        }

    except Exception as exc:
        return {
            "type": "quantum",
            "status": "error",
            "answer": (
                "Quantum processing could not be completed "
                "right now. Please try again."
            ),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


def process_normal_request(
    message: str,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        answer = ask_queen(
            message,
            history,
        )

        return {
            "type": "normal",
            "status": "completed",
            "answer": answer,
        }

    except Exception as exc:
        return {
            "type": "normal",
            "status": "error",
            "answer": (
                "Queen AI could not process your request "
                "right now. Please try again."
            ),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


def process_request(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:

    try:
        message = validate_message(
            user_message,
        )
    except ValueError as exc:
        return {
            "type": "error",
            "status": "invalid_request",
            "answer": str(exc),
        }

    cleaned_history = clean_history(
        history,
    )

    if is_quantum_question(message):
        return process_quantum_request(
            message,
            cleaned_history,
        )

    return process_normal_request(
        message,
        cleaned_history,
    )