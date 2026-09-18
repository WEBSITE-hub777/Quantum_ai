import re
from typing import Any

from queen import ask_queen
from quantum import run_quantum


# ==========================================
# QUANTUM TERMS
# ==========================================

QUANTUM_TERMS = {
    "quantum",
    "qubit",
    "qubits",
    "qiskit",
    "quantum circuit",
    "quantum computing",
    "quantum computer",
    "superposition",
    "entanglement",
    "entangled",
    "bell state",
    "quantum state"
}


# ==========================================
# NORMALIZE TEXT
# ==========================================

def normalize_text(text: str) -> str:

    return re.sub(
        r"\s+",
        " ",
        text.lower()
    ).strip()


# ==========================================
# QUANTUM QUESTION CHECK
# ==========================================

def is_quantum_question(text: str) -> bool:

    text = normalize_text(text)

    for term in sorted(
        QUANTUM_TERMS,
        key=len,
        reverse=True
    ):

        if term in text:
            return True

    return False


# ==========================================
# CLEAN CHAT HISTORY
# ==========================================

def clean_history(
    history: list[dict[str, Any]]
) -> list[dict[str, str]]:

    cleaned = []

    for item in history[-20:]:

        role = item.get("role")
        content = item.get("content")

        if role not in {
            "user",
            "assistant"
        }:
            continue

        if not isinstance(
            content,
            str
        ):
            continue

        if not content.strip():
            continue

        cleaned.append({
            "role": role,
            "content": content[:12000]
        })

    return cleaned


# ==========================================
# MAIN ROUTER
# ==========================================

def process_request(
    user_message: str,
    history: list[dict[str, Any]] | None = None
):

    history = clean_history(
        history or []
    )

    message = user_message.strip()

    if not message:

        return {
            "type": "error",
            "answer": "Please enter a message."
        }


    # ======================================
    # QUANTUM ROUTE
    # ======================================

    if is_quantum_question(message):

        quantum_result = run_quantum(
            num_qubits=20,
            shots=1
        )


        quantum_context = f"""
Qiskit Aer simulation result:

Qubits:
{quantum_result["qubits"]}

Shots:
{quantum_result["shots"]}

Measured state:
{quantum_result["binary"]}

Integer:
{quantum_result["integer"]}

Normalized value:
{quantum_result["factor"]:.6f}

Circuit depth:
{quantum_result["depth"]}

Operations:
{quantum_result["operations"]}
"""


        prompt = f"""
You are Quantum Queen AI.

Answer the user's question clearly.

USER QUESTION:
{message}

QUANTUM ENGINE RESULT:
{quantum_context}

IMPORTANT RULES:

1. The quantum result is computational context.

2. Do NOT claim that 20 qubits means
1,048,576 separate problems were individually
solved.

3. Do NOT claim this simulator is an IBM Quantum
real QPU.

4. Explain what the measured state actually means.

5. If the quantum result is not relevant to the
user's actual question, say so and answer normally.
"""


        answer = ask_queen(
            prompt,
            history
        )


        return {
            "type": "quantum",

            "answer": answer,

            "quantum": quantum_result
        }


    # ======================================
    # NORMAL ROUTE
    # ======================================

    answer = ask_queen(
        message,
        history
    )


    return {
        "type": "normal",
        "answer": answer
    }