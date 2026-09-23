import re
from enum import Enum
from typing import Any

from ai_tools import (
    generate_image,
    solve_math,
    understand_image,
)
from queen import ask_queen
from quantum import run_quantum


MAX_HISTORY = 30
MAX_MESSAGE_LENGTH = 20000

QUANTUM_QUBITS = 20
QUANTUM_SHOTS = 1024


class Intent(str, Enum):
    NORMAL = "normal"
    MATH = "math"
    QUANTUM = "quantum"
    VISION = "vision"
    IMAGE_GENERATION = "image_generation"


QUANTUM_PATTERNS = (
    "quantum",
    "qubit",
    "qubits",
    "qiskit",
    "quantum circuit",
    "quantum computing",
    "quantum computer",
    "quantum gate",
    "quantum gates",
    "superposition",
    "entanglement",
    "bell state",
    "quantum algorithm",
)


IMAGE_GENERATION_PATTERNS = (
    "generate image",
    "create image",
    "make an image",
    "generate a picture",
    "create a picture",
    "make a picture",
    "image bana",
    "photo bana",
    "picture bana",
    "इमेज बनाओ",
    "फोटो बनाओ",
    "चित्र बनाओ",
)


MATH_PATTERNS = (
    "solve equation",
    "solve this equation",
    "calculate",
    "mathematics",
    "mathematical",
    "math problem",
    "math question",
    "algebra",
    "geometry",
    "trigonometry",
    "percentage",
    "fraction",
    "quadratic",
    "derivative",
    "integral",
    "probability",
    "गणित",
    "मैथ",
    "गणित का सवाल",
    "सवाल हल",
    "समीकरण",
    "प्रतिशत",
)


def normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()


def contains_pattern(
    text: str,
    patterns: tuple[str, ...],
) -> bool:

    normalized = normalize(text)

    return any(
        pattern in normalized
        for pattern in sorted(
            patterns,
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

        if role not in {
            "user",
            "assistant",
        }:
            continue

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        cleaned.append(
            {
                "role": role,
                "content": content[
                    :MAX_MESSAGE_LENGTH
                ],
            }
        )

    return cleaned


def looks_like_math_structure(
    message: str,
) -> bool:

    text = normalize(message)

    equation = bool(
        re.search(
            r"\b[a-z]\s*[\+\-\*/^]\s*[\dx0-9]",
            text,
        )
    )

    equality = "=" in text

    arithmetic = bool(
        re.search(
            r"\d+\s*[\+\-\*/]\s*\d+",
            text,
        )
    )

    fraction = "/" in text and bool(
        re.search(r"\d+\s*/\s*\d+", text)
    )

    return any(
        (
            equation,
            equality,
            arithmetic,
            fraction,
        )
    )


def classify_request(
    message: str,
    has_image: bool = False,
) -> Intent:

    if has_image:
        return Intent.VISION

    if contains_pattern(
        message,
        IMAGE_GENERATION_PATTERNS,
    ):
        return Intent.IMAGE_GENERATION

    if contains_pattern(
        message,
        QUANTUM_PATTERNS,
    ):
        return Intent.QUANTUM

    if (
        contains_pattern(
            message,
            MATH_PATTERNS,
        )
        or looks_like_math_structure(message)
    ):
        return Intent.MATH

    return Intent.NORMAL


def build_quantum_prompt(
    user_message: str,
    result: dict[str, Any],
) -> str:

    return f"""
You are Quantum Queen AI.

The user explicitly asked about quantum computing.

User question:
{user_message}

Quantum computation:
{result}

Rules:
1. Explain the actual quantum computation.
2. A measurement result is not automatically a solution.
3. Never inject the measured binary state into an unrelated mathematics answer.
4. Never claim that a generic circuit solved an arbitrary problem.
5. Never claim Qiskit Aer is an IBM Quantum QPU.
6. Explain raw binary/decimal values only when they are relevant.
7. Answer the user's actual question first.
8. Do not dump internal metadata unnecessarily.
""".strip()


def process_math(
    message: str,
    history: list[dict[str, str]],
) -> dict[str, Any]:

    result = solve_math(message)

    if result.get("solved"):
        return {
            "type": "math",
            "status": "completed",
            "answer": result["answer"],
            "math": result,
        }

    answer = ask_queen(
        message,
        history=history,
        system_context=result.get(
            "context",
            "",
        ),
    )

    return {
        "type": "math",
        "status": "completed",
        "answer": answer,
        "math": result,
    }


def process_quantum(
    message: str,
    history: list[dict[str, str]],
) -> dict[str, Any]:

    result = run_quantum(
        num_qubits=QUANTUM_QUBITS,
        shots=QUANTUM_SHOTS,
    )

    prompt = build_quantum_prompt(
        message,
        result,
    )

    answer = ask_queen(
        prompt,
        history=history,
    )

    return {
        "type": "quantum",
        "status": "completed",
        "answer": answer,
        "quantum": result,
    }


def process_vision(
    message: str,
    image_data: str,
    image_mime: str | None,
) -> dict[str, Any]:

    result = understand_image(
        question=message,
        image_data=image_data,
        image_mime=image_mime,
    )

    return {
        "type": "vision",
        "status": result.get(
            "status",
            "completed",
        ),
        "answer": result.get(
            "answer",
            "",
        ),
        "vision": result,
    }


def process_image_generation(
    message: str,
) -> dict[str, Any]:

    result = generate_image(message)

    return {
        "type": "image",
        "status": result.get(
            "status",
            "completed",
        ),
        "answer": result.get(
            "answer",
            "",
        ),
        "image": result,
    }


def process_request(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    image_data: str | None = None,
    image_mime: str | None = None,
) -> dict[str, Any]:

    message = user_message.strip()

    if not message:
        raise ValueError(
            "Please enter a message."
        )

    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            "Your message is too long."
        )

    cleaned_history = clean_history(
        history
    )

    intent = classify_request(
        message,
        has_image=bool(image_data),
    )

    if intent == Intent.MATH:
        return process_math(
            message,
            cleaned_history,
        )

    if intent == Intent.QUANTUM:
        return process_quantum(
            message,
            cleaned_history,
        )

    if intent == Intent.VISION:
        return process_vision(
            message,
            image_data,
            image_mime,
        )

    if intent == Intent.IMAGE_GENERATION:
        return process_image_generation(
            message
        )

    answer = ask_queen(
        message,
        history=cleaned_history,
    )

    return {
        "type": "normal",
        "status": "completed",
        "answer": answer,
    }
