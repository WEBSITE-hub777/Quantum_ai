import re
from enum import Enum
from typing import Any

from ai_tools import edit_image, generate_image, solve_math, understand_image
from queen import ask_queen
from quantum import run_quantum

MAX_HISTORY = 30
MAX_MESSAGE_LENGTH = 20000
QUANTUM_QUBITS = 20
QUANTUM_SHOTS = 1


class Intent(str, Enum):
    NORMAL = "normal"
    MATH = "math"
    QUANTUM = "quantum"
    VISION = "vision"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"


QUANTUM_PATTERNS = (
    "quantum", "qubit", "qubits", "qiskit", "quantum circuit",
    "quantum computing", "quantum computer", "quantum gate",
    "quantum gates", "superposition", "entanglement", "bell state",
    "quantum algorithm"
)

IMAGE_GENERATION_PATTERNS = (
    "generate image", "create image", "make an image", "draw an image",
    "draw a", "generate a picture", "create a picture", "make a picture",
    "image bana", "photo bana", "picture bana", "banao photo",
    "इमेज बनाओ", "फोटो बनाओ", "चित्र बनाओ", "draw", "paint"
)

IMAGE_EDIT_PATTERNS = (
    "edit image", "edit this image", "modify image", "modify this image",
    "change image", "change this image", "transform image", "transform this image",
    "background change", "change the background", "remove background",
    "replace background", "change color", "make it", "turn it into",
    "फोटो बदल", "इमेज बदल", "बैकग्राउंड बदल", "background बदल",
    "edit करो", "फोटो को बदल", "इमेज को बदल", "फोटो में बदल",
    "इमेज में बदल", "रंग बदल", "बैकग्राउंड हटा"
)

MATH_PATTERNS = (
    "solve equation", "solve this equation", "calculate", "mathematics",
    "mathematical", "math problem", "math question", "algebra",
    "geometry", "trigonometry", "percentage", "fraction", "quadratic",
    "derivative", "integral", "probability", "गणित", "मैथ",
    "गणित का सवाल", "सवाल हल", "समीकरण", "प्रतिशत"
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    normalized = normalize(text)
    return any(p in normalized for p in patterns)


def clean_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not history:
        return []

    return [
        {"role": item["role"], "content": item["content"].strip()[:MAX_MESSAGE_LENGTH]}
        for item in history[-MAX_HISTORY:]
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]


def looks_like_math_structure(message: str) -> bool:
    text = normalize(message)
    equation = bool(re.search(r"\b[a-z]\s*[\+\-\*/^]\s*[\dx0-9]", text))
    equality = "=" in text
    arithmetic = bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", text))
    fraction = bool(re.search(r"\d+\s*/\s*\d+", text))
    return equation or equality or arithmetic or fraction


def classify_request(
    message: str,
    has_image: bool = False,
) -> Intent:
    if has_image:
        if contains_pattern(message, IMAGE_EDIT_PATTERNS):
            return Intent.IMAGE_EDIT
        return Intent.VISION

    if contains_pattern(message, IMAGE_GENERATION_PATTERNS):
        return Intent.IMAGE_GENERATION
    if contains_pattern(message, QUANTUM_PATTERNS):
        return Intent.QUANTUM
    if contains_pattern(message, MATH_PATTERNS) or looks_like_math_structure(message):
        return Intent.MATH
    return Intent.NORMAL


def process_request(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    image_data: str | None = None,
    image_mime: str | None = None,
) -> dict[str, Any]:
    message = user_message.strip()
    if not message:
        raise ValueError("Please enter a message.")

    cleaned_history = clean_history(history)
    intent = classify_request(message, has_image=bool(image_data))

    if intent == Intent.MATH:
        res = solve_math(message)
        if res.get("solved"):
            return {"type": "math", "status": "completed", "answer": res["answer"], "math": res}
        answer = ask_queen(message, history=cleaned_history, system_context=res.get("context", ""))
        return {"type": "math", "status": "completed", "answer": answer, "math": res}

    if intent == Intent.QUANTUM:
        q_res = run_quantum(num_qubits=QUANTUM_QUBITS, shots=QUANTUM_SHOTS)
        prompt = f"User question: {message}\nQuantum computation: {q_res}"
        answer = ask_queen(prompt, history=cleaned_history)
        return {"type": "quantum", "status": "completed", "answer": answer, "quantum": q_res}

    if intent == Intent.IMAGE_EDIT:
        img_res = edit_image(
            prompt=message,
            image_data=image_data,
            image_mime=image_mime,
        )
        return {
            "type": "image_edit",
            "status": img_res.get("status", "completed"),
            "answer": img_res.get("answer", ""),
            "image": img_res,
        }

    if intent == Intent.VISION:
        v_res = understand_image(question=message, image_data=image_data, image_mime=image_mime)
        return {"type": "vision", "status": v_res.get("status", "completed"), "answer": v_res.get("answer", ""), "vision": v_res}

    if intent == Intent.IMAGE_GENERATION:
        img_res = generate_image(message)
        return {"type": "image", "status": img_res.get("status", "completed"), "answer": img_res.get("answer", ""), "image": img_res}

    answer = ask_queen(message, history=cleaned_history)
    return {"type": "normal", "status": "completed", "answer": answer}
