import base64
import os
import random
import re
import urllib.parse
import urllib.request
from typing import Any

import sympy as sp
from huggingface_hub import InferenceClient

HF_VAR_NAMES = ["HF_TOKEN", "HF1_TOKEN", "HF2_TOKEN", "HF3_TOKEN", "HF4_TOKEN", "HF_TOKENS"]
HF_TOKENS = list(dict.fromkeys([
    token.strip()
    for var in HF_VAR_NAMES
    if (val := os.getenv(var, "").strip())
    for token in val.split(",")
    if token.strip()
]))

VISION_MODEL = os.getenv("VISION_MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")
MAX_MATH_LENGTH = 12000
MAX_IMAGE_PROMPT_LENGTH = 8000


def get_vision_clients() -> list[InferenceClient]:
    if not HF_TOKENS:
        raise RuntimeError("HF_TOKEN environment variables are missing.")
    tokens = list(HF_TOKENS)
    random.shuffle(tokens)
    return [InferenceClient(provider="auto", token=t) for t in tokens]


def safe_expression(expression: str) -> str:
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    return re.sub(r"[^0-9a-zA-Z_+\-*/().,\s*]", "", expression).strip()


def solve_equation(question: str) -> dict[str, Any] | None:
    match = re.search(r"([a-zA-Z0-9xX_().+\-*/^ ]+)\s*=\s*([a-zA-Z0-9xX_().+\-*/^ ]+)", question)
    if not match:
        return None

    left = safe_expression(match.group(1))
    right = safe_expression(match.group(2))

    try:
        variable = sp.Symbol("x")
        left_expr = sp.sympify(left, locals={"x": variable})
        right_expr = sp.sympify(right, locals={"x": variable})
        equation = sp.Eq(left_expr, right_expr)
        solutions = sp.solve(equation, variable)

        if not solutions:
            return None

        solution_text = ", ".join(str(v) for v in solutions)
        return {
            "method": "symbolic_equation",
            "equation": str(equation),
            "solutions": [str(v) for v in solutions],
            "answer": f"Equation: {equation}\nSolution: x = {solution_text}",
        }
    except Exception:
        return None


def solve_expression(question: str) -> dict[str, Any] | None:
    match = re.search(r"(?<![A-Za-z])(\d+(?:\.\d+)?(?:\s*[\+\-*/]\s*\d+(?:\.\d+)?)+)(?![A-Za-z])", question)
    if not match:
        return None

    expression = safe_expression(match.group(1))
    try:
        parsed = sp.sympify(expression)
        result = sp.simplify(parsed)
        return {
            "method": "symbolic_arithmetic",
            "expression": expression,
            "result": str(result),
            "answer": f"{expression} = {result}",
        }
    except Exception:
        return None


def solve_math(question: str) -> dict[str, Any]:
    question = question.strip()
    if not question:
        return {"solved": False, "answer": "", "context": ""}

    if len(question) > MAX_MATH_LENGTH:
        return {"solved": False, "answer": "", "context": "The mathematics request is too long."}

    if eq_res := solve_equation(question):
        return {"solved": True, **eq_res, "context": ""}

    if exp_res := solve_expression(question):
        return {"solved": True, **exp_res, "context": ""}

    return {
        "solved": False,
        "answer": "",
        "context": "This is a mathematics request. Solve the complete problem carefully.",
    }


def normalize_image_data(image_data: str, image_mime: str | None) -> str:
    value = image_data.strip()
    if value.startswith(("data:image/", "http://", "https://")):
        return value

    mime = (image_mime or "image/jpeg").strip().lower()
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        mime = "image/jpeg"

    return f"data:{mime};base64,{value}"


def _extract_response_text(response: Any) -> str:
    if not getattr(response, "choices", None):
        raise RuntimeError("Vision model returned no choices.")

    message = response.choices[0].message
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "\n".join(
            str(item["text"])
            for item in content
            if isinstance(item, dict) and item.get("text")
        ).strip()

    return str(content).strip()


def understand_image(question: str, image_data: str | None, image_mime: str | None = None) -> dict[str, Any]:
    if not image_data:
        return {"status": "error", "answer": "No image was provided."}

    question = question.strip() or "Analyze this image carefully and explain what you can see."

    try:
        clients = get_vision_clients()
        image_url = normalize_image_data(image_data, image_mime)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": question},
                ],
            }
        ]

        last_error = None
        for client in clients:
            try:
                response = client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.2,
                )
                return {"status": "completed", "model": VISION_MODEL, "answer": _extract_response_text(response)}
            except Exception as exc:
                last_error = exc
                continue

        return {
            "status": "error",
            "answer": "Image analysis failed after attempting all HF tokens.",
            "error": str(last_error),
        }
    except Exception as exc:
        return {"status": "error", "answer": "Vision engine error.", "error": str(exc)}


def generate_image(prompt: str) -> dict[str, Any]:
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {"status": "error", "answer": "Invalid or overly long prompt."}

    clean_prompt = re.sub(
        r"^(generate|create|make|draw)\s+(an?\s+)?(image|picture|photo|illustration)?\s*(of|about)?\s*",
        "",
        prompt,
        flags=re.IGNORECASE,
    ).strip() or prompt

    try:
        encoded_prompt = urllib.parse.quote(clean_prompt)
        pollinations_url = f"https://pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true"

        req = urllib.request.Request(pollinations_url, headers={"User-Agent": "Quantum-Queen-AI/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            image_bytes = response.read()

        encoded_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "status": "completed",
            "model": "Pollinations-FLUX.1-schnell",
            "mime": "image/jpeg",
            "data": f"data:image/jpeg;base64,{encoded_base64}",
            "answer": "Image generated successfully.",
        }
    except Exception as exc:
        return {"status": "error", "answer": "Image generation failed.", "error": type(exc).__name__}
        