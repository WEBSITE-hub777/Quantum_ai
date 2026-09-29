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
MAX_IMAGE_BYTES = 10 * 1024 * 1024

_ALLOWED_IMAGE_MIMES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}


def get_vision_clients() -> list[InferenceClient]:
    if not HF_TOKENS:
        raise RuntimeError("HF_TOKEN environment variables are missing.")

    tokens = list(HF_TOKENS)
    random.shuffle(tokens)

    return [
        InferenceClient(
            model=VISION_MODEL,
            provider="auto",
            token=token,
        )
        for token in tokens
    ]


def safe_expression(expression: str) -> str:
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    return re.sub(r"[^0-9a-zA-Z_+\-*/().,\s*]", "", expression).strip()


def solve_equation(question: str) -> dict[str, Any] | None:
    match = re.search(
        r"([a-zA-Z0-9xX_().+\-*/^ ]+)\s*=\s*([a-zA-Z0-9xX_().+\-*/^ ]+)",
        question,
    )
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
    match = re.search(
        r"(?<![A-Za-z])(\d+(?:\.\d+)?(?:\s*[\+\-*/]\s*\d+(?:\.\d+)?)+)(?![A-Za-z])",
        question,
    )
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
        return {
            "solved": False,
            "answer": "",
            "context": "The mathematics request is too long.",
        }

    if eq_res := solve_equation(question):
        return {"solved": True, **eq_res, "context": ""}

    if exp_res := solve_expression(question):
        return {"solved": True, **exp_res, "context": ""}

    return {
        "solved": False,
        "answer": "",
        "context": "This is a mathematics request. Solve the complete problem carefully.",
    }


def _allowed_image_mime(image_mime: str | None) -> str:
    mime = (image_mime or "image/jpeg").strip().lower()
    return mime if mime in _ALLOWED_IMAGE_MIMES else "image/jpeg"


def _is_our_public_host(hostname: str | None) -> bool:
    if not hostname:
        return False

    hostname = hostname.lower().split(":", 1)[0].rstrip(".")
    allowed_hosts = set()

    render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip()
    if render_host:
        allowed_hosts.add(render_host.lower().split(":", 1)[0].rstrip("."))

    public_base_url = os.getenv("PUBLIC_BASE_URL", "").strip()
    if public_base_url:
        try:
            parsed = urllib.parse.urlparse(public_base_url)
            if parsed.hostname:
                allowed_hosts.add(parsed.hostname.lower().rstrip("."))
        except ValueError:
            pass

    return hostname in allowed_hosts


def _download_own_image_as_data_url(url: str, fallback_mime: str) -> str:
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme != "https":
        raise ValueError("Uploaded image URL must use HTTPS.")

    if not _is_our_public_host(parsed.hostname):
        return url

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Quantum-Queen-AI/1.0"},
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        content_type = response.headers.get_content_type().lower()
        if content_type not in _ALLOWED_IMAGE_MIMES:
            content_type = fallback_mime

        image_bytes = response.read(MAX_IMAGE_BYTES + 1)

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image is larger than the 10 MB limit.")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def normalize_image_data(image_data: str, image_mime: str | None) -> str:
    value = image_data.strip()
    mime = _allowed_image_mime(image_mime)

    if value.startswith("data:image/"):
        return value

    if value.startswith("https://") or value.startswith("http://"):
        return _download_own_image_as_data_url(value, mime)

    # A browser/mobile content:// URI is not readable by the Hugging Face
    # server. It must be uploaded through /upload-image first.
    if value.startswith("content://") or value.startswith("file://"):
        raise ValueError(
            "The app sent a local content/file URI instead of the uploaded image URL."
        )

    # Otherwise the API received raw base64.
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


def understand_image(
    question: str,
    image_data: str | None,
    image_mime: str | None = None,
) -> dict[str, Any]:
    if not image_data:
        return {"status": "error", "answer": "No image was provided."}

    question = question.strip() or "Analyze this image carefully and explain what you can see."

    try:
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

        clients = get_vision_clients()
        errors: list[str] = []

        for index, client in enumerate(clients, start=1):
            try:
                response = client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.2,
                )

                answer = _extract_response_text(response)
                if not answer:
                    raise RuntimeError("Vision model returned an empty response.")

                return {
                    "status": "completed",
                    "model": VISION_MODEL,
                    "answer": answer,
                }

            except Exception as exc:
                # Keep the real provider error for debugging, but never expose
                # the token itself.
                errors.append(f"HF attempt {index}: {type(exc).__name__}: {exc}")

        return {
            "status": "error",
            "answer": "Image analysis failed. The vision provider rejected all configured attempts.",
            "error": " | ".join(errors[-4:]),
        }

    except Exception as exc:
        return {
            "status": "error",
            "answer": "Vision engine error.",
            "error": f"{type(exc).__name__}: {exc}",
        }


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
        pollinations_url = (
            f"https://pollinations.ai/p/{encoded_prompt}"
            "?width=1024&height=1024&model=flux&nologo=true"
        )

        req = urllib.request.Request(
            pollinations_url,
            headers={"User-Agent": "Quantum-Queen-AI/1.0"},
        )
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
        return {
            "status": "error",
            "answer": "Image generation failed.",
            "error": type(exc).__name__,
        }
