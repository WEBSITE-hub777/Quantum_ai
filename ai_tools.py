import base64
import io
import os
import re
import urllib.request
from urllib.parse import urlparse
from typing import Any

import sympy as sp
from huggingface_hub import InferenceClient


HF_TOKEN = os.getenv("HF_TOKEN")

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)

IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL",
    "black-forest-labs/FLUX.1-schnell",
)

MAX_MATH_LENGTH = 12000
MAX_IMAGE_PROMPT_LENGTH = 8000


def get_client() -> InferenceClient:

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing."
        )

    # Keep the general AI/image-generation client on automatic routing.
    return InferenceClient(
        provider="auto",
        token=HF_TOKEN,
    )


def get_vision_client() -> InferenceClient:

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing."
        )

    # Qwen2.5-VL needs a provider that supports VLM/chat vision.
    # Featherless AI currently supports chat-completion VLMs through
    # Hugging Face Inference Providers. It can be overridden on Render
    # with VISION_PROVIDER if needed.
    provider = os.getenv(
        "VISION_PROVIDER",
        "featherless-ai",
    ).strip()

    return InferenceClient(
        provider=provider,
        token=HF_TOKEN,
    )


def safe_expression(
    expression: str,
) -> str:

    expression = (
        expression
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
    )

    expression = re.sub(
        r"[^0-9a-zA-Z_+\-*/().,\s*]",
        "",
        expression,
    )

    return expression.strip()


def solve_equation(
    question: str,
) -> dict[str, Any] | None:

    match = re.search(
        r"([a-zA-Z0-9xX_().+\-*/^ ]+)"
        r"\s*=\s*"
        r"([a-zA-Z0-9xX_().+\-*/^ ]+)",
        question,
    )

    if not match:
        return None

    left = safe_expression(
        match.group(1)
    )

    right = safe_expression(
        match.group(2)
    )

    try:

        variable = sp.Symbol("x")

        left_expr = sp.sympify(
            left,
            locals={"x": variable},
        )

        right_expr = sp.sympify(
            right,
            locals={"x": variable},
        )

        equation = sp.Eq(
            left_expr,
            right_expr,
        )

        solutions = sp.solve(
            equation,
            variable,
        )

        if not solutions:
            return None

        solution_text = ", ".join(
            str(value)
            for value in solutions
        )

        return {
            "method": "symbolic_equation",
            "equation": str(equation),
            "solutions": [
                str(value)
                for value in solutions
            ],
            "answer": (
                f"Equation: {equation}\n"
                f"Solution: x = {solution_text}"
            ),
        }

    except Exception:
        return None


def solve_expression(
    question: str,
) -> dict[str, Any] | None:

    match = re.search(
        r"(?<![A-Za-z])"
        r"(\d+(?:\.\d+)?"
        r"(?:\s*[\+\-*/]\s*"
        r"\d+(?:\.\d+)?)+)"
        r"(?![A-Za-z])",
        question,
    )

    if not match:
        return None

    expression = safe_expression(
        match.group(1)
    )

    try:

        parsed = sp.sympify(
            expression
        )

        result = sp.simplify(
            parsed
        )

        return {
            "method": "symbolic_arithmetic",
            "expression": expression,
            "result": str(result),
            "answer": (
                f"{expression} = {result}"
            ),
        }

    except Exception:
        return None


def solve_math(
    question: str,
) -> dict[str, Any]:

    question = question.strip()

    if not question:
        return {
            "solved": False,
            "answer": "",
            "context": "",
        }

    if len(question) > MAX_MATH_LENGTH:
        return {
            "solved": False,
            "answer": "",
            "context": (
                "The mathematics request is too long. "
                "Solve only after understanding the "
                "complete problem."
            ),
        }

    equation_result = solve_equation(
        question
    )

    if equation_result:

        return {
            "solved": True,
            **equation_result,
            "context": "",
        }

    expression_result = solve_expression(
        question
    )

    if expression_result:

        return {
            "solved": True,
            **expression_result,
            "context": "",
        }

    return {
        "solved": False,
        "answer": "",
        "context": (
            "This is a mathematics request. "
            "Solve the complete problem carefully. "
            "Show the required steps. "
            "Do not use quantum measurement output "
            "as a substitute for mathematics."
        ),
    }


def normalize_image_data(
    image_data: str,
    image_mime: str | None,
) -> str:
    value = image_data.strip()

    if value.startswith("data:image/"):
        return value

    mime = (image_mime or "image/jpeg").strip().lower()

    if mime not in {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
    }:
        mime = "image/jpeg"

    # Render-hosted uploads are public in the browser, but some vision
    # providers cannot fetch another Render URL directly. Fetch the image
    # on our backend first and send the actual image bytes to the VLM.
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        host = parsed.netloc.lower()

        render_host = os.getenv(
            "RENDER_EXTERNAL_HOSTNAME",
            "quantum-ai-1-og05.onrender.com",
        ).lower()

        if host == render_host or host.endswith(".onrender.com"):
            request = urllib.request.Request(
                value,
                headers={
                    "User-Agent": "Quantum-Queen-AI/1.0",
                    "Accept": "image/*",
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=20,
            ) as response:
                content_type = (
                    response.headers.get("Content-Type", "")
                    .split(";")[0]
                    .strip()
                    .lower()
                )

                if content_type in {
                    "image/jpeg",
                    "image/png",
                    "image/webp",
                    "image/gif",
                }:
                    mime = content_type

                data = response.read(10 * 1024 * 1024 + 1)

            if len(data) > 10 * 1024 * 1024:
                raise RuntimeError("Uploaded image is larger than 10 MB.")

            if not data:
                raise RuntimeError("Uploaded image is empty.")

            encoded = base64.b64encode(data).decode("utf-8")
            return f"data:{mime};base64,{encoded}"

        # Keep normal external image URLs as URLs because those are already
        # supported by the configured vision provider.
        return value

    return f"data:{mime};base64,{value}"


def _extract_response_text(response: Any) -> str:
    if not getattr(response, "choices", None):
        raise RuntimeError("Vision model returned no choices.")

    message = response.choices[0].message
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()

    return str(content).strip()


def understand_image(
    question: str,
    image_data: str | None,
    image_mime: str | None = None,
) -> dict[str, Any]:
    if not image_data:
        return {
            "status": "error",
            "answer": "No image was provided.",
        }

    question = (
        question.strip()
        or "Analyze this image carefully and explain what you can see."
    )

    try:
        client = get_vision_client()
        image_url = normalize_image_data(image_data, image_mime)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ]

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=4096,
            temperature=0.2,
        )

        answer = _extract_response_text(response)

        if not answer:
            raise RuntimeError("Vision model returned empty output.")

        return {
            "status": "completed",
            "model": VISION_MODEL,
            "answer": answer,
        }

    except Exception as exc:
        return {
            "status": "error",
            "answer": (
                "Image analysis failed. The uploaded image could not "
                "be prepared for the vision model."
            ),
            "error": str(exc)[:300],
            "error_type": type(exc).__name__,
            "provider": os.getenv(
                "VISION_PROVIDER",
                "featherless-ai",
            ),
        }


def generate_image(
    prompt: str,
) -> dict[str, Any]:

    prompt = prompt.strip()

    if not prompt:
        return {
            "status": "error",
            "answer": (
                "Please provide an image description."
            ),
        }

    if len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {
            "status": "error",
            "answer": (
                "The image prompt is too long."
            ),
        }

    try:

        client = get_client()

        image = client.text_to_image(
            prompt=prompt,
            model=IMAGE_MODEL,
            width=1024,
            height=1024,
        )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG",
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return {
            "status": "completed",
            "model": IMAGE_MODEL,
            "mime": "image/png",
            "data": (
                "data:image/png;base64,"
                + encoded
            ),
            "answer": (
                "Image generated successfully."
            ),
        }

    except Exception as exc:

        return {
            "status": "error",
            "answer": (
                "Image generation failed right now."
            ),
            "error": type(exc).__name__,
        }