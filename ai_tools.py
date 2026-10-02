import base64
import io
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

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "deepseek-ai/DeepSeek-V4.1-Flash",
).strip()

MAX_MATH_LENGTH = 12000
MAX_IMAGE_PROMPT_LENGTH = 8000
MAX_IMAGE_BYTES = 20 * 1024 * 1024

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
            timeout=120,
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

    if value.startswith("content://") or value.startswith("file://"):
        raise ValueError(
            "The app sent a local content/file URI instead of the uploaded image URL."
        )

    return f"data:{mime};base64,{value}"


def _image_data_to_bytes(image_data: str, image_mime: str | None) -> bytes:
    normalized = normalize_image_data(image_data, image_mime)

    if normalized.startswith("data:"):
        try:
            _, encoded = normalized.split(",", 1)
            image_bytes = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("Invalid image data.") from exc
    else:
        request = urllib.request.Request(
            normalized,
            headers={"User-Agent": "Quantum-Queen-AI/1.0"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            image_bytes = response.read(MAX_IMAGE_BYTES + 1)

    if not image_bytes:
        raise ValueError("Image data is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image is larger than the 10 MB limit.")

    return image_bytes


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


# ---------------------------------------------------------------------------
# Pollinations image stack
# ---------------------------------------------------------------------------
POLLINATIONS_API_KEY = os.getenv("Quantum1", "").strip()
POLLINATIONS_BASE_URL = "https://gen.pollinations.ai"
POLLINATIONS_VISION_MODEL = os.getenv(
    "POLLINATIONS_VISION_MODEL",
    "deepseek/deepseek-v4-flash-vision-exp",
).strip()
POLLINATIONS_IMAGE_MODEL = os.getenv(
    "POLLINATIONS_IMAGE_MODEL",
    "black-forest-labs/flux.1-kontext-pro",
).strip()
POLLINATIONS_IMAGE_EDIT_MODEL = os.getenv(
    "POLLINATIONS_IMAGE_EDIT_MODEL",
    "black-forest-labs/flux.1-kontext-pro",
).strip()


def _pollinations_headers() -> dict[str, str]:
    if not POLLINATIONS_API_KEY:
        raise RuntimeError("Quantum1 environment variable is missing.")
    return {
        "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
        "Content-Type": "application/json",
    }


def _pollinations_json_post(
    path: str,
    payload: dict[str, Any],
    timeout: int = 180,
) -> dict[str, Any]:
    import json
    import urllib.error

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{POLLINATIONS_BASE_URL}{path}",
        data=body,
        headers=_pollinations_headers(),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1500]
        raise RuntimeError(f"Pollinations HTTP {exc.code}: {detail}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("Pollinations returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Pollinations returned an invalid response.")
    return payload


def _extract_pollinations_image(response: dict[str, Any]) -> str:
    items = response.get("data") or []
    if not items or not isinstance(items[0], dict):
        raise RuntimeError(f"Pollinations returned no image: {str(response)[:1200]}")

    item = items[0]
    if item.get("b64_json"):
        return "data:image/png;base64," + str(item["b64_json"])

    # Never expose a provider URL to the frontend. Download it on the
    # backend and convert it to an inline data URL first.
    if item.get("url"):
        return _download_generated_image(str(item["url"]))

    raise RuntimeError(
        f"Pollinations image response contained neither b64_json nor url: "
        f"{str(response)[:1200]}"
    )


def _download_generated_image(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError("Pollinations returned an invalid image URL.")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Quantum-Queen-AI/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_type = response.headers.get_content_type().lower()
            data = response.read(MAX_IMAGE_BYTES + 1)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download generated image: {type(exc).__name__}: {exc}"
        ) from exc

    if len(data) > MAX_IMAGE_BYTES:
        raise RuntimeError("Generated image is larger than the 20 MiB limit.")
    if not data:
        raise RuntimeError("Generated image is empty.")

    # Do not turn an HTML/error page into a fake PNG data URL.
    allowed = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/svg+xml",
    }
    if content_type not in allowed:
        preview = data[:120].decode("utf-8", errors="replace").replace("\n", " ")
        raise RuntimeError(
            f"Generated URL returned non-image content ({content_type}). "
            f"Preview: {preview}"
        )

    return (
        f"data:{content_type};base64,"
        + base64.b64encode(data).decode("ascii")
    )


def _generate_image_via_direct_get(prompt: str) -> str:
    """Generate directly from Pollinations and always return image bytes as a data URL.

    This avoids surfacing provider-hosted URLs or proxy pages to the browser.
    """
    encoded_prompt = urllib.parse.quote(prompt, safe="")
    query = urllib.parse.urlencode({"model": POLLINATIONS_IMAGE_MODEL})
    url = f"{POLLINATIONS_BASE_URL}/image/{encoded_prompt}?{query}"

    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
            "User-Agent": "Quantum-Queen-AI/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            content_type = response.headers.get_content_type().lower()
            data = response.read(MAX_IMAGE_BYTES + 1)
    except Exception as exc:
        raise RuntimeError(
            f"Pollinations direct image generation failed: {type(exc).__name__}: {exc}"
        ) from exc

    if len(data) > MAX_IMAGE_BYTES:
        raise RuntimeError("Generated image is larger than the 20 MiB limit.")
    if not data:
        raise RuntimeError("Pollinations returned an empty image.")

    allowed = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/svg+xml",
    }
    if content_type not in allowed:
        preview = data[:160].decode("utf-8", errors="replace").replace("\n", " ")
        raise RuntimeError(
            f"Pollinations direct endpoint returned non-image content ({content_type}). "
            f"Preview: {preview}"
        )

    return (
        f"data:{content_type};base64,"
        + base64.b64encode(data).decode("ascii")
    )


def _prepare_pollinations_image(image_data: str, image_mime: str | None) -> str:
    normalized = normalize_image_data(image_data, image_mime)
    if normalized.startswith("data:image/"):
        return normalized
    return normalized


def understand_image(
    question: str,
    image_data: str | None,
    image_mime: str | None = None,
) -> dict[str, Any]:
    if not image_data:
        return {"status": "error", "answer": "No image was provided."}

    question = (
        question.strip()
        or "Analyze this image carefully and explain what you can see."
    )

    try:
        image_url = _prepare_pollinations_image(image_data, image_mime)

        response = _pollinations_json_post(
            "/v1/chat/completions",
            {
                "model": POLLINATIONS_VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 4096,
            },
            timeout=180,
        )

        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("Pollinations vision returned no choices.")

        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("text")
            )

        answer = str(content).strip()
        if not answer:
            raise RuntimeError("Pollinations vision returned an empty response.")

        return {
            "status": "completed",
            "model": POLLINATIONS_VISION_MODEL,
            "provider": "Pollinations",
            "answer": answer,
        }
    except Exception as exc:
        return {
            "status": "error",
            "answer": "Image analysis failed.",
            "error": f"{type(exc).__name__}: {exc}",
        }


def generate_image(prompt: str) -> dict[str, Any]:
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {"status": "error", "answer": "Invalid or overly long prompt."}

    clean_prompt = re.sub(
        r"^(generate|create|make|draw)\s+(an?\s+)?"
        r"(image|picture|photo|illustration)?\s*(of|about)?\s*",
        "",
        prompt,
        flags=re.IGNORECASE,
    ).strip() or prompt

    try:
        # Use Pollinations' native image endpoint and keep the generated
        # bytes entirely on the backend. The browser receives only a data URL.
        data_url = _generate_image_via_direct_get(clean_prompt)

        return {
            "status": "completed",
            "model": POLLINATIONS_IMAGE_MODEL,
            "provider": "Pollinations",
            "mime": data_url.split(";", 1)[0].replace("data:", ""),
            "data": data_url,
            "answer": "Image generated successfully with Pollinations.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "answer": "Image generation failed.",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _prepare_edit_prompt(prompt: str) -> str:
    text = re.sub(r"\s+", " ", prompt.strip())
    replacements = (
        ("iske pichhe", "behind the subject"),
        ("iske peeche", "behind the subject"),
        ("is ke pichhe", "behind the subject"),
        ("is ke peeche", "behind the subject"),
        ("iske saamne", "in front of the subject"),
        ("iske samne", "in front of the subject"),
        ("is ke saamne", "in front of the subject"),
        ("is ke samne", "in front of the subject"),
        ("background mein", "in the background"),
        ("background me", "in the background"),
        ("background mai", "in the background"),
        ("peeche", "behind the subject"),
        ("pichhe", "behind the subject"),
        ("saamne", "in front of the subject"),
        ("samne", "in front of the subject"),
        ("laga do", "add"),
        ("laga de", "add"),
        ("bana do", "add"),
        ("bana de", "add"),
        ("jod do", "add"),
        ("jod de", "add"),
        ("hata do", "remove"),
        ("hata de", "remove"),
        ("jungle", "forest"),
        ("जंगल", "forest"),
        ("पीछे", "behind the subject"),
        ("सामने", "in front of the subject"),
        ("बैकग्राउंड", "background"),
        ("लगा दो", "add"),
        ("बना दो", "add"),
        ("जोड़ दो", "add"),
        ("हटा दो", "remove"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def _multipart_edit_request(
    image_bytes: bytes,
    image_mime: str,
    prompt: str,
    model: str,
    timeout: int = 300,
) -> dict[str, Any]:
    """Call the OpenAI-compatible Pollinations image-edit endpoint."""
    import json
    import urllib.error
    import uuid

    boundary = "----QuantumQueenBoundary" + uuid.uuid4().hex
    chunks: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )

    extension = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
    }.get(image_mime, "png")

    add_field("model", model)
    add_field("prompt", prompt)
    add_field("response_format", "b64_json")

    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="input.{extension}"\r\n'
            f"Content-Type: {image_mime}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(image_bytes)
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        f"{POLLINATIONS_BASE_URL}/v1/images/edits",
        data=b"".join(chunks),
        headers={
            "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1500]
        raise RuntimeError(
            f"Pollinations edit HTTP {exc.code}: {detail}"
        ) from exc

    try:
        result = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("Pollinations edit returned invalid JSON.") from exc

    if not isinstance(result, dict):
        raise RuntimeError("Pollinations edit returned an invalid response.")
    return result


def edit_image(
    prompt: str,
    image_data: str | None,
    image_mime: str | None = None,
) -> dict[str, Any]:
    prompt = _prepare_edit_prompt(prompt)

    if not image_data:
        return {
            "status": "error",
            "answer": "No image was provided for editing.",
        }

    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {
            "status": "error",
            "answer": "Invalid or overly long edit prompt.",
        }

    if not POLLINATIONS_API_KEY:
        return {
            "status": "error",
            "answer": "Image editing failed.",
            "error": "Quantum1 environment variable is missing.",
        }

    try:
        mime = _allowed_image_mime(image_mime)
        image_bytes = _image_data_to_bytes(image_data, mime)

        response = _multipart_edit_request(
            image_bytes=image_bytes,
            image_mime=mime,
            prompt=prompt,
            model=POLLINATIONS_IMAGE_EDIT_MODEL,
            timeout=300,
        )

        data_url = _extract_pollinations_image(response)

        return {
            "status": "completed",
            "model": POLLINATIONS_IMAGE_EDIT_MODEL,
            "provider": "Pollinations",
            "mime": data_url.split(";", 1)[0].replace("data:", ""),
            "data": data_url,
            "answer": "Image edited successfully with Pollinations.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "answer": "Image editing failed.",
            "error": f"{type(exc).__name__}: {exc}",
        }
