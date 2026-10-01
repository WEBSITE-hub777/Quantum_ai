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

# xAI Grok handles image understanding plus Imagine generation/editing.
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
GROK_VISION_MODEL = os.getenv("GROK_VISION_MODEL", "grok-4.7").strip()
GROK_IMAGE_MODEL = os.getenv("GROK_IMAGE_MODEL", "grok-imagine-image-2.0").strip()
XAI_API_URL = "https://api.x.ai/v1"

# FLUX for text -> image through Hugging Face Inference Providers.
IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL",
    "black-forest-labs/FLUX.1-dev",
).strip()

# FLUX Kontext for image + prompt -> edited image.
IMAGE_EDIT_MODEL = os.getenv(
    "IMAGE_EDIT_MODEL",
    "black-forest-labs/FLUX.1-Kontext-dev",
).strip()

# Try the dedicated image provider first, then Hugging Face automatic
# provider selection as a fallback.
IMAGE_EDIT_PROVIDERS = [
    provider.strip()
    for provider in os.getenv("IMAGE_EDIT_PROVIDERS", "fal-ai").split(",")
    if provider.strip()
]

IMAGE_EDIT_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "IMAGE_EDIT_FALLBACK_MODELS",
        "black-forest-labs/FLUX.2-klein-9B",
    ).split(",")
    if model.strip() and model.strip() != IMAGE_EDIT_MODEL
]

IMAGE_EDIT_STEPS = max(
    10,
    min(int(os.getenv("IMAGE_EDIT_STEPS", "25")), 50),
)
IMAGE_EDIT_TIMEOUT = max(
    60,
    min(int(os.getenv("IMAGE_EDIT_TIMEOUT", "150")), 300),
)

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


def _xai_request(path: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY environment variable is missing.")
    import json
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{XAI_API_URL}{path}",
        data=body,
        headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1500]
        raise RuntimeError(f"xAI HTTP {exc.code}: {detail}") from exc
    return json.loads(raw.decode("utf-8"))


def _image_data_uri_for_grok(image_data: str, image_mime: str | None) -> str:
    normalized = normalize_image_data(image_data, image_mime)
    if not normalized.startswith("data:image/"):
        return normalized
    header, encoded = normalized.split(",", 1)
    mime = header.split(";", 1)[0].lower()
    if mime in {"data:image/jpeg", "data:image/png"}:
        return normalized
    try:
        from PIL import Image
        raw = base64.b64decode(encoded)
        with Image.open(io.BytesIO(raw)) as source:
            converted = source.convert("RGB")
            buffer = io.BytesIO()
            converted.save(buffer, format="PNG", optimize=True)
            png = buffer.getvalue()
        if len(png) > MAX_IMAGE_BYTES:
            raise ValueError("Converted image is larger than the 20 MiB limit.")
        return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    except Exception as exc:
        raise ValueError(f"Could not prepare image for Grok: {exc}") from exc


def _download_generated_image(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Quantum-Queen-AI/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            content_type = response.headers.get_content_type().lower()
            data = response.read(MAX_IMAGE_BYTES + 1)
    except Exception as exc:
        raise RuntimeError(f"Could not download generated image: {type(exc).__name__}: {exc}") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise RuntimeError("Generated image is larger than the 20 MiB limit.")
    if not data:
        raise RuntimeError("Generated image is empty.")
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        content_type = "image/jpeg"
    return f"data:{content_type};base64," + base64.b64encode(data).decode("ascii")


def _extract_xai_b64(response: dict[str, Any]) -> str:
    items = response.get("data") or []
    if not items or not isinstance(items[0], dict):
        raise RuntimeError("xAI returned no image.")
    encoded = items[0].get("b64_json")
    if not encoded:
        raise RuntimeError("xAI returned no base64 image.")
    return "data:image/jpeg;base64," + encoded


def _extract_xai_image_url(response: dict[str, Any]) -> str:
    items = response.get("data") or []
    if not items or not isinstance(items[0], dict) or not items[0].get("url"):
        raise RuntimeError("xAI returned no image.")
    return items[0]["url"]


def understand_image(
    question: str,
    image_data: str | None,
    image_mime: str | None = None,
) -> dict[str, Any]:
    if not image_data:
        return {"status": "error", "answer": "No image was provided."}
    question = question.strip() or "Analyze this image carefully and explain what you can see."
    try:
        image_url = _image_data_uri_for_grok(image_data, image_mime)
        response = _xai_request("/chat/completions", {
            "model": GROK_VISION_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
            "temperature": 0.2,
        }, timeout=120)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("Grok returned no choices.")
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("text"))
        answer = str(content).strip()
        if not answer:
            raise RuntimeError("Grok returned an empty response.")
        return {"status": "completed", "model": GROK_VISION_MODEL, "answer": answer}
    except Exception as exc:
        return {"status": "error", "answer": "Image analysis failed.", "error": f"{type(exc).__name__}: {exc}"}


def _image_to_png_data_url(image: Any) -> str:
    if image is None:
        raise RuntimeError("Hugging Face returned no image.")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    image_bytes = buffer.getvalue()

    if not image_bytes:
        raise RuntimeError("Generated image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise RuntimeError("Generated image is larger than the 10 MB limit.")

    encoded_base64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded_base64}"


def generate_image(prompt: str) -> dict[str, Any]:
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {"status": "error", "answer": "Invalid or overly long prompt."}
    clean_prompt = re.sub(
        r"^(generate|create|make|draw)\s+(an?\s+)?(image|picture|photo|illustration)?\s*(of|about)?\s*",
        "", prompt, flags=re.IGNORECASE,
    ).strip() or prompt
    try:
        # Let xAI return its default temporary URL, then download it on
        # the backend and convert it to a data URL for the frontend. This
        # avoids depending on b64_json support/format differences.
        response = _xai_request("/images/generations", {
            "model": GROK_IMAGE_MODEL,
            "prompt": clean_prompt,
        }, timeout=240)

        items = response.get("data") or []
        if not items or not isinstance(items[0], dict):
            raise RuntimeError(f"xAI returned no image data: {str(response)[:1200]}")

        item = items[0]
        if item.get("url"):
            data_url = _download_generated_image(item["url"])
        elif item.get("b64_json"):
            data_url = _extract_xai_b64(response)
        else:
            raise RuntimeError(f"xAI image response contained neither url nor b64_json: {str(response)[:1200]}")

        return {
            "status": "completed",
            "model": GROK_IMAGE_MODEL,
            "mime": "image/jpeg",
            "data": data_url,
            "answer": "Image generated successfully with Grok Imagine.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "answer": "Image generation failed.",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _prepare_edit_prompt(prompt: str) -> str:
    text = re.sub(r"\\s+", " ", prompt.strip())
    replacements = (
        ("iske pichhe", "behind the subject"), ("iske peeche", "behind the subject"),
        ("is ke pichhe", "behind the subject"), ("is ke peeche", "behind the subject"),
        ("iske saamne", "in front of the subject"), ("iske samne", "in front of the subject"),
        ("is ke saamne", "in front of the subject"), ("is ke samne", "in front of the subject"),
        ("background mein", "in the background"), ("background me", "in the background"),
        ("background mai", "in the background"), ("peeche", "behind the subject"),
        ("pichhe", "behind the subject"), ("saamne", "in front of the subject"),
        ("samne", "in front of the subject"), ("laga do", "add"), ("laga de", "add"),
        ("bana do", "add"), ("bana de", "add"), ("jod do", "add"), ("jod de", "add"),
        ("hata do", "remove"), ("hata de", "remove"), ("jungle", "forest"), ("जंगल", "forest"),
        ("पीछे", "behind the subject"), ("सामने", "in front of the subject"),
        ("बैकग्राउंड", "background"), ("लगा दो", "add"), ("बना दो", "add"),
        ("जोड़ दो", "add"), ("हटा दो", "remove"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def edit_image(
    prompt: str,
    image_data: str | None,
    image_mime: str | None = None,
) -> dict[str, Any]:
    prompt = _prepare_edit_prompt(prompt)
    if not image_data:
        return {"status": "error", "answer": "No image was provided for editing."}
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {"status": "error", "answer": "Invalid or overly long edit prompt."}
    try:
        image_url = _image_data_uri_for_grok(image_data, image_mime)
        response = _xai_request("/images/edits", {
            "model": GROK_IMAGE_MODEL,
            "prompt": prompt,
            "image": {"url": image_url, "type": "image_url"},
            "response_format": "url",
        }, timeout=240)
        data_url = _download_generated_image(_extract_xai_image_url(response))
        return {"status": "completed", "model": GROK_IMAGE_MODEL, "provider": "xAI",
                "mime": "image/jpeg", "data": data_url,
                "answer": "Image edited successfully with Grok Imagine."}
    except Exception as exc:
        return {"status": "error", "answer": "Image editing failed.",
                "error": f"{type(exc).__name__}: {exc}"}
def edit_image(
    prompt: str,
    image_data: str | None,
    image_mime: str | None = None,
) -> dict[str, Any]:
    prompt = _prepare_edit_prompt(prompt)
    if not image_data:
        return {"status": "error", "answer": "No image was provided for editing."}
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {"status": "error", "answer": "Invalid or overly long edit prompt."}

    if not HF_TOKENS:
        return {
            "status": "error",
            "answer": "Image editing failed.",
            "error": "HF_TOKEN environment variables are missing.",
        }

    try:
        image_bytes = _image_data_to_bytes(image_data, image_mime)
    except Exception as exc:
        return {
            "status": "error",
            "answer": "Image editing failed.",
            "error": f"{type(exc).__name__}: {exc}",
        }

    tokens = list(HF_TOKENS)
    random.shuffle(tokens)
    errors: list[str] = []
    attempt = 0

    models = [IMAGE_EDIT_MODEL, *IMAGE_EDIT_FALLBACK_MODELS]

    for model in models:
        for provider in IMAGE_EDIT_PROVIDERS:
            for token in tokens:
                attempt += 1
                try:
                    client = InferenceClient(
                        provider=provider,
                        api_key=token,
                        timeout=IMAGE_EDIT_TIMEOUT,
                    )

                    image = client.image_to_image(
                        image_bytes,
                        prompt=prompt,
                        model=model,
                        num_inference_steps=IMAGE_EDIT_STEPS,
                    )

                    return {
                        "status": "completed",
                        "model": model,
                        "provider": provider,
                        "mime": "image/png",
                        "data": _image_to_png_data_url(image),
                        "answer": "Image edited successfully.",
                    }

                except Exception as exc:
                    errors.append(
                        f"HF edit attempt {attempt} ({model}, {provider}): "
                        f"{type(exc).__name__}: {str(exc)[:350]}"
                    )

    return {
        "status": "error",
        "answer": "Image editing failed.",
        "error": " | ".join(errors[-4:]),
    }
