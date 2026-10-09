import base64
import io
import os
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
    "Qwen/Qwen2.5-VL-3B-Instruct",
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
        raise ValueError("Image is larger than the 20 MB limit.")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _optimize_local_image_bytes(image_bytes: bytes, mime: str) -> tuple[bytes, str]:
    """
    Keep large browser-uploaded images provider-friendly.
    Images are downscaled to a reasonable vision/edit size and compressed only
    when that produces a smaller payload.
    """
    try:
        from PIL import Image

        source = Image.open(io.BytesIO(image_bytes))
        source.load()

        max_side = 1600
        if max(source.size) > max_side:
            source.thumbnail(
                (max_side, max_side),
                Image.Resampling.LANCZOS,
            )

        working = source.convert("RGB") if source.mode not in {"RGB", "L"} else source
        output = io.BytesIO()
        working.save(output, format="JPEG", quality=88, optimize=True)
        compressed = output.getvalue()

        if len(compressed) < len(image_bytes):
            return compressed, "image/jpeg"
    except Exception:
        pass

    return image_bytes, mime


def normalize_image_data(image_data: str, image_mime: str | None) -> str:
    value = image_data.strip()
    mime = _allowed_image_mime(image_mime)

    if value.startswith("data:image/"):
        try:
            _, encoded = value.split(",", 1)
            raw = base64.b64decode(encoded, validate=True)
            optimized, optimized_mime = _optimize_local_image_bytes(raw, mime)
            encoded = base64.b64encode(optimized).decode("ascii")
            return f"data:{optimized_mime};base64,{encoded}"
        except Exception as exc:
            raise ValueError("Invalid image data.") from exc

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
        raise ValueError("Image is larger than the 20 MB limit.")

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
# Hugging Face Llama + FLUX image stack
# ---------------------------------------------------------------------------
HF_VISION_MODEL = os.getenv(
    "HF_VISION_MODEL",
    "Qwen/Qwen2.5-VL-3B-Instruct",
).strip()

HF_IMAGE_MODEL = os.getenv(
    "HF_IMAGE_MODEL",
    "black-forest-labs/FLUX.1-schnell",
).strip()

HF_IMAGE_EDIT_MODEL = os.getenv(
    "HF_IMAGE_EDIT_MODEL",
    "Qwen/Qwen-Image-Edit",
).strip()

HF_VISION_FALLBACK_MODEL = os.getenv(
    "HF_VISION_FALLBACK_MODEL",
    "zai-org/GLM-4.5V",
).strip()
HF_IMAGE_FALLBACK_MODEL = os.getenv("HF_IMAGE_FALLBACK_MODEL", "Qwen/Qwen-Image").strip()
HF_IMAGE_EDIT_FALLBACK_MODEL = os.getenv("HF_IMAGE_EDIT_FALLBACK_MODEL", "black-forest-labs/FLUX.1-Kontext-dev").strip()


def _hf_image_clients() -> list[InferenceClient]:
    if not HF_TOKENS:
        raise RuntimeError("No HF tokens configured.")

    tokens = list(HF_TOKENS)
    return [
        InferenceClient(
            api_key=token,
            provider="auto",
            timeout=180,
        )
        for token in tokens
    ]


def _pil_to_data_url(image: Any) -> str:
    if not hasattr(image, "save"):
        raise RuntimeError("Hugging Face returned an invalid image object.")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _hf_vision_content(image_url: str, question: str) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]


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
        image_url = normalize_image_data(image_data, image_mime)
        messages = [
            {
                "role": "user",
                "content": _hf_vision_content(image_url, question),
            }
        ]

        errors: list[str] = []
        models = [HF_VISION_MODEL, HF_VISION_FALLBACK_MODEL]
        for model_name in dict.fromkeys(models):
            for client in _hf_image_clients():
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.2,
                    )
                    answer = _extract_response_text(response)
                    if answer:
                        return {
                            "status": "completed",
                            "model": model_name,
                            "provider": "Hugging Face",
                            "answer": answer,
                        }
                    errors.append(f"{model_name}: empty response")
                except Exception as exc:
                    errors.append(f"{model_name}: {type(exc).__name__}: {str(exc)[:250]}")

        raise RuntimeError(" | ".join(errors[-4:]) or "All vision attempts failed.")
    except Exception as exc:
        return {
            "status": "error",
            "answer": "Image analysis failed.",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _hf_generate_image(prompt: str) -> tuple[str, str]:
    errors: list[str] = []

    for model_name in dict.fromkeys([HF_IMAGE_MODEL, HF_IMAGE_FALLBACK_MODEL]):
        for client in _hf_image_clients():
            try:
                image = client.text_to_image(prompt=prompt, model=model_name)
                return _pil_to_data_url(image), model_name
            except Exception as exc:
                errors.append(f"{model_name}: {type(exc).__name__}: {str(exc)[:250]}")

    raise RuntimeError(
        "All Hugging Face image-generation attempts failed: "
        + " | ".join(errors[-4:])
    )


def _pollinations_generate_image(prompt: str) -> tuple[str, str]:
    api_key = os.getenv("Quantum1", "").strip()
    if not api_key:
        raise RuntimeError("Pollinations image API key is not configured.")

    model = os.getenv("POLLINATIONS_IMAGE_MODEL", "flux").strip() or "flux"
    encoded_prompt = urllib.parse.quote(prompt, safe="")
    url = f"https://gen.pollinations.ai/image/{encoded_prompt}?model={urllib.parse.quote(model, safe='/')}"

    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Quantum-Queen-AI/1.0",
        },
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        image_bytes = response.read(MAX_IMAGE_BYTES + 1)
        content_type = response.headers.get("Content-Type", "image/jpeg").split(";", 1)[0].lower()

    if not image_bytes:
        raise RuntimeError("Pollinations returned an empty image.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise RuntimeError("Pollinations image is larger than the 20 MB limit.")
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        content_type = "image/jpeg"

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{content_type};base64,{encoded}", f"Pollinations/{model}"


def _prepare_generation_prompt(prompt: str) -> str:
    text = re.sub(r"\s+", " ", prompt.strip())
    replacements = (
        ("एक इमेज बनाओ", "create an image"),
        ("एक इमेज बना दो", "create an image"),
        ("इमेज बनाओ", "create an image"),
        ("इमेज बना दो", "create an image"),
        ("एक फोटो बनाओ", "create a photo"),
        ("फोटो बनाओ", "create a photo"),
        ("तस्वीर बनाओ", "create an image"),
        ("चित्र बनाओ", "create an image"),
        ("बनाओ", "create"),
        ("बना दो", "create"),
        ("बना", "create"),
        ("बनवा दो", "create"),
        ("तैयार करो", "create"),
        ("एक", "a"),
        ("और", "and"),
        ("के साथ", "with"),
        ("के ऊपर", "on top of"),
        ("के नीचे", "under"),
        ("पीछे", "behind"),
        ("सामने", "in front of"),
        ("बैकग्राउंड", "background"),
        ("जंगल", "forest"),
        ("आसमान", "sky"),
        ("समुद्र", "ocean"),
        ("पहाड़", "mountain"),
        ("घर", "house"),
        ("पेड़", "tree"),
        ("फूल", "flower"),
        ("सूरज", "sun"),
        ("चाँद", "moon"),
        ("लड़का", "boy"),
        ("लड़की", "girl"),
        ("बिल्ली", "cat"),
        ("कुत्ता", "dog"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    text = re.sub(
        r"^(generate|create|make|draw)\s+(an?\s+)?"
        r"(image|picture|photo|illustration)?\s*(of|about)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text or prompt


def generate_image(prompt: str) -> dict[str, Any]:
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {"status": "error", "answer": "Invalid or overly long prompt."}

    clean_prompt = _prepare_generation_prompt(prompt)

    try:
        data_url, used_model = _hf_generate_image(clean_prompt)
        return {
            "status": "completed",
            "model": used_model,
            "provider": "Hugging Face",
            "mime": "image/png",
            "data": data_url,
            "answer": f"Image generated successfully with {used_model}.",
        }
    except Exception as hf_exc:
        try:
            data_url, used_model = _pollinations_generate_image(clean_prompt)
            return {
                "status": "completed",
                "model": used_model,
                "provider": "Pollinations",
                "mime": "image/png",
                "data": data_url,
                "answer": f"Image generated successfully with {used_model}.",
            }
        except Exception as pollinations_exc:
            return {
                "status": "error",
                "answer": "Image generation failed.",
                "error": (
                    f"Hugging Face: {type(hf_exc).__name__}: {hf_exc} | "
                    f"Pollinations: {type(pollinations_exc).__name__}: {pollinations_exc}"
                ),
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
        ("हटा", "remove"),
        ("निकाल दो", "remove"),
        ("निकाल", "remove"),
        ("बदल दो", "change"),
        ("बदल", "change"),
        ("इसका", "its"),
        ("इसकी", "its"),
        ("इसमें", "in it"),
        ("एक", "a"),
        ("और", "and"),
        ("पेड़", "tree"),
        ("जंगल", "forest"),
        ("समुद्र", "ocean"),
        ("आसमान", "sky"),
        ("आदमी", "man"),
        ("लड़का", "boy"),
        ("लड़की", "girl"),
        ("बिल्ली", "cat"),
        ("कुत्ता", "dog"),
    )
    for source, target in replacements:
        text = text.replace(source, target)

    text = re.sub(
        r"\bbehind the subject\s+(?:a\s+)?(.+?)\s+add$",
        r"add a \1 behind the subject",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bin front of the subject\s+(?:a\s+)?(.+?)\s+add$",
        r"add a \1 in front of the subject",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _hf_edit_image(image_bytes: bytes, prompt: str, image_mime: str) -> tuple[str, str]:
    from PIL import Image

    source = Image.open(io.BytesIO(image_bytes))
    errors: list[str] = []

    for model_name in dict.fromkeys([HF_IMAGE_EDIT_MODEL, HF_IMAGE_EDIT_FALLBACK_MODEL]):
        for client in _hf_image_clients():
            try:
                edited = client.image_to_image(
                    image=source,
                    prompt=prompt,
                    model=model_name,
                )
                data_url = _pil_to_data_url(edited)
                if not data_url.startswith("data:image/"):
                    raise RuntimeError("Image editor did not return image data.")
                return data_url, model_name
            except Exception as exc:
                errors.append(f"{model_name}: {type(exc).__name__}: {str(exc)[:250]}")

    raise RuntimeError(
        "All Hugging Face image-edit attempts failed: "
        + " | ".join(errors[-4:])
    )



def _pollinations_edit_image(image_bytes: bytes, prompt: str, image_mime: str) -> tuple[str, str]:
    """Edit an uploaded image through Pollinations' OpenAI-compatible edits endpoint."""
    api_key = os.getenv("Quantum1", "").strip()
    if not api_key:
        raise RuntimeError("Pollinations image API key is not configured.")

    boundary = "----QuantumQueenImageEditBoundary7MA4YWxk"
    safe_mime = _allowed_image_mime(image_mime)
    filename = "upload." + {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif"}.get(safe_mime, "jpg")

    def field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")

    body = b"".join([
        field("prompt", prompt),
        field("model", os.getenv("POLLINATIONS_EDIT_MODEL", "kontext").strip() or "kontext"),
        field("size", "1024x1024"),
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: {safe_mime}\r\n\r\n"
        ).encode("utf-8"),
        image_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ])

    request = urllib.request.Request(
        "https://gen.pollinations.ai/v1/images/edits",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Quantum-Queen-AI/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = response.read(MAX_IMAGE_BYTES * 2 + 1024 * 1024)

    try:
        payload = __import__("json").loads(result.decode("utf-8"))
        item = (payload.get("data") or [{}])[0]
        encoded = item.get("b64_json")
        if encoded:
            return f"data:image/png;base64,{encoded}", "Pollinations/kontext"
        image_url = item.get("url")
        if image_url:
            if not image_url.startswith("https://"):
                raise RuntimeError("Pollinations returned a non-HTTPS image URL.")
            req = urllib.request.Request(image_url, headers={"User-Agent": "Quantum-Queen-AI/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                edited_bytes = response.read(MAX_IMAGE_BYTES + 1)
                content_type = response.headers.get("Content-Type", "image/png").split(";", 1)[0].lower()
            if len(edited_bytes) > MAX_IMAGE_BYTES:
                raise RuntimeError("Edited image is larger than the 20 MB limit.")
            if content_type not in {"image/jpeg", "image/png", "image/webp"}:
                content_type = "image/png"
            return f"data:{content_type};base64,{base64.b64encode(edited_bytes).decode('ascii')}", "Pollinations/kontext"
        raise RuntimeError("Pollinations edit response did not contain image data.")
    except (UnicodeDecodeError, ValueError):
        # Some compatible providers may return the image bytes directly.
        if result.startswith(b"\\x89PNG\\r\\n\\x1a\\n"):
            return f"data:image/png;base64,{base64.b64encode(result).decode('ascii')}", "Pollinations/kontext"
        if result.startswith(b"\\xff\\xd8\\xff"):
            return f"data:image/jpeg;base64,{base64.b64encode(result).decode('ascii')}", "Pollinations/kontext"
        raise RuntimeError("Pollinations returned an unrecognized edit response.")


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

    try:
        mime = _allowed_image_mime(image_mime)
        image_bytes = _image_data_to_bytes(image_data, mime)
        try:
            data_url, used_model = _hf_edit_image(image_bytes, prompt, mime)
            provider = "Hugging Face"
        except Exception as hf_exc:
            try:
                data_url, used_model = _pollinations_edit_image(image_bytes, prompt, mime)
                provider = "Pollinations"
            except Exception as pollinations_exc:
                raise RuntimeError(
                    f"Hugging Face edit failed: {type(hf_exc).__name__}: {hf_exc} | "
                    f"Pollinations edit failed: {type(pollinations_exc).__name__}: {pollinations_exc}"
                ) from pollinations_exc

        if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
            return {
                "status": "error",
                "answer": "Image editing failed to return the edited image.",
                "error": "Expected a data:image URL from the image editor.",
            }

        return {
            "status": "completed",
            "model": used_model,
            "provider": provider,
            "mime": "image/png",
            "data": data_url,
            "answer": "",
        }
    except Exception as exc:
        return {
            "status": "error",
            "answer": "Image editing failed.",
            "error": f"{type(exc).__name__}: {exc}",
        }
    