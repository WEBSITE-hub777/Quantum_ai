import base64
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import sympy as sp
from huggingface_hub import InferenceClient

HF_VAR_NAMES = ("HF_TOKEN", "HF1_TOKEN", "HF2_TOKEN", "HF3_TOKEN", "HF4_TOKEN", "HF_TOKENS")
HF_TOKENS = list(dict.fromkeys(token.strip() for name in HF_VAR_NAMES for token in os.getenv(name, "").split(",") if token.strip()))
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell").strip()
HF_IMAGE_FALLBACK_MODEL = os.getenv("HF_IMAGE_FALLBACK_MODEL", "Qwen/Qwen-Image").strip()
HF_IMAGE_EDIT_MODEL = os.getenv("HF_IMAGE_EDIT_MODEL", "Qwen/Qwen-Image-Edit").strip()
HF_IMAGE_EDIT_FALLBACK_MODEL = os.getenv("HF_IMAGE_EDIT_FALLBACK_MODEL", "black-forest-labs/FLUX.1-Kontext-dev").strip()
HF_VISION_MODEL = os.getenv("HF_VISION_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct").strip()
HF_VISION_FALLBACK_MODEL = os.getenv("HF_VISION_FALLBACK_MODEL", "zai-org/GLM-4.5V").strip()

MAX_MATH_LENGTH = 12000
MAX_IMAGE_PROMPT_LENGTH = 8000
MAX_IMAGE_BYTES = 20 * 1024 * 1024
_ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
GEMINI_KEY_NAMES = ("Gemini_1", "Gemini_2", "Gemini_3")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash").strip()
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image").strip()
GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL", "qwen/qwen3.8-27b"
).strip()


def _gemini_keys() -> list[str]:
    return list(dict.fromkeys(os.getenv(name, "").strip() for name in GEMINI_KEY_NAMES if os.getenv(name, "").strip()))


def _gemini_generate(payload: dict[str, Any], model: str) -> tuple[dict[str, Any], str]:
    keys = _gemini_keys()
    if not keys:
        raise RuntimeError("Gemini_1, Gemini_2 and Gemini_3 are not configured in Render.")
    errors = []
    body = json.dumps(payload).encode("utf-8")
    for index, key in enumerate(keys, 1):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model, safe='.-')}:generateContent"
        request = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=150) as response:
                data = json.loads(response.read(25 * 1024 * 1024).decode("utf-8"))
            if not data.get("candidates"):
                raise RuntimeError("Gemini returned no candidates.")
            return data, model
        except Exception as exc:
            errors.append(f"Gemini key {index}: {type(exc).__name__}: {str(exc)[:220]}")
    raise RuntimeError("All configured Gemini keys failed: " + " | ".join(errors[-3:]))


def _parts_text(response: dict[str, Any]) -> str:
    parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "\n".join(str(p.get("text", "")) for p in parts if isinstance(p, dict) and p.get("text")).strip()


def _extract_gemini_image(response: dict[str, Any]) -> tuple[str, str]:
    parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if isinstance(inline, dict) and inline.get("data"):
            mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            encoded = inline["data"]
            base64.b64decode(encoded, validate=True)
            return f"data:{mime};base64,{encoded}", mime
    raise RuntimeError("Gemini did not return image data.")


def _gemini_image_interaction(prompt: str, image_bytes: bytes | None = None, image_mime: str = "image/jpeg") -> tuple[str, str, str]:
    """Call Gemini's image-capable Interactions API and return a browser-ready data URL."""
    keys = _gemini_keys()
    if not keys:
        raise RuntimeError("Gemini_1, Gemini_2 and Gemini_3 are not configured in Render.")

    inputs: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_bytes is not None:
        inputs.append({
            "type": "image",
            "mime_type": image_mime,
            "data": base64.b64encode(image_bytes).decode("ascii"),
        })
    payload = {"model": GEMINI_IMAGE_MODEL, "input": inputs}
    body = json.dumps(payload).encode("utf-8")
    errors: list[str] = []

    for index, key in enumerate(keys, 1):
        request = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read(30 * 1024 * 1024).decode("utf-8"))

            image_parts: list[dict[str, Any]] = []
            for item in result.get("output", []) if isinstance(result.get("output"), list) else []:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_parts.append(item)
            for step in result.get("steps", []) if isinstance(result.get("steps"), list) else []:
                if not isinstance(step, dict):
                    continue
                for item in step.get("content", []) if isinstance(step.get("content"), list) else []:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_parts.append(item)

            direct = result.get("output_image")
            if isinstance(direct, dict):
                image_parts.append({
                    "data": direct.get("data"),
                    "mime_type": direct.get("mime_type") or direct.get("mimeType"),
                    "type": "image",
                })

            for part in image_parts:
                encoded = part.get("data") or part.get("b64_json")
                if not encoded:
                    continue
                # Some wrappers may already return a data URL.
                if isinstance(encoded, str) and encoded.startswith("data:image/"):
                    header, encoded = encoded.split(",", 1)
                    mime_match = re.match(r"data:(image/[a-zA-Z0-9.+-]+);base64", header)
                    output_mime = mime_match.group(1) if mime_match else "image/png"
                else:
                    output_mime = part.get("mime_type") or part.get("mimeType") or "image/png"
                base64.b64decode(encoded, validate=True)
                return f"data:{output_mime};base64,{encoded}", output_mime, GEMINI_IMAGE_MODEL

            detail = result.get("error") or result.get("status") or "no image output returned"
            raise RuntimeError(f"Gemini image API {detail}")
        except Exception as exc:
            errors.append(f"Gemini key {index}: {type(exc).__name__}: {str(exc)[:220]}")
    raise RuntimeError("All Gemini image attempts failed: " + " | ".join(errors[-3:]))


def _hf_image_clients() -> list[InferenceClient]:
    if not HF_TOKENS:
        raise RuntimeError("No HF tokens configured (set HF_TOKEN or HF1_TOKEN).")
    return [InferenceClient(api_key=token, provider="auto", timeout=180) for token in HF_TOKENS]


def _pil_to_data_url(image: Any) -> str:
    if not hasattr(image, "save"):
        raise RuntimeError("Hugging Face returned an invalid image object.")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _hf_generate_image(prompt: str) -> tuple[str, str]:
    errors = []
    for model_name in dict.fromkeys([HF_IMAGE_MODEL, HF_IMAGE_FALLBACK_MODEL]):
        for client in _hf_image_clients():
            try:
                image = client.text_to_image(prompt=prompt, model=model_name)
                return _pil_to_data_url(image), model_name
            except Exception as exc:
                errors.append(f"{model_name}: {type(exc).__name__}: {str(exc)[:200]}")
    raise RuntimeError("All Hugging Face image-generation attempts failed: " + " | ".join(errors[-4:]))


def _pollinations_generate_image(prompt: str) -> tuple[str, str, str]:
    api_key = os.getenv("Quantum1", "").strip()
    if not api_key:
        raise RuntimeError("Pollinations key Quantum1 is not configured in Render.")
    model = os.getenv("POLLINATIONS_IMAGE_MODEL", "flux").strip() or "flux"
    url = f"https://gen.pollinations.ai/image/{urllib.parse.quote(prompt, safe='')}?model={urllib.parse.quote(model, safe='/')}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}", "User-Agent": "Browser-Queen-AI/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        image_bytes = response.read(MAX_IMAGE_BYTES + 1)
        mime = response.headers.get("Content-Type", "image/jpeg").split(";", 1)[0].lower()
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise RuntimeError("Pollinations returned an empty image or an image over 20 MB.")
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii"), mime, f"Pollinations/{model}"


def _hf_edit_image(image_bytes: bytes, prompt: str) -> tuple[str, str]:
    from PIL import Image
    source = Image.open(io.BytesIO(image_bytes))
    errors = []
    for model_name in dict.fromkeys([HF_IMAGE_EDIT_MODEL, HF_IMAGE_EDIT_FALLBACK_MODEL]):
        for client in _hf_image_clients():
            try:
                edited = client.image_to_image(image=source, prompt=prompt, model=model_name)
                return _pil_to_data_url(edited), model_name
            except Exception as exc:
                errors.append(f"{model_name}: {type(exc).__name__}: {str(exc)[:200]}")
    raise RuntimeError("All Hugging Face image-edit attempts failed: " + " | ".join(errors[-4:]))


def _hf_vision(question: str, image_bytes: bytes, image_mime: str) -> tuple[str, str]:
    data_url = f"data:{image_mime};base64," + base64.b64encode(image_bytes).decode("ascii")
    errors = []
    for model_name in dict.fromkeys([HF_VISION_MODEL, HF_VISION_FALLBACK_MODEL]):
        for client in _hf_image_clients():
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]}],
                    max_tokens=2048,
                    temperature=0.2,
                )
                answer = (response.choices[0].message.content or "").strip() if response.choices else ""
                if answer:
                    return answer, model_name
                errors.append(f"{model_name}: empty response")
            except Exception as exc:
                errors.append(f"{model_name}: {type(exc).__name__}: {str(exc)[:180]}")
    raise RuntimeError("Hugging Face vision failed: " + " | ".join(errors[-4:]))


def safe_expression(expression: str) -> str:
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    return re.sub(r"[^0-9a-zA-Z_+\-*/().,\s*]", "", expression).strip()


def solve_equation(question: str) -> dict[str, Any] | None:
    match = re.search(r"([a-zA-Z0-9xX_().+\-*/^ ]+)\s*=\s*([a-zA-Z0-9xX_().+\-*/^ ]+)", question)
    if not match:
        return None
    left, right = safe_expression(match.group(1)), safe_expression(match.group(2))
    try:
        variable = sp.Symbol("x")
        equation = sp.Eq(sp.sympify(left, locals={"x": variable}), sp.sympify(right, locals={"x": variable}))
        solutions = sp.solve(equation, variable)
        if not solutions:
            return None
        text = ", ".join(str(v) for v in solutions)
        return {"method": "symbolic_equation", "equation": str(equation), "solutions": [str(v) for v in solutions], "answer": f"Equation: {equation}\nSolution: x = {text}"}
    except Exception:
        return None


def solve_expression(question: str) -> dict[str, Any] | None:
    match = re.search(r"(?<![A-Za-z])(\d+(?:\.\d+)?(?:\s*[\+\-*/]\s*\d+(?:\.\d+)?)+)(?![A-Za-z])", question)
    if not match:
        return None
    expression = safe_expression(match.group(1))
    try:
        result = sp.simplify(sp.sympify(expression))
        return {"method": "symbolic_arithmetic", "expression": expression, "result": str(result), "answer": f"{expression} = {result}"}
    except Exception:
        return None


def solve_math(question: str) -> dict[str, Any]:
    question = question.strip()
    if not question:
        return {"solved": False, "answer": "", "context": ""}
    if len(question) > MAX_MATH_LENGTH:
        return {"solved": False, "answer": "", "context": "The mathematics request is too long."}
    if result := solve_equation(question):
        return {"solved": True, **result, "context": ""}
    if result := solve_expression(question):
        return {"solved": True, **result, "context": ""}
    return {"solved": False, "answer": "", "context": "This is a mathematics request. Solve the complete problem carefully."}


def _allowed_image_mime(image_mime: str | None) -> str:
    mime = (image_mime or "image/jpeg").strip().lower()
    return mime if mime in _ALLOWED_IMAGE_MIMES else "image/jpeg"


def _image_data_to_bytes(image_data: str, image_mime: str | None) -> tuple[bytes, str]:
    value = image_data.strip()
    mime = _allowed_image_mime(image_mime)
    if value.startswith("data:image/"):
        header, encoded = value.split(",", 1)
        mime_match = re.match(r"data:(image/[a-zA-Z0-9.+-]+);base64", header)
        if mime_match:
            mime = _allowed_image_mime(mime_match.group(1))
        raw = base64.b64decode(encoded, validate=True)
    elif value.startswith("https://"):
        parsed = urllib.parse.urlparse(value)
        if not parsed.hostname:
            raise ValueError("Invalid image URL.")
        request = urllib.request.Request(value, headers={"User-Agent": "Browser-Queen-AI/1.0"})
        with urllib.request.urlopen(request, timeout=25) as response:
            mime = _allowed_image_mime(response.headers.get_content_type())
            raw = response.read(MAX_IMAGE_BYTES + 1)
    else:
        raw = base64.b64decode(value, validate=True)
    if not raw:
        raise ValueError("Image data is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is larger than the 20 MB limit.")
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(raw))
        image.load()
        if max(image.size) > 1600:
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        converted = image.convert("RGB")
        output = io.BytesIO()
        converted.save(output, format="JPEG", quality=88, optimize=True)
        compressed = output.getvalue()
        if len(compressed) < len(raw):
            raw, mime = compressed, "image/jpeg"
    except Exception:
        pass
    return raw, mime


def _gemini_content_parts(prompt: str, image_bytes: bytes | None = None, image_mime: str = "image/jpeg") -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    if image_bytes is not None:
        parts.append({"inline_data": {"mime_type": image_mime, "data": base64.b64encode(image_bytes).decode("ascii")}})
    return parts


def _groq_vision(prompt: str, image_bytes: bytes, image_mime: str) -> tuple[str, str]:
    from groq import Groq
    key_names = ("GROQ_API_KEY", "GROQ1_API_KEY", "GROQ2_API_KEY", "GROQ3_API_KEY", "GROQ4_API_KEY", "GROQ_API_KEYS", "Groq1", "Groq2", "Groq3", "Groq4")
    keys = list(dict.fromkeys(k.strip() for name in key_names for k in os.getenv(name, "").split(",") if k.strip()))
    if not keys:
        raise RuntimeError("No Groq API keys configured.")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{image_mime};base64,{encoded}"
    model = GROQ_VISION_MODEL
    errors = []
    for index, key in enumerate(keys, 1):
        try:
            client = Groq(api_key=key)
            result = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}],
                max_completion_tokens=2048,
                temperature=0.2,
            )
            answer = (result.choices[0].message.content or "").strip() if result.choices else ""
            if not answer:
                raise RuntimeError("Groq vision returned an empty response.")
            return answer, model
        except Exception as exc:
            errors.append(f"Groq key {index}: {type(exc).__name__}: {str(exc)[:180]}")
    raise RuntimeError("Groq vision failed: " + " | ".join(errors[-3:]))


def understand_image(question: str, image_data: str | None, image_mime: str | None = None) -> dict[str, Any]:
    if not image_data:
        return {"status": "error", "answer": "No image was provided."}
    question = question.strip() or "Analyze this image carefully and explain what you can see."
    try:
        image_bytes, mime = _image_data_to_bytes(image_data, image_mime)
    except Exception as exc:
        return {"status": "error", "answer": "Image analysis failed.", "error": f"{type(exc).__name__}: {exc}"}
    try:
        answer, model = _groq_vision(question, image_bytes, mime)
        return {"status": "completed", "provider": "Groq", "model": model, "answer": answer}
    except Exception as groq_exc:
        try:
            answer, model = _hf_vision(question, image_bytes, mime)
            return {"status": "completed", "provider": "Hugging Face", "model": model, "answer": answer}
        except Exception as hf_exc:
            return {"status": "error", "answer": "Image analysis failed.", "error": f"Groq: {type(groq_exc).__name__}: {groq_exc} | Hugging Face: {type(hf_exc).__name__}: {hf_exc}"}

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


def _try_groq_image_operation(operation: str) -> None:
    # Groq's documented API supports text and vision, not image output generation/editing.
    # Do not fake an image call: raise a clear capability error so Gemini can handle it.
    raise RuntimeError(
        f"Groq has no image-{operation.split('-')[-1]} endpoint configured; handing off to Gemini."
    )


def generate_image(prompt: str) -> dict[str, Any]:
    """Generate images with the original image pipeline, independent of chat fallback.
    
    Image generation uses Hugging Face first, then Pollinations. Gemini is reserved
    for text chat fallback; a Gemini text/API quota must not break image generation.
    """
    prompt = _prepare_generation_prompt(prompt)
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {"status": "error", "answer": "Invalid or overly long prompt."}
    errors = []
    if HF_TOKENS:
        try:
            data_url, model = _hf_generate_image(prompt)
            return {
                "status": "completed", "provider": "Hugging Face", "model": model,
                "mime": "image/png", "data": data_url,
                "answer": f"Image generated successfully with {model}.",
            }
        except Exception as exc:
            errors.append(f"Hugging Face: {type(exc).__name__}: {str(exc)[:250]}")
    try:
        data_url, mime, model = _pollinations_generate_image(prompt)
        return {
            "status": "completed", "provider": "Pollinations", "model": model,
            "mime": mime, "data": data_url,
            "answer": f"Image generated successfully with {model}.",
        }
    except Exception as exc:
        errors.append(f"Pollinations: {type(exc).__name__}: {str(exc)[:250]}")
    return {
        "status": "error",
        "answer": "Image generation failed. Check that Hugging Face tokens or the Quantum1 Pollinations key are configured and have quota.",
        "error": " | ".join(errors),
    }


def _prepare_edit_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())


def edit_image(prompt: str, image_data: str | None, image_mime: str | None = None) -> dict[str, Any]:
    prompt = _prepare_edit_prompt(prompt)
    if not image_data:
        return {"status": "error", "answer": "No image was provided for editing."}
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {"status": "error", "answer": "Invalid or overly long edit prompt."}
    try:
        image_bytes, _mime = _image_data_to_bytes(image_data, image_mime)
        data_url, model = _hf_edit_image(image_bytes, prompt)
        return {"status": "completed", "provider": "Hugging Face", "model": model, "mime": "image/png", "data": data_url, "answer": ""}
    except Exception as exc:
        return {"status": "error", "answer": "Image editing failed.", "error": f"{type(exc).__name__}: {exc}"}

