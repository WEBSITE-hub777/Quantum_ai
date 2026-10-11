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
            response, model = _gemini_generate({
                "contents": [{"role": "user", "parts": _gemini_content_parts(question, image_bytes, mime)}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
            }, GEMINI_TEXT_MODEL)
            answer = _parts_text(response)
            if not answer:
                raise RuntimeError("Gemini vision returned an empty response.")
            return {"status": "completed", "provider": "Gemini", "model": model, "answer": answer}
        except Exception as gemini_exc:
            return {"status": "error", "answer": "Image analysis failed.", "error": f"Groq: {type(groq_exc).__name__}: {groq_exc} | Gemini: {type(gemini_exc).__name__}: {gemini_exc}"}


def _prepare_generation_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())


def _try_groq_image_operation(operation: str) -> None:
    # Groq's chat API does not expose image-generation/editing output.
    # Keep the primary-first routing explicit, then hand off to Gemini's image model.
    raise RuntimeError(f"Groq currently has no supported public {operation} endpoint in this integration; use Gemini image fallback.")


def generate_image(prompt: str) -> dict[str, Any]:
    prompt = _prepare_generation_prompt(prompt)
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {"status": "error", "answer": "Invalid or overly long prompt."}
    try:
        _try_groq_image_operation("image-generation")
    except Exception as groq_exc:
        try:
            response, model = _gemini_generate({
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
            }, GEMINI_IMAGE_MODEL)
            data_url, mime = _extract_gemini_image(response)
            return {"status": "completed", "provider": "Gemini", "model": model, "mime": mime, "data": data_url, "answer": "Image generated successfully with Gemini."}
        except Exception as gemini_exc:
            return {"status": "error", "answer": "Image generation failed.", "error": f"Groq: {groq_exc} | Gemini: {type(gemini_exc).__name__}: {gemini_exc}"}


def _prepare_edit_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())


def edit_image(prompt: str, image_data: str | None, image_mime: str | None = None) -> dict[str, Any]:
    prompt = _prepare_edit_prompt(prompt)
    if not image_data:
        return {"status": "error", "answer": "No image was provided for editing."}
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {"status": "error", "answer": "Invalid or overly long edit prompt."}
    try:
        image_bytes, mime = _image_data_to_bytes(image_data, image_mime)
    except Exception as exc:
        return {"status": "error", "answer": "Image editing failed.", "error": f"{type(exc).__name__}: {exc}"}
    try:
        _try_groq_image_operation("image-editing")
    except Exception as groq_exc:
        try:
            response, model = _gemini_generate({
                "contents": [{"role": "user", "parts": _gemini_content_parts(prompt, image_bytes, mime)}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
            }, GEMINI_IMAGE_MODEL)
            data_url, output_mime = _extract_gemini_image(response)
            return {"status": "completed", "provider": "Gemini", "model": model, "mime": output_mime, "data": data_url, "answer": ""}
        except Exception as gemini_exc:
            return {"status": "error", "answer": "Image editing failed.", "error": f"Groq: {groq_exc} | Gemini: {type(gemini_exc).__name__}: {gemini_exc}"}
