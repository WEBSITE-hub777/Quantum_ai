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
        # Image recognition fallback is Google Gemini (not the text router and
        # not Hugging Face). Keep image generation/editing on its own pipeline.
        try:
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": _gemini_content_parts(question, image_bytes, mime),
                }],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
            }
            response, model = _gemini_generate(payload, GEMINI_TEXT_MODEL)
            answer = _parts_text(response)
            if not answer:
                raise RuntimeError("Gemini vision returned an empty response.")
            return {"status": "completed", "provider": "Google Gemini", "model": model, "answer": answer}
        except Exception as gemini_exc:
            return {
                "status": "error",
                "answer": "Image analysis failed.",
                "error": f"Groq vision failed: {type(groq_exc).__name__}: {groq_exc} | Google Gemini vision failed: {type(gemini_exc).__name__}: {gemini_exc}",
            }

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


GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image").strip()
GEMINI_IMAGE_FALLBACK_MODEL = os.getenv("GEMINI_IMAGE_FALLBACK_MODEL", "gemini-3.1-flash-lite-image").strip()

def _groq_image_prompt(prompt: str, image_bytes: bytes | None = None, image_mime: str = "image/jpeg") -> str:
    """Use Groq to refine the image request; Groq does not return generated image pixels."""
    from groq import Groq
    key_names = ("GROQ_API_KEY", "GROQ1_API_KEY", "GROQ2_API_KEY", "GROQ3_API_KEY", "GROQ4_API_KEY",
                 "GROQ_API_KEYS", "Groq1", "Groq2", "Groq3", "Groq4")
    keys = list(dict.fromkeys(k.strip() for name in key_names for k in os.getenv(name, "").split(",") if k.strip()))
    if not keys:
        raise RuntimeError("No Groq API keys configured.")
    model = os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-120b"
    errors = []
    task = (
        "Rewrite the user's request as a concise, vivid image-generation prompt. Preserve the exact subject, "
        "composition, style, colors, and any text requested. Do not answer the user; return only the prompt.\\n"
        f"Request: {prompt}"
    )
    for index, key in enumerate(keys, 1):
        try:
            result = Groq(api_key=key).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": task}],
                max_completion_tokens=700,
                temperature=0.4,
            )
            refined = (result.choices[0].message.content or "").strip() if result.choices else ""
            if refined:
                return refined[:MAX_IMAGE_PROMPT_LENGTH]
            errors.append(f"Groq key {index}: empty prompt response")
        except Exception as exc:
            errors.append(f"Groq key {index}: {type(exc).__name__}: {str(exc)[:160]}")
    raise RuntimeError("Groq image-prompt step failed: " + " | ".join(errors[-3:]))

def _gemini_generate_image(prompt: str, image_bytes: bytes | None = None, image_mime: str = "image/jpeg") -> tuple[str, str]:
    """Render or edit pixels with Gemini; rotate configured keys and supported image models."""
    keys = _gemini_keys()
    if not keys:
        raise RuntimeError("Gemini_1, Gemini_2 and Gemini_3 are not configured in Render.")

    models = list(dict.fromkeys(m for m in (
        GEMINI_IMAGE_MODEL,
        GEMINI_IMAGE_FALLBACK_MODEL,
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
        "gemini-2.5-flash-image",
    ) if m))
    encoded_input = base64.b64encode(image_bytes).decode("ascii") if image_bytes is not None else None
    errors: list[str] = []

    # Try the image-oriented Interactions API first, then the generateContent
    # route. Groq's chat API can refine a prompt, but Gemini must render pixels.
    for model in models:
        items: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if encoded_input:
            items.append({"type": "image", "mime_type": image_mime, "data": encoded_input})
        interaction_body = json.dumps({
            "model": model,
            "input": items,
            "response_format": {
                "type": "image",
                "mime_type": "image/png",
                "aspect_ratio": "1:1",
            },
        }).encode("utf-8")

        for index, key in enumerate(keys, 1):
            request = urllib.request.Request(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                data=interaction_body,
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    data = json.loads(response.read(30 * 1024 * 1024).decode("utf-8"))
                output_image = data.get("output_image") or {}
                encoded = output_image.get("data")
                if not encoded:
                    for step in data.get("steps", []):
                        blocks = step.get("content", []) if isinstance(step, dict) else []
                        for block in blocks:
                            if isinstance(block, dict) and block.get("type") == "image" and block.get("data"):
                                encoded, output_image = block["data"], block
                                break
                        if encoded:
                            break
                if not encoded:
                    raise RuntimeError("Gemini Interactions returned no image data.")
                base64.b64decode(encoded, validate=True)
                mime = output_image.get("mime_type") or output_image.get("mimeType") or "image/png"
                return f"data:{mime};base64,{encoded}", mime
            except Exception as exc:
                errors.append(f"{model}/interactions key {index}: {type(exc).__name__}: {str(exc)[:140]}")

        parts: list[dict[str, Any]] = [{"text": prompt}]
        if encoded_input:
            parts.append({"inline_data": {"mime_type": image_mime, "data": encoded_input}})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(model, safe='.-')}:generateContent"
        )
        body = json.dumps(payload).encode("utf-8")
        for index, key in enumerate(keys, 1):
            request = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    data = json.loads(response.read(30 * 1024 * 1024).decode("utf-8"))
                for candidate in data.get("candidates") or []:
                    for part in candidate.get("content", {}).get("parts", []):
                        inline = part.get("inline_data") or part.get("inlineData")
                        if isinstance(inline, dict) and inline.get("data"):
                            encoded = inline["data"]
                            base64.b64decode(encoded, validate=True)
                            mime = inline.get("mime_type") or inline.get("mimeType") or "image/png"
                            return f"data:{mime};base64,{encoded}", mime
                raise RuntimeError("Gemini generateContent returned no image data.")
            except Exception as exc:
                errors.append(f"{model}/generateContent key {index}: {type(exc).__name__}: {str(exc)[:140]}")

    if errors and all("429" in error or "Too Many Requests" in error for error in errors):
        raise RuntimeError(
            "Every configured Gemini image attempt hit HTTP 429 (quota/rate limit). "
            "The code tried all configured keys, models, and both image API routes. "
            "Wait for quota reset or enable image-model quota in Google AI Studio."
        )
    raise RuntimeError("Gemini image routes/models failed: " + " | ".join(errors[-12:]))

def _gemini_image_with_groq_first(prompt: str, image_bytes: bytes | None = None, image_mime: str = "image/jpeg") -> tuple[str, str, str]:
    """Groq-first prompt preparation followed by Gemini's actual image generation/editing."""
    groq_note = ""
    try:
        image_prompt = _groq_image_prompt(prompt, image_bytes, image_mime)
        groq_note = "Groq prompt preparation succeeded. "
    except Exception as exc:
        image_prompt = prompt
        groq_note = f"Groq prompt preparation failed ({type(exc).__name__}); using the original prompt. "
    data_url, mime = _gemini_generate_image(image_prompt, image_bytes, image_mime)
    return data_url, mime, f"{GEMINI_IMAGE_MODEL} (Groq-first prompt pipeline)"

def generate_image(prompt: str) -> dict[str, Any]:
    """Generate an image: Groq prepares the prompt; Gemini returns the actual image."""
    prompt = _prepare_generation_prompt(prompt)
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {"status": "error", "answer": "Invalid or overly long prompt."}
    groq_error = ""
    try:
        try:
            image_prompt = _groq_image_prompt(prompt)
        except Exception as exc:
            image_prompt = prompt
            groq_error = f"Groq prompt step: {type(exc).__name__}: {str(exc)[:180]}. "
        data_url, mime = _gemini_generate_image(image_prompt)
        return {
            "status": "completed", "provider": "Google Gemini", "model": GEMINI_IMAGE_MODEL,
            "mime": mime, "data": data_url,
            "answer": "Image generated successfully with Gemini.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "answer": "Image generation failed. Groq can prepare/refine the prompt, but it does not provide image pixels through this API; Gemini must render the image. If the technical detail shows HTTP 429, Gemini's image quota/rate limit is exhausted or unavailable for these keys/models. Check Google AI Studio quota/billing or wait for the limit to reset.",
            "error": groq_error + f"Gemini: {type(exc).__name__}: {str(exc)[:500]}",
        }

def _prepare_edit_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())

def edit_image(prompt: str, image_data: str | None, image_mime: str | None = None) -> dict[str, Any]:
    """Edit an uploaded image using Gemini; Groq prepares the edit prompt first."""
    prompt = _prepare_edit_prompt(prompt)
    if not image_data:
        return {"status": "error", "answer": "No image was provided for editing."}
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        return {"status": "error", "answer": "Invalid or overly long edit prompt."}
    try:
        image_bytes, mime = _image_data_to_bytes(image_data, image_mime)
    except Exception as exc:
        return {"status": "error", "answer": "Image editing failed.", "error": f"{type(exc).__name__}: {exc}"}
    groq_error = ""
    try:
        try:
            image_prompt = _groq_image_prompt(prompt, image_bytes, mime)
        except Exception as exc:
            image_prompt = prompt
            groq_error = f"Groq prompt step: {type(exc).__name__}: {str(exc)[:180]}. "
        data_url, out_mime = _gemini_generate_image(image_prompt, image_bytes, mime)
        return {
            "status": "completed", "provider": "Google Gemini", "model": GEMINI_IMAGE_MODEL,
            "mime": out_mime, "data": data_url, "answer": "",
        }
    except Exception as exc:
        return {
            "status": "error", "answer": "Image editing failed. Groq can refine the edit prompt, but Gemini must render the edited image. If the technical detail shows HTTP 429, check Gemini image quota/rate limits in Google AI Studio or wait for the limit to reset.",
            "error": groq_error + f"Gemini: {type(exc).__name__}: {str(exc)[:500]}",
        }

