import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from groq import Groq

APP_NAME = "Browser Queen AI"
GROQ_VAR_NAMES = (
    "GROQ_API_KEY", "GROQ1_API_KEY", "GROQ2_API_KEY", "GROQ3_API_KEY",
    "GROQ4_API_KEY", "GROQ_API_KEYS", "Groq1", "Groq2", "Groq3", "Groq4",
)
GROQ_API_KEYS = list(dict.fromkeys(
    key.strip()
    for name in GROQ_VAR_NAMES
    for key in os.getenv(name, "").split(",")
    if key.strip()
))
_configured_groq_model = os.getenv("GROQ_MODEL", "").strip()
_DEPRECATED_GROQ_MODELS = {
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3.6-27b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
}
DEFAULT_GROQ_MODEL = (
    "openai/gpt-oss-120b"
    if not _configured_groq_model or _configured_groq_model.lower() in _DEPRECATED_GROQ_MODELS
    else _configured_groq_model
)
GEMINI_KEY_NAMES = ("Gemini_1", "Gemini_2", "Gemini_3")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash").strip()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.65"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "30"))
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "20000"))

SYSTEM_PROMPT = """
You are Browser Queen AI, a helpful browser-aware assistant.
Answer clearly and match the user's language, including Hindi and Hinglish.
You can answer questions, help with coding, analyze images when an image tool is used,
and help users navigate websites. Do not claim that you have clicked, submitted, paid,
booked, or completed a website action unless the application actually performed it.
The browser panel is for displaying/navigating pages; the user remains responsible for
logins, OTP/CAPTCHA, payments, and final booking/submission confirmation.
Never ask the user to reveal API keys or passwords in chat.
Developer identity: if asked who developed the app, answer that it was developed by Mrityunjay Dubey.
""".strip()


def _short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip() or type(exc).__name__
    return f"{type(exc).__name__}: {message[:300]}"


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


def _gemini_keys() -> list[str]:
    return list(dict.fromkeys(
        os.getenv(name, "").strip()
        for name in GEMINI_KEY_NAMES
        if os.getenv(name, "").strip()
    ))


def _ask_gemini(messages: list[dict[str, str]]) -> str:
    keys = _gemini_keys()
    if not keys:
        raise RuntimeError("Gemini_1, Gemini_2 and Gemini_3 are missing in Render Environment.")
    contents = []
    system_text = ""
    for item in messages:
        role = item.get("role", "user")
        if role == "system":
            system_text += item.get("content", "") + "\n"
            continue
        contents.append({
            "role": "model" if role == "assistant" else "user",
            "parts": [{"text": item.get("content", "")}],
        })
    payload: dict[str, Any] = {"contents": contents, "generationConfig": {
        "temperature": TEMPERATURE, "maxOutputTokens": MAX_TOKENS,
    }}
    if system_text.strip():
        payload["systemInstruction"] = {"parts": [{"text": system_text.strip()}]}
    body = json.dumps(payload).encode("utf-8")
    errors = []
    for index, key in enumerate(keys, 1):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(GEMINI_TEXT_MODEL, safe='.-')}:generateContent"
        request = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read(8 * 1024 * 1024).decode("utf-8"))
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError("Gemini returned no candidates.")
            parts = candidates[0].get("content", {}).get("parts", [])
            answer = "\n".join(str(part.get("text", "")) for part in parts if part.get("text")).strip()
            if not answer:
                raise RuntimeError("Gemini returned an empty text response.")
            return answer
        except Exception as exc:
            errors.append(f"Gemini key {index}: {_short_error(exc)}")
    raise RuntimeError("Gemini fallback failed: " + " | ".join(errors[-3:]))


def ask_queen_groq(messages: list[dict[str, str]]) -> str:
    if not GROQ_API_KEYS:
        raise RuntimeError("No Groq API keys configured.")
    errors = []
    for index, key in enumerate(GROQ_API_KEYS, 1):
        try:
            client = Groq(api_key=key)
            completion = client.chat.completions.create(
                model=DEFAULT_GROQ_MODEL,
                messages=messages,
                max_completion_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                include_reasoning=False,
            )
            answer = (completion.choices[0].message.content or "").strip() if completion.choices else ""
            if not answer:
                raise RuntimeError("Groq returned an empty response.")
            return answer
        except Exception as exc:
            errors.append(f"Groq key {index}: {_short_error(exc)}")
    raise RuntimeError("All Groq attempts failed: " + " | ".join(errors[-4:]))


def ask_queen(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    system_context: str | None = None,
) -> str:
    if not isinstance(user_message, str) or not user_message.strip():
        return "Please enter a valid text message."
    sys_prompt = SYSTEM_PROMPT
    if system_context:
        sys_prompt += f"\n\nSPECIALIZED CONTEXT:\n{system_context.strip()}"
    messages = [{"role": "system", "content": sys_prompt}]
    messages.extend(clean_history(history))
    messages.append({"role": "user", "content": user_message.strip()})
    try:
        return ask_queen_groq(messages)
    except Exception as groq_exc:
        try:
            return _ask_gemini(messages)
        except Exception as gemini_exc:
            return (
                "AI service error. Groq failed: "
                f"{_short_error(groq_exc)}. Gemini fallback failed: {_short_error(gemini_exc)}"
            )


def queen_status() -> dict[str, Any]:
    return {
        "status": "ready",
        "app": APP_NAME,
        "groq_keys_loaded": len(GROQ_API_KEYS),
        "gemini_keys_loaded": len(_gemini_keys()),
        "primary_engine": "Groq",
        "fallback_engine": "Gemini_1 → Gemini_2 → Gemini_3",
        "text_model": DEFAULT_GROQ_MODEL,
        "gemini_text_model": GEMINI_TEXT_MODEL,
    }


def classify_multilingual_intent(user_message: str, has_image: bool = False) -> str:
    if not isinstance(user_message, str) or not user_message.strip():
        return "NORMAL"
    context = (
        "An image is attached. Distinguish editing it from asking about it."
        if has_image else
        "No image is attached. IMAGE_GENERATION means creating a new image from text."
    )
    system = """
You are a strict multilingual intent router for Browser Queen AI.
Understand the user's meaning regardless of language or script.
Output exactly ONE label: IMAGE_EDIT, IMAGE_GENERATION, VISION, MATH, QUANTUM, or NORMAL.
IMAGE_EDIT means modifying an attached image.
IMAGE_GENERATION means creating a new image from text when no image is attached.
VISION means inspecting, describing, reading, or analyzing an attached image.
MATH means mathematics. QUANTUM means quantum computing/science. NORMAL is everything else.
If an image is attached and the user wants to change it, choose IMAGE_EDIT.
If an image is attached and the user asks about its contents, choose VISION.
Output only the label.
""".strip()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{context}\nUser text: {user_message.strip()}"},
    ]
    labels = ("IMAGE_EDIT", "IMAGE_GENERATION", "VISION", "MATH", "QUANTUM", "NORMAL")
    try:
        answer = ask_queen_groq(messages).upper()
        for label in labels:
            if label in answer:
                return label
    except Exception:
        pass
    try:
        answer = _ask_gemini(messages).upper()
        for label in labels:
            if label in answer:
                return label
    except Exception:
        pass
    return "VISION" if has_image else "NORMAL"
