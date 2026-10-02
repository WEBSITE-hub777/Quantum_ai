import os
import random
from typing import Any

from groq import Groq
from huggingface_hub import InferenceClient


APP_NAME = "Quantum Queen AI"

GROQ_VAR_NAMES = [
    "GROQ_API_KEY",
    "GROQ1_API_KEY",
    "GROQ2_API_KEY",
    "GROQ3_API_KEY",
    "GROQ4_API_KEY",
    "GROQ_API_KEYS",
    "Groq1",
    "Groq2",
    "Groq3",
    "Groq4",
]

GROQ_API_KEYS = list(dict.fromkeys([
    key.strip()
    for var in GROQ_VAR_NAMES
    if (val := os.getenv(var, "").strip())
    for key in val.split(",")
    if key.strip()
]))

# Groq retired several older model IDs. Keep an explicit safety list so an
# old Render GROQ_MODEL environment variable cannot silently break the app.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEPRECATED_GROQ_MODELS = {
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3.32b",
    "qwen/qwen3.6-27b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
}

_configured_groq_model = os.getenv("GROQ_MODEL", "").strip()
GROQ_MODEL = (
    DEFAULT_GROQ_MODEL
    if not _configured_groq_model
    or _configured_groq_model.lower() in DEPRECATED_GROQ_MODELS
    else _configured_groq_model
)

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_MODEL_NAME = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-ai/DeepSeek-V4.1-Flash",
).strip()

# Final normal-chat fallback after Hugging Face DeepSeek and all Groq keys.
POLLINATIONS_API_KEY = os.getenv("Quantum1", "").strip()
POLLINATIONS_API_URL = "https://gen.pollinations.ai/v1/chat/completions"
POLLINATIONS_CHAT_MODEL = os.getenv(
    "POLLINATIONS_CHAT_MODEL",
    "deepseek/deepseek-v4.1-flash",
).strip()

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.65"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "30"))
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "20000"))

SYSTEM_PROMPT = """
You are Quantum Queen AI, the main intelligence of the Quantum Queen AI application.
Answer directly and clearly. Match the user's language (Hindi, Hinglish, English).
Mathematics questions must be treated as mathematics.
When quantum results are supplied in context, explain them accurately.
The application also has integrated vision, image generation, and image editing tools. These are application capabilities, not limitations of the chat model. Do not tell the user that Quantum Queen AI cannot see, create, or edit images when the application provides those tools. If asked about capabilities, answer for the complete Quantum Queen AI application.

Developer identity:
- If the user asks who owns, developed, created, made, or built Quantum Queen AI, or asks for the developer or owner's name, answer: "mere developer ka nam mrityunjay dubey hai"
- If asked in English, answer naturally in English while preserving the same name.
- Do not invent additional personal details about the developer.
""".strip()


def _short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    if not message:
        message = type(exc).__name__
    return f"{type(exc).__name__}: {message[:350]}"


def queen_status() -> dict[str, Any]:
    return {
        "status": "ready",
        "app": APP_NAME,
        "groq_keys_loaded": len(GROQ_API_KEYS),
        "primary_engine": "Hugging Face DeepSeek",
        "fallback_engine": "Groq1 → Groq2 → Groq3 → Groq4 → Pollinations",
        "model": HF_MODEL_NAME,
        "fallback_model": GROQ_MODEL,
        "pollinations_configured": bool(POLLINATIONS_API_KEY),
        "pollinations_model": POLLINATIONS_CHAT_MODEL,
    }


def clean_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not history:
        return []

    return [
        {
            "role": item["role"],
            "content": item["content"].strip()[:MAX_MESSAGE_LENGTH],
        }
        for item in history[-MAX_HISTORY:]
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]


def ask_queen_groq(messages: list[dict[str, str]]) -> str:
    if not GROQ_API_KEYS:
        raise RuntimeError("No Groq API Keys configured.")

    keys = list(GROQ_API_KEYS)
    random.shuffle(keys)
    errors: list[str] = []

    for index, key in enumerate(keys, start=1):
        try:
            client = Groq(api_key=key)
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_completion_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                include_reasoning=False,
            )

            if not completion.choices:
                raise RuntimeError("Groq returned no choices.")

            answer = (completion.choices[0].message.content or "").strip()
            if not answer:
                raise RuntimeError("Groq returned an empty response.")

            return answer
        except Exception as exc:
            errors.append(f"Groq key {index}: {_short_error(exc)}")

    raise RuntimeError(" | ".join(errors[-4:]))


def ask_queen_hf(messages: list[dict[str, str]]) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing.")

    client = InferenceClient(
        token=HF_TOKEN,
        provider="auto",
        timeout=120,
    )

    completion = client.chat.completions.create(
        model=HF_MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    if not completion.choices:
        raise RuntimeError("Hugging Face returned no choices.")

    answer = (completion.choices[0].message.content or "").strip()
    if not answer:
        raise RuntimeError("Hugging Face returned an empty response.")

    return answer



def ask_queen_pollinations(messages: list[dict[str, str]]) -> str:
    """Final fallback for normal text chat."""
    if not POLLINATIONS_API_KEY:
        raise RuntimeError("Quantum1 environment variable is missing.")

    import json
    import urllib.error
    import urllib.request

    body = json.dumps({
        "model": POLLINATIONS_CHAT_MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }).encode("utf-8")

    request = urllib.request.Request(
        POLLINATIONS_API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Pollinations HTTP {exc.code}: {detail}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("Pollinations returned invalid JSON.") from exc

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"Pollinations returned no choices: {str(payload)[:1000]}")

    answer = (choices[0].get("message", {}).get("content") or "").strip()
    if not answer:
        raise RuntimeError("Pollinations returned an empty response.")

    return answer


def classify_multilingual_intent(user_message: str, has_image: bool = False) -> str:
    """Multilingual semantic fallback using the existing remote AI model."""
    if not isinstance(user_message, str) or not user_message.strip():
        return "NORMAL"
    context = ("An image is attached. Distinguish editing it from asking about it."
               if has_image else
               "No image is attached. IMAGE_GENERATION means creating a new image.")
    system = """
You are a strict multilingual intent router for Quantum Queen AI.
Understand the user's meaning regardless of language or script.
Output exactly ONE label: IMAGE_EDIT, IMAGE_GENERATION, VISION, MATH, QUANTUM, or NORMAL.
IMAGE_EDIT means modifying an attached image: add, remove, change, replace, background,
recolor, enhance, transform, redraw, etc.
IMAGE_GENERATION means creating a NEW image from text when no image is attached.
VISION means inspecting, describing, identifying, reading, or analyzing an attached image.
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
        if HF_TOKEN:
            client = InferenceClient(token=HF_TOKEN, provider="auto", timeout=30)
            completion = client.chat.completions.create(
                model=HF_MODEL_NAME, messages=messages, max_tokens=8, temperature=0.0
            )
            answer = (completion.choices[0].message.content or "").strip().upper()
            for label in labels:
                if label in answer:
                    return label
    except Exception:
        pass
    try:
        if GROQ_API_KEYS:
            keys = list(GROQ_API_KEYS)
            random.shuffle(keys)
            for key in keys:
                try:
                    client = Groq(api_key=key)
                    completion = client.chat.completions.create(
                        model=GROQ_MODEL, messages=messages, max_completion_tokens=8,
                        temperature=0.0, include_reasoning=False
                    )
                    answer = (completion.choices[0].message.content or "").strip().upper()
                    for label in labels:
                        if label in answer:
                            return label
                except Exception:
                    continue
    except Exception:
        pass
    return "VISION" if has_image else "NORMAL"


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

    hf_error = None
    try:
        return ask_queen_hf(messages)
    except Exception as exc:
        hf_error = _short_error(exc)

    groq_error = None
    try:
        return ask_queen_groq(messages)
    except Exception as exc:
        groq_error = _short_error(exc)

    pollinations_error = None
    try:
        return ask_queen_pollinations(messages)
    except Exception as exc:
        pollinations_error = _short_error(exc)

    # Final diagnostic only after every configured provider has failed.
    return (
        "AI service error. "
        f"Hugging Face: {hf_error or 'not available'}. "
        f"Groq: {groq_error or 'not available'}. "
        f"Pollinations: {pollinations_error or 'not available'}."
    )
