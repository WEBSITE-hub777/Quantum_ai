import os
import random
from typing import Any

from groq import Groq
from huggingface_hub import InferenceClient

APP_NAME = "Quantum Queen AI"

GROQ_VAR_NAMES = [
    "GROQ_API_KEY", "GROQ1_API_KEY", "GROQ2_API_KEY", "GROQ3_API_KEY",
    "GROQ4_API_KEY", "GROQ_API_KEYS", "Groq1", "Groq2", "Groq3", "Groq4"
]
GROQ_API_KEYS = list(dict.fromkeys([
    key.strip()
    for var in GROQ_VAR_NAMES
    if (val := os.getenv(var, "").strip())
    for key in val.split(",")
    if key.strip()
]))

GROQ_MODEL = os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.65"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "30"))
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "20000"))

SYSTEM_PROMPT = """
You are Quantum Queen AI, the main intelligence of the Quantum Queen AI application.
Answer directly and clearly. Match the user's language (Hindi, Hinglish, English).
Mathematics questions must be treated as mathematics.
When quantum results are supplied in context, explain them accurately.
""".strip()


def queen_status() -> dict[str, Any]:
    return {
        "status": "ready",
        "app": APP_NAME,
        "groq_keys_loaded": len(GROQ_API_KEYS),
        "primary_engine": "Groq Cloud API (DeepSeek R1 Distill)",
        "model": GROQ_MODEL,
    }


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


def ask_queen_groq(messages: list[dict[str, str]]) -> str:
    if not GROQ_API_KEYS:
        raise RuntimeError("No Groq API Keys configured.")

    keys = list(GROQ_API_KEYS)
    random.shuffle(keys)
    last_exception = None

    for key in keys:
        try:
            client = Groq(api_key=key)
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            last_exception = exc
            continue

    raise RuntimeError(f"All Groq keys failed. Last error: {last_exception}")


def ask_queen_hf(messages: list[dict[str, str]]) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing.")

    client = InferenceClient(provider="auto", token=HF_TOKEN)
    completion = client.chat.completions.create(
        model=HF_MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return completion.choices[0].message.content.strip()


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
    except Exception:
        pass

    try:
        return ask_queen_hf(messages)
    except Exception:
        return "Quantum Queen AI is experiencing high traffic right now. Please try again in a moment."
        