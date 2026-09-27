import os
import time
from typing import Any

from huggingface_hub import InferenceClient


APP_NAME = "Quantum Queen AI"

MODEL_NAME = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-ai/DeepSeek-V4.1-Flash",
)

HF_TOKEN = os.getenv("HF_TOKEN")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.65"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "30"))
MAX_MESSAGE_LENGTH = int(
    os.getenv("MAX_MESSAGE_LENGTH", "20000")
)
MAX_CONTEXT_LENGTH = int(
    os.getenv("MAX_CONTEXT_LENGTH", "30000")
)

_client: InferenceClient | None = None


SYSTEM_PROMPT = """
You are Quantum Queen AI, the main intelligence of the Quantum Queen AI application.

You are a helpful, intelligent, accurate and practical general-purpose AI assistant.

Answer directly and clearly.
Match the user's language whenever practical.
If the user writes Hindi or Hinglish, respond naturally in Hindi/Hinglish.
If the user writes English, respond in English.
Do not unnecessarily repeat the user's question.

Never intentionally invent facts, sources, measurements, results, APIs, features, files or actions.
If something is uncertain, clearly say so.
Do not pretend that you performed an action that the application did not actually perform.

Mathematics questions must be treated as mathematics.
Show useful steps when appropriate.
Do not randomly introduce quantum computing into ordinary mathematics.

Quantum features are handled by the application's Qiskit engine.
When quantum results are supplied in the context, explain them accurately.
Never fabricate quantum measurements.
Never claim that Qiskit AerSimulator is physical quantum hardware.
Clearly distinguish quantum simulation from physical quantum hardware.

If image-analysis information is supplied by the application, use it carefully.
Do not pretend to see an image when image information has not been supplied.

Image generation is handled by the application's image-generation engine.
Do not claim an image was generated unless the application actually provides a generation result.

When the user requests code, provide complete working code whenever possible.
Put a requested complete file in one code block.
Preserve existing interfaces and unrelated working functionality.

SPECIALIZED CONTEXT may contain results from Quantum Queen's math, quantum, vision, image or file engines.
Treat that context as application-provided information.
Do not invent information that is missing from it.

Do not provide dangerous or illegal instructions.
Do not help bypass authentication, steal credentials, compromise systems or access accounts without permission.

Prefer useful answers over filler.
Finish the requested task.
""".strip()


def get_client() -> InferenceClient:
    global _client

    if _client is not None:
        return _client

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing. Add HF_TOKEN to the Render environment variables."
        )

    _client = InferenceClient(
        provider="auto",
        token=HF_TOKEN,
    )

    return _client


def clean_history(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:

    if not history:
        return []

    result = []

    for item in history[-MAX_HISTORY:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        result.append(
            {
                "role": role,
                "content": content[:MAX_MESSAGE_LENGTH],
            }
        )

    return result


def queen_status() -> dict[str, Any]:

    if not HF_TOKEN:
        return {
            "status": "error",
            "app": APP_NAME,
            "model": MODEL_NAME,
            "provider": "Hugging Face Inference Providers",
            "error": "HF_TOKEN is missing",
        }

    return {
        "status": "ready",
        "app": APP_NAME,
        "model": MODEL_NAME,
        "provider": "Hugging Face Inference Providers",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_history": MAX_HISTORY,
    }


def extract_content(completion: Any) -> str:

    if completion is None:
        raise RuntimeError("The AI returned no response.")

    choices = getattr(completion, "choices", None)

    if not choices:
        raise RuntimeError("The AI returned no choices.")

    message = getattr(choices[0], "message", None)

    if message is None:
        raise RuntimeError(
            "The AI response did not contain a message."
        )

    content = getattr(message, "content", None)

    if content is None:
        raise RuntimeError("The AI returned empty content.")

    if not isinstance(content, str):
        content = str(content)

    content = content.strip()

    if not content:
        raise RuntimeError("The AI returned an empty response.")

    return content


def build_messages(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    system_context: str | None = None,
) -> list[dict[str, str]]:

    system_prompt = SYSTEM_PROMPT

    if system_context:
        context = str(system_context).strip()

        if context:
            system_prompt += (
                "\n\nSPECIALIZED CONTEXT FROM QUANTUM QUEEN:\n"
                + context[:MAX_CONTEXT_LENGTH]
            )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    messages.extend(clean_history(history))

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    return messages


def ask_queen(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    system_context: str | None = None,
) -> str:

    if not isinstance(user_message, str):
        return "Please send your message as text."

    message = user_message.strip()

    if not message:
        return "Please enter a message."

    if len(message) > MAX_MESSAGE_LENGTH:
        return (
            "Your message is too long. "
            "Please send a shorter message."
        )

    client = get_client()

    messages = build_messages(
        user_message=message,
        history=history,
        system_context=system_context,
    )

    started_at = time.time()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

    except Exception as exc:
        elapsed = round(
            time.time() - started_at,
            2,
        )

        raise RuntimeError(
            f"Quantum Queen AI request failed after {elapsed}s: {exc}"
        ) from exc

    return extract_content(completion)