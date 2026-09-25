import os
from typing import Any

from huggingface_hub import InferenceClient


APP_NAME = "Quantum Queen AI"

MODEL_NAME = os.getenv(
    "QWEN_MODEL",
    "Qwen/Qwen2.5-72B-Instruct",
)

HF_TOKEN = os.getenv("HF_TOKEN")

MAX_TOKENS = int(
    os.getenv(
        "MAX_TOKENS",
        "4096",
    )
)

TEMPERATURE = float(
    os.getenv(
        "TEMPERATURE",
        "0.65",
    )
)

TOP_P = float(
    os.getenv(
        "TOP_P",
        "0.9",
    )
)

MAX_HISTORY = 30
MAX_MESSAGE_LENGTH = 20000

_client: InferenceClient | None = None


SYSTEM_PROMPT = """
You are Quantum Queen AI, a general-purpose AI assistant.

Core behavior:
- Understand the user's actual request before answering.
- Answer the question directly.
- Do not confuse difficulty with quantum relevance.
- Mathematics questions must be answered as mathematics.
- Quantum computation must only be discussed when quantum computation is relevant.
- Never insert random quantum measurement states into normal answers.
- A binary measurement is not automatically a mathematical solution.
- Never claim that Qiskit Aer is an IBM Quantum hardware device.
- Do not invent facts.
- If information is uncertain, say so.
- Answer in the user's language whenever practical.

Code formatting:
- When the user asks for code, put the COMPLETE code in a single fenced code block.
- Never put part of a requested code file outside its code block.
- Do not split one requested file into multiple code blocks.
- Keep explanations outside the code block.
- Preserve the requested programming language.
- If multiple files are explicitly requested, use one complete code block per file.

Long answers:
- Finish the requested answer.
- Do not intentionally truncate.
- Prefer complete useful output over unnecessary filler.
- Use headings and lists when they improve readability.
""".strip()


def get_client() -> InferenceClient:

    global _client

    if _client is not None:
        return _client

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing."
        )

    _client = InferenceClient(
    provider=os.getenv("QWEN_PROVIDER", "auto"),
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

        if role not in {
            "user",
            "assistant",
        }:
            continue

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        result.append(
            {
                "role": role,
                "content": content[
                    :MAX_MESSAGE_LENGTH
                ],
            }
        )

    return result


def queen_status() -> dict[str, Any]:

    if not HF_TOKEN:
        return {
            "status": "error",
            "model": MODEL_NAME,
            "provider": "Hugging Face",
            "error": "HF_TOKEN is missing",
        }

    return {
        "status": "ready",
        "model": MODEL_NAME,
        "provider": "Hugging Face Inference Providers",
        "max_tokens": MAX_TOKENS,
    }


def ask_queen(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    system_context: str | None = None,
) -> str:

    message = user_message.strip()

    if not message:
        return "Please enter a message."

    if len(message) > MAX_MESSAGE_LENGTH:
        return (
            "Your message is too long. "
            "Please send a shorter message."
        )

    client = get_client()

    system_prompt = SYSTEM_PROMPT

    if system_context:
        system_prompt += (
            "\n\nSPECIALIZED CONTEXT:\n"
            + system_context[:30000]
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    messages.extend(
        clean_history(history)
    )

    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    if not completion.choices:
        raise RuntimeError(
            "The AI returned no choices."
        )

    content = (
        completion
        .choices[0]
        .message
        .content
    )

    if not content:
        raise RuntimeError(
            "The AI returned an empty response."
        )

    return content.strip()
