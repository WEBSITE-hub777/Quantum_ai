import os
from typing import Any
from huggingface_hub import InferenceClient

APP_NAME = "Quantum Queen AI"

MODEL_NAME = os.getenv(
    "QWEN_MODEL",
    "Qwen/Qwen2.5-72B-Instruct"
)

HF_TOKEN = os.getenv("HF_TOKEN")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
MAX_HISTORY = 20
MAX_MESSAGE_LENGTH = 12000

_client = None

def get_client() -> InferenceClient:
    global _client
    if _client is not None:
        return _client
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing. Add HF_TOKEN to Render Environment Variables."
        )
    _client = InferenceClient(
        provider="auto",
        token=HF_TOKEN,
    )
    return _client

def clean_history(
    history: list[dict[str, Any]] | None
) -> list[dict[str, str]]:
    if not history:
        return []
    cleaned: list[dict[str, str]] = []
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
        cleaned.append(
            {
                "role": role,
                "content": content[:MAX_MESSAGE_LENGTH],
            }
        )
    return cleaned

SYSTEM_PROMPT = """
You are Queen AI, the conversational AI of Quantum Queen AI.

Your job is to:

1. Understand the user's question.
2. Give accurate and useful answers.
3. Explain difficult topics in simple language.
4. Be friendly, clear and natural.
5. Do not invent facts.
6. If you are uncertain, clearly say so.
7. When quantum-computing information is supplied by the
   Quantum Engine, explain that result accurately.
8. Never claim that a quantum simulation solved a problem
   unless the supplied quantum computation actually did so.
9. Keep answers reasonably concise unless the user asks
   for a detailed explanation.
"""

def queen_status() -> dict[str, Any]:
    if not HF_TOKEN:
        return {
            "status": "error",
            "model": MODEL_NAME,
            "provider": "Hugging Face Inference Providers",
            "error": "HF_TOKEN is missing",
        }
    return {
        "status": "ready",
        "model": MODEL_NAME,
        "provider": "auto",
    }

def ask_queen(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
) -> str:
    message = user_message.strip()
    if not message:
        return "Please enter a message."
    if len(message) > MAX_MESSAGE_LENGTH:
        return (
            "Your message is too long. Please send a shorter message."
        )
    try:
        client = get_client()
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
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
            return "Queen AI did not return an answer."
        answer = completion.choices[0].message.content
        if not answer:
            return "Queen AI returned an empty response."
        return answer.strip()
    except Exception as exc:
        print(
            "❌ Queen AI Error:",
            type(exc).__name__,
            str(exc),
        )
        return (
            "Queen AI could not process the request right now. Please try again."
        )

if __name__ == "__main__":
    print("===================================")
    print("👑 Quantum Queen AI")
    print("===================================")
    status = queen_status()
    print("Status:", status)
    if status["status"] == "ready":
        print("\nType 'exit' to stop.\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                print("Queen AI: Goodbye! 👑")
                break
            answer = ask_queen(
                user_message=user_input,
                history=[],
            )
            print("\nQueen AI:", answer)
            print()
