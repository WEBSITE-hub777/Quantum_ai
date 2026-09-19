import os
from typing import Any

from huggingface_hub import InferenceClient


# =========================================================
# 👑 QUANTUM QUEEN AI
# Remote Hugging Face Inference Version
# =========================================================

APP_NAME = "Quantum Queen AI"

# Hugging Face model
MODEL_NAME = os.getenv(
    "QWEN_MODEL",
    "Qwen/Qwen2.5-72B-Instruct"
)

# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

# Generation settings
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# Maximum history messages used
MAX_HISTORY = 20

# Maximum characters per message
MAX_MESSAGE_LENGTH = 12000


# =========================================================
# 🤗 HUGGING FACE CLIENT
# =========================================================

_client = None


def get_client() -> InferenceClient:
    """
    Create the Hugging Face client only when needed.
    The large Qwen model is NOT downloaded into Render.
    """

    global _client

    if _client is not None:
        return _client

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing. Add HF_TOKEN "
            "to Render Environment Variables."
        )

    _client = InferenceClient(
        provider="auto",
        token=HF_TOKEN,
    )

    return _client


# =========================================================
# 🧹 HISTORY CLEANER
# =========================================================

def clean_history(
    history: list[dict[str, Any]] | None
) -> list[dict[str, str]]:
    """
    Clean conversation history before sending it to Qwen.
    """

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


# =========================================================
# 🧠 SYSTEM PERSONALITY
# =========================================================

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


# =========================================================
# 📊 STATUS
# =========================================================

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


# =========================================================
# 👑 ASK QUEEN
# =========================================================

def ask_queen(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """
    Send a conversation to the remote Qwen model.

    The model itself is NOT loaded into Render RAM.
    Render sends the request to Hugging Face inference.
    """

    message = user_message.strip()

    if not message:
        return "Please enter a message."

    if len(message) > MAX_MESSAGE_LENGTH:
        return (
            "Your message is too long. "
            "Please send a shorter message."
        )

    try:

        client = get_client()

        # ---------------------------------------------
        # Conversation messages
        # ---------------------------------------------

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
            }
        ]

        # Add previous conversation
        messages.extend(
            clean_history(history)
        )

        # Add current user message
        messages.append(
            {
                "role": "user",
                "content": message,
            }
        )

        # ---------------------------------------------
        # Call Qwen through Hugging Face
        # ---------------------------------------------

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        # ---------------------------------------------
        # Extract answer safely
        # ---------------------------------------------

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
            "Queen AI could not process the request right now. "
            "Please try again."
        )


# =========================================================
# 🧪 SIMPLE LOCAL TEST
# =========================================================

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

"requirements.txt" में यह भी होना चाहिए

fastapi
uvicorn[standard]
pydantic
qiskit
qiskit-aer
huggingface_hub

इस version में "torch" और "transformers" की जरूरत "queen.py" के लिए नहीं है, क्योंकि Render खुद Qwen model load नहीं कर रहा; "InferenceClient" remote inference service को request भेज रहा है। Hugging Face की documentation इसी "InferenceClient" approach को server-side inference providers के लिए बताती है।

और एक महत्वपूर्ण बात: "HF_TOKEN" को "queen.py" में लिखना नहीं है। Render → Environment Variables में रखना है। इससे तुम्हारी secret key GitHub code में नहीं जाएगी।