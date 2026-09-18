import os
import threading
from typing import Any

import torch

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM
)


# ==========================================
# MODEL CONFIG
# ==========================================

MODEL_NAME = os.getenv(
    "QWEN_MODEL",
    "Qwen/Qwen3.8-27B"
)

MAX_NEW_TOKENS = int(
    os.getenv(
        "MAX_NEW_TOKENS",
        "300"
    )
)

TEMPERATURE = float(
    os.getenv(
        "TEMPERATURE",
        "0.7"
    )
)

TOP_P = float(
    os.getenv(
        "TOP_P",
        "0.9"
    )
)


# ==========================================
# GLOBAL MODEL OBJECTS
# ==========================================

_processor = None

_model = None

_load_error = None

_model_lock = threading.Lock()


# ==========================================
# LOAD QUEEN AI
# ==========================================

def _load_model():

    global _processor
    global _model
    global _load_error


    # Already loaded
    if (
        _processor is not None
        and _model is not None
    ):

        return (
            _processor,
            _model
        )


    # Prevent two requests loading
    # the model simultaneously
    with _model_lock:

        if (
            _processor is not None
            and _model is not None
        ):

            return (
                _processor,
                _model
            )


        try:

            print(
                f"👑 Loading Queen AI: {MODEL_NAME}"
            )


            # Processor
            _processor = (
                AutoProcessor.from_pretrained(
                    MODEL_NAME
                )
            )


            # Model
            _model = (
                AutoModelForMultimodalLM
                .from_pretrained(
                    MODEL_NAME,
                    device_map="auto",
                    torch_dtype="auto"
                )
            )


            _load_error = None


            print(
                "✅ Queen AI loaded."
            )


        except Exception as exc:

            _load_error = (
                f"{type(exc).__name__}: {exc}"
            )

            _processor = None
            _model = None

            print(
                "❌ Queen AI loading failed:"
            )

            print(
                _load_error
            )

            raise


    return (
        _processor,
        _model
    )


# ==========================================
# STATUS
# ==========================================

def queen_status() -> dict[str, Any]:

    if (
        _model is not None
        and _processor is not None
    ):

        return {
            "status": "ready",
            "model": MODEL_NAME
        }


    if _load_error:

        return {
            "status": "error",
            "model": MODEL_NAME,
            "error": _load_error[:500]
        }


    return {
        "status": "not_loaded",
        "model": MODEL_NAME
    }


# ==========================================
# BUILD MESSAGES
# ==========================================

def _build_messages(
    user_message: str,
    history: list[dict[str, Any]]
):

    messages = []


    # Previous conversation
    for item in history[-20:]:

        role = item.get("role")

        content = item.get("content")


        if role not in {
            "user",
            "assistant"
        }:

            continue


        if not isinstance(
            content,
            str
        ):

            continue


        if not content.strip():

            continue


        messages.append({

            "role": role,

            "content": [
                {
                    "type": "text",
                    "text": content[:12000]
                }
            ]

        })


    # Current question
    messages.append({

        "role": "user",

        "content": [
            {
                "type": "text",
                "text": user_message
            }
        ]

    })


    return messages


# ==========================================
# ASK QUEEN
# ==========================================

def ask_queen(
    user_message: str,
    history: list[dict[str, Any]] | None = None
) -> str:

    history = history or []


    # Load only when first request arrives
    processor, model = _load_model()


    messages = _build_messages(
        user_message,
        history
    )


    # ======================================
    # TOKENIZE
    # ======================================

    inputs = (
        processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
    )


    # ======================================
    # MOVE INPUT TO MODEL DEVICE
    # ======================================

    inputs = {

        key: (
            value.to(model.device)
            if hasattr(value, "to")
            else value
        )

        for key, value in inputs.items()
    }


    # ======================================
    # GENERATE
    # ======================================

    with torch.inference_mode():

        outputs = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=True,

            temperature=TEMPERATURE,

            top_p=TOP_P
        )


    # ======================================
    # REMOVE INPUT TOKENS
    # ======================================

    input_length = (
        inputs["input_ids"]
        .shape[-1]
    )


    answer_tokens = (
        outputs[0][input_length:]
    )


    # ======================================
    # DECODE
    # ======================================

    answer = (
        processor.decode(
            answer_tokens,
            skip_special_tokens=True
        )
    )


    return answer.strip()