import base64
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any


def generate_id() -> str:

    return uuid.uuid4().hex


def clean_text(
    value: Any,
) -> str:

    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def safe_filename(
    filename: str,
) -> str:

    filename = Path(
        filename
    ).name

    filename = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        filename,
    )

    return filename or "file"


def detect_mime_type(
    filename: str,
) -> str:

    mime, _ = mimetypes.guess_type(
        filename
    )

    return mime or "application/octet-stream"


def encode_base64(
    data: bytes,
) -> str:

    return base64.b64encode(
        data
    ).decode("utf-8")


def decode_base64(
    value: str,
) -> bytes:

    if "," in value and value.startswith(
        "data:"
    ):
        value = value.split(
            ",",
            1,
        )[1]

    return base64.b64decode(
        value
    )


def data_url(
    data: bytes,
    mime_type: str,
) -> str:

    return (
        f"data:{mime_type};base64,"
        f"{encode_base64(data)}"
    )


def truncate(
    text: str,
    limit: int,
) -> str:

    text = str(text)

    if len(text) <= limit:
        return text

    return (
        text[:limit]
        + "\n\n[Content truncated]"
    )


def clean_history(
    history: list[dict[str, Any]],
    limit: int = 30,
) -> list[dict[str, str]]:

    result = []

    for item in history[-limit:]:

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
                "content": content,
            }
        )

    return result


def json_safe(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    return str(value)