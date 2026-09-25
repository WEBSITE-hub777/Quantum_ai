import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from pathlib import Path
import uuid

from router import process_request
from queen import queen_status
from quantum import quantum_status, run_quantum


APP_NAME = "Quantum Queen AI"
APP_VERSION = "4.0.0"

MAX_MESSAGE_LENGTH = 20000
MAX_IMAGE_SIZE = 10 * 1024 * 1024

raw_origins = os.getenv("BITTOOL_ORIGINS", "*").strip()

origins = (
    ["*"]
    if raw_origins == "*"
    else [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip()
    ]
)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Advanced hybrid AI backend.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
    )
    history: list[dict[str, Any]] = Field(
        default_factory=list
    )
    image_data: str | None = None
    image_mime: str | None = None


class QuantumRequest(BaseModel):
    qubits: int = Field(
        default=20,
        ge=1,
        le=30,
    )
    shots: int = Field(
        default=1,
        ge=1,
        le=10000,
    )


@app.get("/")
async def root():
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "features": {
            "chat": True,
            "math": True,
            "quantum": True,
            "vision": True,
            "image_generation": True,
            "file_generation": False,
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "queen_ai": queen_status(),
        "quantum_engine": quantum_status(),
    }


@app.get("/status")
async def status():
    return {
        "app": {
            "name": APP_NAME,
            "version": APP_VERSION,
        },
        "queen_ai": queen_status(),
        "quantum_engine": quantum_status(),
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        return process_request(
            user_message=request.message,
            history=request.history,
            image_data=request.image_data,
            image_mime=request.image_mime,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AI request failed: {type(exc).__name__}",
        ) from exc


@app.post("/quantum")
async def quantum(request: QuantumRequest):
    try:
        result = run_quantum(
            num_qubits=request.qubits,
            shots=request.shots,
        )

        return {
            "type": "quantum",
            "status": "completed",
            "result": result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Quantum request failed: {type(exc).__name__}",
        ) from exc


MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "media"))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


@app.post("/upload-image")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Use JPG, PNG, WebP or GIF.",
        )

    data = await file.read()

    if len(data) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB.",
        )

    if not data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    filename = f"{uuid.uuid4().hex}{ALLOWED_IMAGE_TYPES[file.content_type]}"
    target = MEDIA_DIR / filename
    target.write_bytes(data)

    public_base = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if not public_base:
        public_base = str(request.base_url).rstrip("/")

    return {
        "status": "uploaded",
        "filename": file.filename or filename,
        "mime": file.content_type,
        "size": len(data),
        "url": f"{public_base}/media/{filename}",
    }


@app.get("/media/{filename}")
async def get_media(filename: str):
    safe_name = Path(filename).name
    target = MEDIA_DIR / safe_name

    if not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(target)
