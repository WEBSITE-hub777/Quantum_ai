import os
from pathlib import Path
from typing import Any
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from quantum import quantum_status, run_quantum
from queen import queen_status
from router import process_request

APP_NAME = "Quantum Queen AI"
APP_VERSION = "4.0.0"
MAX_MESSAGE_LENGTH = 20000
MAX_IMAGE_SIZE = 20 * 1024 * 1024

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    history: list[dict[str, Any]] = Field(default_factory=list)
    image_data: str | None = None
    image_mime: str | None = None
    image_url: str | None = None


class QuantumRequest(BaseModel):
    qubits: int = Field(default=20, ge=1, le=30)
    shots: int = Field(default=1, ge=1, le=10000)


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
            "image_editing": True,
        },
    }


@app.get("/health")
@app.get("/status")
async def health():
    return {
        "status": "healthy",
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
            image_url=request.image_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI request failed: {type(exc).__name__}")


@app.post("/quantum")
async def quantum(request: QuantumRequest):
    try:
        result = run_quantum(num_qubits=request.qubits, shots=request.shots)
        return {"type": "quantum", "status": "completed", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Quantum request failed: {type(exc).__name__}")


MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "media"))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


@app.post("/upload-image")
async def upload_image(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    data = await file.read()
    if len(data) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image larger than 20 MB limit.")

    filename = f"{uuid.uuid4().hex}{ALLOWED_IMAGE_TYPES[file.content_type]}"
    target = MEDIA_DIR / filename
    target.write_bytes(data)

    public_base = os.getenv("PUBLIC_BASE_URL", str(request.base_url)).rstrip("/")
    return {
        "status": "uploaded",
        "filename": file.filename or filename,
        "mime": file.content_type,
        "url": f"{public_base}/media/{filename}",
    }


@app.get("/media/{filename}")
async def get_media(filename: str):
    target = MEDIA_DIR / Path(filename).name
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(target)
    