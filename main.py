import os
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from router import process_request
from queen import queen_status
from quantum import quantum_status

APP_NAME = "Quantum Queen AI"
APP_VERSION = "2.0.0"

raw_origins = os.getenv(
    "BITTOOL_ORIGINS",
    "*"
).strip()

if raw_origins == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip()
    ]

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Hybrid Qwen + Qiskit Quantum AI backend."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=12000
    )
    history: list[dict[str, Any]] = Field(
        default_factory=list
    )

@app.get("/")
async def home():
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "endpoints": [
            "/chat",
            "/health",
            "/status",
            "/quantum"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "queen_ai": queen_status(),
        "quantum_engine": quantum_status()
    }

@app.get("/status")
async def status():
    return {
        "app": {
            "name": APP_NAME,
            "version": APP_VERSION
        },
        "queen_ai": queen_status(),
        "quantum_engine": quantum_status()
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = process_request(
            user_message=request.message,
            history=request.history
        )
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AI request failed: {type(exc).__name__}"
        ) from exc

@app.post("/quantum")
async def quantum_test():
    try:
        from quantum import run_quantum
        result = run_quantum(
            num_qubits=20,
            shots=1
        )
        return {
            "type": "quantum",
            "result": result
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Quantum request failed: {type(exc).__name__}"
        ) from exc