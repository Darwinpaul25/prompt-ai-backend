from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gemini_client import get_gemini_response
from session_manager import SessionManager


import os
if not os.path.exists("sessions"):
    os.makedirs("sessions")

app = FastAPI(title="AI Prompt Creation API")
session_manager = SessionManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    user_input: str


class SessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


@app.post("/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    try:
        history = session_manager.get_history(payload.session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    history.append({"role": "user", "parts": [payload.user_input]})

    try:
        gemini_response = get_gemini_response(history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini call failed: {exc}") from exc

    history.append(
        {
            "role": "model",
            "parts": [json.dumps(gemini_response, ensure_ascii=False)],
        }
    )

    try:
        session_manager.save_history(payload.session_id, history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save history: {exc}") from exc

    return gemini_response


@app.post("/reset")
def reset(payload: SessionRequest) -> dict[str, Any]:
    try:
        deleted = session_manager.delete_history(payload.session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session_id": payload.session_id, "reset": deleted}


@app.get("/summary/{session_id}")
def summary(session_id: str) -> dict[str, Any]:
    try:
        history = session_manager.get_history(session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user_answers: list[str] = []
    for message in history:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        parts = message.get("parts", [])
        if not isinstance(parts, list):
            continue
        if not parts:
            user_answers.append("")
            continue
        first = parts[0]
        user_answers.append("" if first is None else str(first))

    return {"session_id": session_id, "user_answers": user_answers}
