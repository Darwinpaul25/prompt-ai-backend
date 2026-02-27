from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gemini_client import get_gemini_response, get_session_title
from session_manager import SessionManager

SESSIONS_DIR = "sessions"
METADATA_FILE = os.path.join(SESSIONS_DIR, "metadata.json")
os.makedirs(SESSIONS_DIR, exist_ok=True)

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


def _load_metadata() -> list[dict[str, str]]:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    if not os.path.exists(METADATA_FILE):
        return []

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        session_id = item.get("id")
        title = item.get("title")
        if isinstance(session_id, str) and isinstance(title, str):
            normalized.append({"id": session_id, "title": title})
    return normalized


def update_metadata(session_id: str, title: str) -> None:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    metadata = _load_metadata()
    updated = False

    for item in metadata:
        if item["id"] == session_id:
            item["title"] = title
            updated = True
            break

    if not updated:
        metadata.append({"id": session_id, "title": title})

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


@app.post("/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    is_first_message = False
    try:
        is_first_message = not session_manager.session_exists(payload.session_id)
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

    if is_first_message:
        try:
            title = get_session_title(payload.user_input)
            update_metadata(payload.session_id, title)
        except Exception:
            # Title generation is a non-blocking secondary call.
            pass

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


@app.get("/sessions")
def list_sessions() -> list[dict[str, str]]:
    return _load_metadata()


@app.get("/history/{session_id}")
def session_history(session_id: str) -> list[dict[str, Any]]:
    try:
        if not session_manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return session_manager.get_history(session_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
