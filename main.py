from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from json import JSONDecodeError
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
session_manager = SessionManager(SESSIONS_DIR)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    user_input: str = Field(..., min_length=1)


class SessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_metadata() -> list[dict[str, str]]:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    if not os.path.exists(METADATA_FILE):
        return []

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []

    items: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        session_id = item.get("id")
        title = item.get("title")
        updated_at = item.get("updated_at", "")
        preview = item.get("preview", "")
        if isinstance(session_id, str) and isinstance(title, str):
            items.append(
                {
                    "id": session_id,
                    "title": title,
                    "updated_at": str(updated_at),
                    "preview": str(preview),
                }
            )
    return items


def update_metadata(
    session_id: str,
    title: str,
    updated_at: str | None = None,
    preview: str | None = None,
) -> None:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    metadata = _load_metadata()
    updated = False

    for item in metadata:
        if item["id"] == session_id:
            item["title"] = title
            if updated_at is not None:
                item["updated_at"] = updated_at
            if preview is not None:
                item["preview"] = preview
            updated = True
            break

    if not updated:
        metadata.append(
            {
                "id": session_id,
                "title": title,
                "updated_at": updated_at or "",
                "preview": preview or "",
            }
        )

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _to_gemini_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", ""))
        if role == "assistant":
            role = "model"
        if role not in {"user", "model", "system"}:
            continue

        content = msg.get("content")
        parts = msg.get("parts")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(parts, list) and parts:
            text = "" if parts[0] is None else str(parts[0])

        mapped.append({"role": role, "parts": [text]})
    return mapped


@app.post("/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    try:
        is_first_message = not session_manager.session_exists(payload.session_id)
        history = session_manager.get_history(payload.session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    now = _now_iso()
    user_message = {
        "id": uuid.uuid4().hex,
        "role": "user",
        "content": payload.user_input,
        "parts": [payload.user_input],
        "created_at": now,
    }
    history.append(user_message)

    try:
        gemini_response = get_gemini_response(_to_gemini_history(history))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini call failed: {exc}") from exc

    assistant_content = json.dumps(gemini_response, ensure_ascii=False)
    history.append(
        {
            "id": uuid.uuid4().hex,
            "role": "assistant",
            "content": assistant_content,
            "parts": [assistant_content],
            "created_at": _now_iso(),
        }
    )

    try:
        session_manager.save_history(payload.session_id, history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save history: {exc}") from exc

    title = "New Session"
    metadata = _load_metadata()
    existing = next((m for m in metadata if m["id"] == payload.session_id), None)
    if existing is not None:
        title = existing["title"]

    if is_first_message:
        try:
            title = get_session_title(payload.user_input)
        except Exception:
            pass

    update_metadata(
        session_id=payload.session_id,
        title=title,
        updated_at=_now_iso(),
        preview=assistant_content[:140],
    )

    return gemini_response


@app.get("/sessions")
def list_sessions() -> list[dict[str, str]]:
    items = _load_metadata()
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return [
        {
            "session_id": item["id"],
            "title": item["title"],
            "updated_at": item.get("updated_at", ""),
            "preview": item.get("preview", ""),
        }
        for item in items
    ]


@app.get("/history/{session_id}")
def session_history(session_id: str) -> list[dict[str, str]]:
    try:
        if not session_manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        history = session_manager.get_history(session_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized: list[dict[str, str]] = []
    for i, msg in enumerate(history):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", ""))
        if role == "model":
            role = "assistant"
        if role not in {"user", "assistant", "system"}:
            role = "assistant"

        content = msg.get("content")
        if not isinstance(content, str):
            parts = msg.get("parts", [])
            if isinstance(parts, list) and parts:
                content = "" if parts[0] is None else str(parts[0])
            else:
                content = ""

        msg_id = msg.get("id")
        created_at = msg.get("created_at")
        normalized.append(
            {
                "id": str(msg_id) if isinstance(msg_id, str) and msg_id else f"msg-{i+1}",
                "role": role,
                "content": content,
                "created_at": str(created_at) if isinstance(created_at, str) else "",
            }
        )

    return normalized


@app.post("/reset")
def reset(payload: SessionRequest) -> dict[str, Any]:
    try:
        deleted = session_manager.delete_history(payload.session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    metadata = [item for item in _load_metadata() if item.get("id") != payload.session_id]
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

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
        content = message.get("content")
        if isinstance(content, str):
            user_answers.append(content)
            continue
        parts = message.get("parts", [])
        if isinstance(parts, list) and parts:
            user_answers.append("" if parts[0] is None else str(parts[0]))
        else:
            user_answers.append("")

    return {"session_id": session_id, "user_answers": user_answers}
