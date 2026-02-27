from __future__ import annotations

import json
import uuid
from typing import Any, Generator

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth import TokenRequest, TokenResponse, create_access_token, get_current_user_id
from database import ChatSession, Message, SessionLocal, User, init_db, utcnow
from gemini_client import get_gemini_response, get_session_title

app = FastAPI(title="AI Prompt Creation API")

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


class SessionListItem(BaseModel):
    session_id: str
    title: str
    updated_at: str
    preview: str | None = None


class HistoryMessage(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def _ensure_user(db: Session, user_id: str) -> User:
    user = db.get(User, user_id)
    if user is None:
        user = User(id=user_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def _to_gemini_history(messages: list[Message]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for msg in messages:
        role = "model" if msg.role == "assistant" else msg.role
        mapped.append({"role": role, "parts": [msg.content]})
    return mapped


@app.post("/auth/token", response_model=TokenResponse)
def create_token(payload: TokenRequest, db: Session = Depends(get_db)) -> TokenResponse:
    _ensure_user(db, payload.user_id)
    token = create_access_token(payload.user_id)
    return TokenResponse(access_token=token)


@app.post("/chat")
def chat(
    payload: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _ensure_user(db, user_id)

    session = db.get(ChatSession, payload.session_id)
    is_first_message = False
    if session is None:
        session = ChatSession(id=payload.session_id, user_id=user_id, title="New Session")
        db.add(session)
        is_first_message = True
    elif session.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    user_message = Message(
        id=uuid.uuid4().hex,
        session_id=payload.session_id,
        role="user",
        content=payload.user_input,
    )
    db.add(user_message)
    session.updated_at = utcnow()
    db.commit()

    db_history = db.scalars(
        select(Message).where(Message.session_id == payload.session_id).order_by(Message.created_at.asc())
    ).all()

    try:
        gemini_response = get_gemini_response(_to_gemini_history(db_history))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini call failed: {exc}") from exc

    assistant_message = Message(
        id=uuid.uuid4().hex,
        session_id=payload.session_id,
        role="assistant",
        content=json.dumps(gemini_response, ensure_ascii=False),
    )
    db.add(assistant_message)
    session.updated_at = utcnow()

    if is_first_message:
        try:
            session.title = get_session_title(payload.user_input)
        except Exception:
            pass

    db.commit()
    return gemini_response


@app.get("/sessions", response_model=list[SessionListItem])
def list_sessions(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> list[SessionListItem]:
    sessions = db.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
    ).all()

    items: list[SessionListItem] = []
    for chat_session in sessions:
        last_message = db.scalars(
            select(Message)
            .where(Message.session_id == chat_session.id)
            .order_by(Message.created_at.desc())
            .limit(1)
        ).first()
        preview = None
        if last_message is not None:
            preview = (last_message.content or "")[:140]

        items.append(
            SessionListItem(
                session_id=chat_session.id,
                title=chat_session.title,
                updated_at=chat_session.updated_at.isoformat(),
                preview=preview,
            )
        )
    return items


@app.get("/history/{session_id}", response_model=list[HistoryMessage])
def session_history(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> list[HistoryMessage]:
    session = db.get(ChatSession, session_id)
    if session is None or session.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    messages = db.scalars(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc())
    ).all()
    return [
        HistoryMessage(
            id=message.id,
            role=message.role,
            content=message.content,
            created_at=message.created_at.isoformat(),
        )
        for message in messages
    ]


@app.post("/reset")
def reset(
    payload: SessionRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = db.get(ChatSession, payload.session_id)
    if session is None or session.user_id != user_id:
        return {"session_id": payload.session_id, "reset": False}

    db.delete(session)
    db.commit()
    return {"session_id": payload.session_id, "reset": True}


@app.get("/summary/{session_id}")
def summary(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = db.get(ChatSession, session_id)
    if session is None or session.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    user_answers = db.scalars(
        select(Message.content)
        .where(Message.session_id == session_id, Message.role == "user")
        .order_by(Message.created_at.asc())
    ).all()
    return {"session_id": session_id, "user_answers": list(user_answers)}
