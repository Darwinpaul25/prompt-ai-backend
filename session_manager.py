from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SessionManager:
    """Persist chat history per session_id under ./sessions as JSON files."""

    def __init__(self, sessions_dir: str | Path = "sessions") -> None:
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_file(self, session_id: str) -> Path:
        safe_session_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        if not safe_session_id:
            raise ValueError("session_id must contain at least one valid character")
        return self.sessions_dir / f"{safe_session_id}.json"

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return session history; [] if the session file does not exist."""
        session_file = self._session_file(session_id)
        if not session_file.exists():
            return []

        with session_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Stored session history must be a list")
        return data

    def save_history(self, session_id: str, history: list[dict[str, Any]]) -> None:
        """Write updated session history to disk.

        Expected format:
        [
            {"role": "user" | "model", "parts": ["text_content"]},
            ...
        ]
        """
        session_file = self._session_file(session_id)

        with session_file.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def delete_history(self, session_id: str) -> bool:
        """Delete a session history file. Returns True if deleted, False if absent."""
        session_file = self._session_file(session_id)
        if not session_file.exists():
            return False
        session_file.unlink()
        return True
