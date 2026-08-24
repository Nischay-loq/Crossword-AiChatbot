"""
agent/session.py
Simple in-memory session/conversation history manager.

Each session is identified by a session_id string.
Stores list of {role, content} message dicts for multi-turn context.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.last_active = time.time()

    def get_history(self, max_turns: int = 10) -> list[dict[str, str]]:
        """Return last N turns (each turn = 1 user + 1 assistant message)."""
        # Each "turn" = 2 messages (user + assistant), so max_turns * 2 messages
        return self.messages[-(max_turns * 2):]

    def clear(self) -> None:
        self.messages = []


class SessionManager:
    """In-memory session store. Sessions are isolated from each other."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create_session(self, session_id: str | None = None) -> Session:
        sid = session_id or str(uuid.uuid4())
        session = Session(session_id=sid)
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            return self.create_session(session_id)
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


# Global singleton for use across the application
session_manager = SessionManager()
