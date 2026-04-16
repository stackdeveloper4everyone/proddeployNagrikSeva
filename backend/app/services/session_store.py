from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class SessionState:
    session_id: str
    user_id: str
    detected_language: str = "en-IN"
    response_language: str = "en-IN"
    intent: str = "general"
    collected_details: dict[str, str] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str | None, user_id: str | None) -> SessionState:
        with self._lock:
            self._prune_locked()
            actual_session_id = session_id or str(uuid.uuid4())
            if actual_session_id not in self._sessions:
                self._sessions[actual_session_id] = SessionState(
                    session_id=actual_session_id,
                    user_id=user_id or actual_session_id,
                )
            session = self._sessions[actual_session_id]
            session.updated_at = time.time()
            return session

    def save(self, session: SessionState) -> None:
        with self._lock:
            session.updated_at = time.time()
            self._sessions[session.session_id] = session

    def _prune_locked(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        expired = [key for key, value in self._sessions.items() if value.updated_at < cutoff]
        for key in expired:
            self._sessions.pop(key, None)
