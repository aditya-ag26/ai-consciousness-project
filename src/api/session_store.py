"""
In-memory conversation store for the chat API.

Holds the message history of each chat session so the bot can answer follow-up
questions coherently. State is process-local: it is lost on restart and is not
shared between replicas, so a deployment that scales beyond a single backend
instance needs to swap this for a shared store (e.g. Redis).
"""
import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    sources: list[str] = field(default_factory=list)
    refused: bool = False


@dataclass
class Session:
    messages: list[Message] = field(default_factory=list)
    last_active: float = field(default_factory=time.monotonic)


class SessionStore:
    """Thread-safe store of chat sessions with idle expiry."""

    def __init__(self, ttl_seconds: int = 1800, max_sessions: int = 1000):
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        """Creates a new empty session and returns its id."""
        session_id = uuid.uuid4().hex
        with self._lock:
            self._evict_expired()
            if len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions, key=lambda s: self._sessions[s].last_active)
                del self._sessions[oldest]
            self._sessions[session_id] = Session()
        return session_id

    def exists(self, session_id: str) -> bool:
        with self._lock:
            self._evict_expired()
            return session_id in self._sessions

    def get_messages(self, session_id: str) -> list[Message]:
        """Returns a copy of the session's messages. Raises KeyError if unknown."""
        with self._lock:
            self._evict_expired()
            session = self._sessions[session_id]
            session.last_active = time.monotonic()
            return list(session.messages)

    def append(self, session_id: str, message: Message) -> None:
        """Appends a message to a session. Raises KeyError if unknown."""
        with self._lock:
            self._evict_expired()
            session = self._sessions[session_id]
            session.messages.append(message)
            session.last_active = time.monotonic()

    def delete(self, session_id: str) -> None:
        """Removes a session. Raises KeyError if unknown."""
        with self._lock:
            del self._sessions[session_id]

    def history_pairs(self, session_id: str) -> list[tuple[str, str]]:
        """Returns the session history in the (role, content) form QueryBot expects."""
        return [(m.role, m.content) for m in self.get_messages(session_id)]

    def _evict_expired(self) -> None:
        """Drops sessions idle for longer than the TTL. Caller must hold the lock."""
        cutoff = time.monotonic() - self._ttl
        for session_id in [
            sid for sid, s in self._sessions.items() if s.last_active < cutoff
        ]:
            del self._sessions[session_id]
