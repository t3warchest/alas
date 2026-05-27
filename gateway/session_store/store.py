"""
gateway/session_store/store.py

Lightweight gateway session store.

Two backends (selected by PERSIST_SESSIONS env var):
  InMemorySessionStore  — pure dict, no persistence, suitable for single-process demo
  SQLiteSessionStore    — SQLite-backed, survives process restarts, enables reconnect

Both implement the same SessionStore interface so the gateway code is
backend-agnostic. Switching is one config flag.

Session lifecycle:
  1. create()          — gateway calls this when POST /api/sessions succeeds
  2. get()             — called by every WS handler to validate the session exists
  3. update_status()   — called on connect, disconnect, end, error
  4. update_turn()     — called after each turn to track progress
  5. delete()          — called after session_end to free memory
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from shared.contracts.session import GatewaySession, SessionStatus
from shared.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class SessionStore(ABC):

    @abstractmethod
    def create(self, session: GatewaySession) -> None:
        """Persist a newly created session."""

    @abstractmethod
    def get(self, session_id: str) -> GatewaySession | None:
        """Retrieve a session by ID. Returns None if not found."""

    @abstractmethod
    def update(self, session: GatewaySession) -> None:
        """Persist changes to an existing session."""

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Remove a session from the store."""

    @abstractmethod
    def list_for_user(self, user_id: str) -> list[GatewaySession]:
        """Return all sessions belonging to a user."""

    # ── Convenience helpers (implemented in terms of get/update) ──────────

    def update_status(self, session_id: str, status: SessionStatus) -> None:
        s = self.get(session_id)
        if s:
            s.status = status
            s.touch()
            self.update(s)

    def update_turn(
        self,
        session_id: str,
        turn_index: int,
        phase: str | None = None,
        composite: float | None = None,
    ) -> None:
        s = self.get(session_id)
        if s:
            s.last_turn_index = turn_index
            s.total_turns = max(s.total_turns, turn_index + 1)
            if phase:
                s.current_phase = phase
            if composite is not None:
                s.latest_composite = composite
            s.touch()
            self.update(s)

    def expire_old_sessions(self, ttl_seconds: int = 7200) -> int:
        """Evict DISCONNECTED sessions older than ttl_seconds. Returns count removed."""
        return 0  # subclasses may override


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------

class InMemorySessionStore(SessionStore):
    """Thread-safe in-memory session store backed by a plain dict."""

    def __init__(self):
        self._data: dict[str, GatewaySession] = {}
        self._lock = threading.Lock()

    def create(self, session: GatewaySession) -> None:
        with self._lock:
            self._data[session.session_id] = session
        log.debug("session_stored", session_id=session.session_id)

    def get(self, session_id: str) -> GatewaySession | None:
        with self._lock:
            return self._data.get(session_id)

    def update(self, session: GatewaySession) -> None:
        with self._lock:
            if session.session_id in self._data:
                self._data[session.session_id] = session

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def list_for_user(self, user_id: str) -> list[GatewaySession]:
        with self._lock:
            return [s for s in self._data.values() if s.user_id == user_id]

    def expire_old_sessions(self, ttl_seconds: int = 7200) -> int:
        cutoff = time.time() - ttl_seconds
        expired = []
        with self._lock:
            for sid, s in self._data.items():
                if s.status == SessionStatus.DISCONNECTED and s.last_active_at < cutoff:
                    expired.append(sid)
            for sid in expired:
                del self._data[sid]
        if expired:
            log.info("sessions_expired", count=len(expired))
        return len(expired)

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

class SQLiteSessionStore(SessionStore):
    """
    SQLite-backed session store.
    Enables sessions to survive gateway restarts (reconnect support).
    Each session is a single JSON-serialised row.
    """

    def __init__(self, db_path: str | Path = "./data/gateway_sessions.db"):
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    user_id      TEXT NOT NULL,
                    data_json    TEXT NOT NULL,
                    created_at   REAL NOT NULL,
                    last_active  REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user ON sessions(user_id)"
            )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def create(self, session: GatewaySession) -> None:
        data = json.dumps(session.to_dict())
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, user_id, data_json, created_at, last_active)
                   VALUES (?, ?, ?, ?, ?)""",
                (session.session_id, session.user_id,
                 data, session.created_at, session.last_active_at),
            )
        log.debug("session_persisted", session_id=session.session_id)

    def get(self, session_id: str) -> GatewaySession | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return GatewaySession.from_dict(json.loads(row["data_json"]))

    def update(self, session: GatewaySession) -> None:
        data = json.dumps(session.to_dict())
        with self._conn() as conn:
            conn.execute(
                """UPDATE sessions SET data_json = ?, last_active = ?
                   WHERE session_id = ?""",
                (data, session.last_active_at, session.session_id),
            )

    def delete(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )

    def list_for_user(self, user_id: str) -> list[GatewaySession]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data_json FROM sessions WHERE user_id = ?", (user_id,)
            ).fetchall()
        return [GatewaySession.from_dict(json.loads(r["data_json"])) for r in rows]

    def expire_old_sessions(self, ttl_seconds: int = 7200) -> int:
        cutoff = time.time() - ttl_seconds
        with self._conn() as conn:
            result = conn.execute(
                """DELETE FROM sessions
                   WHERE last_active < ? AND data_json LIKE '%"disconnected"%'""",
                (cutoff,),
            )
        count = result.rowcount
        if count:
            log.info("sessions_expired_sqlite", count=count)
        return count


# ---------------------------------------------------------------------------
# Factory — selected by config
# ---------------------------------------------------------------------------

def create_session_store(persist: bool = False, db_path: str = "./data/gateway_sessions.db") -> SessionStore:
    if persist:
        log.info("session_store", backend="sqlite", path=db_path)
        return SQLiteSessionStore(db_path)
    log.info("session_store", backend="memory")
    return InMemorySessionStore()
