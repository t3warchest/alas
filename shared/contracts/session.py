"""
shared/contracts/session.py

Session state models shared between the gateway's session store
and any component that needs to reason about a session's lifecycle.

These are plain dataclasses — no pydantic, no ORM, no external deps.
The gateway session store serialises these to/from SQLite or memory.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SessionStatus(str, Enum):
    """Lifecycle states a gateway session can be in."""
    INITIALISING = "initialising"   # POST /sessions received, agent not yet started
    ACTIVE       = "active"         # WebSocket connected, conversation ongoing
    DISCONNECTED = "disconnected"   # Client disconnected but session still alive (reconnectable)
    ENDED        = "ended"          # Session fully complete (agent sent session_end)
    ERROR        = "error"          # Non-recoverable failure


@dataclass
class GatewaySession:
    """
    Gateway-side session record. Tracks connection state and metadata.
    The agent service has its own STM — this is purely the gateway's view.

    Stored in the SessionStore (in-memory dict or SQLite row).
    """
    session_id: str
    user_id: str
    scenario_id: str

    status: SessionStatus = SessionStatus.INITIALISING
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)

    # Populated after agent service responds to session creation
    scenario_title: str = ""
    persona_name: str = ""
    opening_line: str = ""
    current_phase: str = "setup"

    # Turn tracking — used to detect missed events on reconnect
    last_turn_index: int = 0
    total_turns: int = 0

    # Aggregate scores (updated after each turn for reconnect state restoration)
    latest_composite: float | None = None

    def touch(self) -> None:
        self.last_active_at = time.time()

    def mark_active(self) -> None:
        self.status = SessionStatus.ACTIVE
        self.touch()

    def mark_disconnected(self) -> None:
        self.status = SessionStatus.DISCONNECTED
        self.touch()

    def mark_ended(self) -> None:
        self.status = SessionStatus.ENDED
        self.touch()

    def mark_error(self) -> None:
        self.status = SessionStatus.ERROR
        self.touch()

    def is_reconnectable(self) -> bool:
        """A disconnected session can be reconnected within the TTL window."""
        ttl_seconds = 3600  # 1 hour
        return (
            self.status == SessionStatus.DISCONNECTED
            and (time.time() - self.last_active_at) < ttl_seconds
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id":       self.session_id,
            "user_id":          self.user_id,
            "scenario_id":      self.scenario_id,
            "status":           self.status.value,
            "created_at":       self.created_at,
            "last_active_at":   self.last_active_at,
            "scenario_title":   self.scenario_title,
            "persona_name":     self.persona_name,
            "opening_line":     self.opening_line,
            "current_phase":    self.current_phase,
            "last_turn_index":  self.last_turn_index,
            "total_turns":      self.total_turns,
            "latest_composite": self.latest_composite,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GatewaySession":
        return cls(
            session_id      = data["session_id"],
            user_id         = data["user_id"],
            scenario_id     = data["scenario_id"],
            status          = SessionStatus(data.get("status", "active")),
            created_at      = data.get("created_at", time.time()),
            last_active_at  = data.get("last_active_at", time.time()),
            scenario_title  = data.get("scenario_title", ""),
            persona_name    = data.get("persona_name", ""),
            opening_line    = data.get("opening_line", ""),
            current_phase   = data.get("current_phase", "setup"),
            last_turn_index = data.get("last_turn_index", 0),
            total_turns     = data.get("total_turns", 0),
            latest_composite = data.get("latest_composite"),
        )


# ---------------------------------------------------------------------------
# Request/response DTOs for the gateway's REST endpoints
# (Kept here so shared/ is the single source for all public contracts)
# ---------------------------------------------------------------------------

@dataclass
class CreateSessionDTO:
    """POST /api/sessions request body."""
    user_id: str
    scenario_id: str
    session_id: str | None = None   # caller can supply; gateway generates if absent

    @classmethod
    def from_dict(cls, data: dict) -> "CreateSessionDTO":
        return cls(
            user_id     = data["user_id"],
            scenario_id = data["scenario_id"],
            session_id  = data.get("session_id"),
        )


@dataclass
class SessionCreatedDTO:
    """POST /api/sessions response body."""
    session_id:     str
    scenario_id:    str
    scenario_title: str
    persona_name:   str
    opening_line:   str
    ws_url:         str     # e.g. ws://localhost:8001/ws/{session_id}

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id":     self.session_id,
            "scenario_id":    self.scenario_id,
            "scenario_title": self.scenario_title,
            "persona_name":   self.persona_name,
            "opening_line":   self.opening_line,
            "ws_url":         self.ws_url,
        }


@dataclass
class SessionSummaryDTO:
    """GET /api/sessions/{id} response body."""
    session_id:      str
    status:          str
    scenario_id:     str
    scenario_title:  str
    current_phase:   str
    total_turns:     int
    latest_composite: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id":       self.session_id,
            "status":           self.status,
            "scenario_id":      self.scenario_id,
            "scenario_title":   self.scenario_title,
            "current_phase":    self.current_phase,
            "total_turns":      self.total_turns,
            "latest_composite": self.latest_composite,
        }
