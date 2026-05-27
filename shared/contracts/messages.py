"""
shared/contracts/messages.py

Dataclass definitions for every WebSocket message shape.

Rules:
  - All fields have defaults where possible so partial construction works.
  - to_dict() / from_dict() on every class keeps JSON handling explicit and testable.
  - No pydantic dependency — plain dataclasses so this module works
    in environments where pydantic isn't installed (e.g. a lightweight frontend proxy).
  - The gateway and agent service both import from here. The frontend uses
    the companion TypeScript types generated from these definitions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from shared.contracts.events import (
    WS_EVT_AUTH_OK,
    WS_EVT_CONNECTED,
    WS_EVT_ERROR,
    WS_EVT_FATAL,
    WS_EVT_PHASE_CHANGE,
    WS_EVT_PONG,
    WS_EVT_SCORE,
    WS_EVT_SESSION_END,
    WS_EVT_TOKEN,
    WS_EVT_TURN_DONE,
)


# ---------------------------------------------------------------------------
# Base envelope
# ---------------------------------------------------------------------------

@dataclass
class WSMessage:
    """
    Base envelope for every WebSocket message.
    Every message has at minimum a `type` field.
    """
    type: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "WSMessage":
        return cls(type=data.get("type", ""))


# ---------------------------------------------------------------------------
# Client → Gateway messages
# ---------------------------------------------------------------------------

@dataclass
class ClientMessage(WSMessage):
    """A student chat message."""
    content: str = ""
    session_id: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "ClientMessage":
        return cls(
            type=data.get("type", "message"),
            content=data.get("content", ""),
            session_id=data.get("session_id", ""),
        )


@dataclass
class ClientPing(WSMessage):
    type: str = "ping"
    timestamp: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "ClientPing":
        return cls(timestamp=data.get("timestamp", 0.0))


@dataclass
class ClientReconnect(WSMessage):
    """Client wants to resume an existing session after a dropped connection."""
    type: str = "reconnect"
    session_id: str = ""
    last_turn_index: int = 0    # client's last known turn — used to detect missed events

    @classmethod
    def from_dict(cls, data: dict) -> "ClientReconnect":
        return cls(
            session_id=data.get("session_id", ""),
            last_turn_index=data.get("last_turn_index", 0),
        )


# ---------------------------------------------------------------------------
# Gateway → Client messages
# ---------------------------------------------------------------------------

@dataclass
class AuthOkEvent(WSMessage):
    """Sent immediately after a WebSocket connection is authenticated."""
    type: str = WS_EVT_AUTH_OK
    user_id: str = ""
    token_expires_in: int = 0   # seconds

    @classmethod
    def from_dict(cls, data: dict) -> "AuthOkEvent":
        return cls(
            user_id=data.get("user_id", ""),
            token_expires_in=data.get("token_expires_in", 0),
        )


@dataclass
class ConnectedEvent(WSMessage):
    """Sent when the session is fully initialised and the avatar has spoken."""
    type: str = WS_EVT_CONNECTED
    session_id: str = ""
    scenario_id: str = ""
    scenario_title: str = ""
    persona_name: str = ""
    opening_line: str = ""
    phase: str = "setup"

    @classmethod
    def from_dict(cls, data: dict) -> "ConnectedEvent":
        return cls(**{k: data.get(k, v) for k, v in cls.__dataclass_fields__.items()
                      if k != "type"})


@dataclass
class TokenEvent(WSMessage):
    """One streaming token from the avatar's response."""
    type: str = WS_EVT_TOKEN
    content: str = ""
    turn_index: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "TokenEvent":
        return cls(
            content=data.get("content", ""),
            turn_index=data.get("turn_index", 0),
        )


@dataclass
class TurnDoneEvent(WSMessage):
    """
    Signals that the avatar's full response for this turn is complete.
    Contains the complete spoken text, emotion state, and current phase.
    The score arrives separately via ScoreEvent.
    """
    type: str = WS_EVT_TURN_DONE
    session_id: str = ""
    turn_index: int = 0
    avatar_response: str = ""
    emotion: str = "neutral"
    phase: str = ""
    session_ended: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "TurnDoneEvent":
        return cls(
            session_id=data.get("session_id", ""),
            turn_index=data.get("turn_index", 0),
            avatar_response=data.get("avatar_response", ""),
            emotion=data.get("emotion", "neutral"),
            phase=data.get("phase", ""),
            session_ended=data.get("session_ended", False),
        )


@dataclass
class ScoreEvent(WSMessage):
    """Per-turn rubric evaluation result."""
    type: str = WS_EVT_SCORE
    turn_index: int = 0
    clarity: float = 0.0
    empathy: float = 0.0
    structure: float = 0.0
    relevance: float = 0.0
    confidence: float = 0.0
    composite: float = 0.0
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "ScoreEvent":
        return cls(**{k: data.get(k, v)
                      for k, v in cls.__dataclass_fields__.items()
                      if k != "type"})


@dataclass
class PhaseChangeEvent(WSMessage):
    """Emitted when the branching engine transitions to a new scenario phase."""
    type: str = WS_EVT_PHASE_CHANGE
    from_phase: str = ""
    to_phase: str = ""
    reason: str = ""
    turn_index: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "PhaseChangeEvent":
        return cls(
            from_phase=data.get("from_phase", ""),
            to_phase=data.get("to_phase", ""),
            reason=data.get("reason", ""),
            turn_index=data.get("turn_index", 0),
        )


@dataclass
class SessionEndEvent(WSMessage):
    """Final event — session is over, summary attached."""
    type: str = WS_EVT_SESSION_END
    session_id: str = ""
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "session_id": self.session_id,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionEndEvent":
        return cls(
            session_id=data.get("session_id", ""),
            summary=data.get("summary", {}),
        )


@dataclass
class PongEvent(WSMessage):
    type: str = WS_EVT_PONG
    timestamp: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "PongEvent":
        return cls(timestamp=data.get("timestamp", 0.0))


@dataclass
class ErrorEvent(WSMessage):
    """Recoverable error — session stays open."""
    type: str = WS_EVT_ERROR
    code: str = "unknown"
    message: str = ""
    recoverable: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "ErrorEvent":
        return cls(
            code=data.get("code", "unknown"),
            message=data.get("message", ""),
            recoverable=data.get("recoverable", True),
        )


@dataclass
class FatalEvent(WSMessage):
    """Non-recoverable error — gateway is closing the connection."""
    type: str = WS_EVT_FATAL
    code: str = "fatal"
    message: str = ""
    recoverable: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "FatalEvent":
        return cls(
            code=data.get("code", "fatal"),
            message=data.get("message", ""),
        )


# ---------------------------------------------------------------------------
# Factory: parse an incoming client frame into the right dataclass
# ---------------------------------------------------------------------------

_CLIENT_PARSERS = {
    "message":   ClientMessage.from_dict,
    "ping":      ClientPing.from_dict,
    "end":       WSMessage.from_dict,        # bare WSMessage is fine for "end"
    "reconnect": ClientReconnect.from_dict,
}


def parse_client_frame(raw: str) -> WSMessage | None:
    """
    Parse a raw JSON string from the client into the appropriate dataclass.
    Returns None on parse failure.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    event_type = data.get("type", "message")
    parser = _CLIENT_PARSERS.get(event_type, WSMessage.from_dict)
    try:
        return parser(data)
    except Exception:
        return WSMessage.from_dict(data)
