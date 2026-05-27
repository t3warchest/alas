"""
alas/agent_service/api/models.py

Pydantic request and response models for all HTTP/WebSocket endpoints.
These are the public contract — internal types (AgentState, TurnScore, etc.)
never cross the API boundary directly.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., description="Opaque user identifier")
    scenario_id: str = Field(..., description="Scenario to run (see GET /scenarios)")
    session_id: str | None = Field(
        default=None,
        description="Optional caller-provided session ID (UUID). Auto-generated if omitted.",
    )


class SendMessageRequest(BaseModel):
    session_id: str
    student_message: str = Field(..., min_length=1, max_length=2000)


class EndSessionRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class ScenarioSummary(BaseModel):
    id: str
    title: str
    description: str


class CreateSessionResponse(BaseModel):
    session_id: str
    scenario_id: str
    scenario_title: str
    persona_name: str
    opening_line: str


class TurnScoreResponse(BaseModel):
    turn_index: int
    clarity: float
    empathy: float
    structure: float
    relevance: float
    confidence: float
    composite: float
    rationale: str


class SendMessageResponse(BaseModel):
    session_id: str
    turn_index: int
    avatar_response: str
    emotion: str
    scenario_phase: str
    turn_score: TurnScoreResponse | None = None
    session_ended: bool = False
    session_summary: dict[str, Any] | None = None


class SessionScoresResponse(BaseModel):
    session_id: str
    scores: list[TurnScoreResponse]
    turn_count: int


class EndSessionResponse(BaseModel):
    session_id: str
    summary: dict[str, Any] | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    model: str


# ---------------------------------------------------------------------------
# WebSocket event envelope
# ---------------------------------------------------------------------------

class WSEvent(BaseModel):
    """All WebSocket messages use this envelope."""
    type: str  # "token" | "result" | "error" | "ping"
    content: str | None = None          # for token events
    data: dict[str, Any] | None = None  # for result events
    message: str | None = None          # for error events
