"""
alas/agent_service/api/routes.py

All HTTP REST endpoints for the ALAS Agent Service.

Endpoints:
  GET  /health                    — liveness probe
  GET  /scenarios                 — list available scenarios
  POST /sessions                  — create a new session
  POST /sessions/message          — send a student message (non-streaming)
  GET  /sessions/{id}/scores      — get per-turn scores for a session
  POST /sessions/{id}/end         — force-end a session and get summary
"""

from __future__ import annotations

import os
from fastapi import APIRouter, HTTPException, status

from agent_service.api.models import (
    CreateSessionRequest,
    CreateSessionResponse,
    EndSessionRequest,
    EndSessionResponse,
    HealthResponse,
    ScenarioSummary,
    SendMessageRequest,
    SendMessageResponse,
    SessionScoresResponse,
    TurnScoreResponse,
)
from agent_service.scenarios.registry import list_scenarios
from agent_service.session import orchestrator
from agent_service.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["infra"])
async def health():
    return HealthResponse(
        status="ok",
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@router.get("/scenarios", response_model=list[ScenarioSummary], tags=["scenarios"])
async def get_scenarios():
    """List all available simulation scenarios."""
    return [ScenarioSummary(**s) for s in list_scenarios()]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["sessions"],
)
async def create_session(req: CreateSessionRequest):
    """
    Create a new session for a user.
    Returns the session ID and the avatar's opening line.
    """
    try:
        info = await orchestrator.create_session(
            user_id=req.user_id,
            scenario_id=req.scenario_id,
            session_id=req.session_id,
        )
        return CreateSessionResponse(**info.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("create_session_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create session.")


@router.post("/sessions/message", response_model=SendMessageResponse, tags=["sessions"])
async def send_message(req: SendMessageRequest):
    """
    Send one student message and receive the avatar's response.
    Includes the per-turn evaluation score.
    """
    try:
        result = await orchestrator.send_message(
            session_id=req.session_id,
            user_id="",  # user_id resolved from session in real auth flow
            student_message=req.student_message,
        )
        score_resp = None
        if result.turn_score:
            score_resp = TurnScoreResponse(**result.turn_score)

        return SendMessageResponse(
            session_id=result.session_id,
            turn_index=result.turn_index,
            avatar_response=result.avatar_response,
            emotion=result.emotion,
            scenario_phase=result.scenario_phase,
            turn_score=score_resp,
            session_ended=result.session_ended,
            session_summary=result.session_summary,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error("send_message_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to process message.")


@router.get(
    "/sessions/{session_id}/scores",
    response_model=SessionScoresResponse,
    tags=["sessions"],
)
async def get_scores(session_id: str):
    """Return all per-turn evaluation scores accumulated so far."""
    scores = orchestrator.get_session_scores(session_id)
    return SessionScoresResponse(
        session_id=session_id,
        scores=[TurnScoreResponse(**s) for s in scores],
        turn_count=len(scores),
    )


@router.post("/sessions/{session_id}/end", response_model=EndSessionResponse, tags=["sessions"])
async def end_session(session_id: str):
    """
    Explicitly end a session.
    Flushes STM to episodic memory and returns the session evaluation summary.
    """
    result = await orchestrator.end_session(session_id)
    return EndSessionResponse(**result)
