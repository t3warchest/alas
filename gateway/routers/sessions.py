"""
gateway/routers/sessions.py

Session management REST endpoints.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from gateway.auth.dependencies import get_user_id, require_token
from gateway.auth.jwt_handler import verify_token
from gateway.config import get_gateway_settings
from gateway.services.agent_client import AgentServiceError, agent_client
from gateway.session_store.store import SessionStore
from shared.contracts.session import GatewaySession, SessionStatus
from shared.utils.logging import get_logger

log = get_logger(__name__)
_settings = get_gateway_settings()


# ── Pydantic I/O models ───────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    scenario_id: str = Field(..., description="Scenario ID from GET /api/scenarios")
    user_id: str = Field(default="demo-user", description="User identifier")
    session_id: str | None = Field(default=None)


class CreateSessionResponse(BaseModel):
    session_id: str
    scenario_id: str
    scenario_title: str
    persona_name: str
    opening_line: str
    ws_url: str


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    scenario_id: str
    scenario_title: str
    current_phase: str
    total_turns: int
    latest_composite: float | None


# ── Helper: extract user_id from request (token or fallback) ─────────────

def _resolve_user(request: Request) -> str:
    """Try to get user_id from JWT; fall back to demo-user if auth not required."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1]
        try:
            payload = verify_token(token)
            return payload.get("sub", "demo-user")
        except Exception:
            pass
    return "demo-user"


# ── Router factory ────────────────────────────────────────────────────────

def make_sessions_router(session_store: SessionStore) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["sessions"])

    @router.get("/scenarios")
    async def list_scenarios():
        """Proxy the agent service's scenario list."""
        try:
            return await agent_client.list_scenarios()
        except AgentServiceError as e:
            raise HTTPException(status_code=e.status_code, detail=str(e))

    @router.post(
        "/sessions",
        response_model=CreateSessionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_session(req: CreateSessionRequest, request: Request):
        user_id = _resolve_user(request) if _settings.require_auth else (req.user_id or "demo-user")
        session_id = req.session_id or str(uuid.uuid4())

        log.info("create_session", user_id=user_id, scenario_id=req.scenario_id, session_id=session_id)

        try:
            agent_resp = await agent_client.create_session(
                user_id=user_id,
                scenario_id=req.scenario_id,
                session_id=session_id,
            )
        except AgentServiceError as e:
            log.error("agent_create_session_failed", error=str(e))
            raise HTTPException(status_code=e.status_code, detail=str(e))

        gw_session = GatewaySession(
            session_id=session_id,
            user_id=user_id,
            scenario_id=req.scenario_id,
            status=SessionStatus.INITIALISING,
            scenario_title=agent_resp.get("scenario_title", ""),
            persona_name=agent_resp.get("persona_name", ""),
            opening_line=agent_resp.get("opening_line", ""),
        )
        session_store.create(gw_session)

        ws_url = f"ws://localhost:{_settings.port}/ws/{session_id}"

        return CreateSessionResponse(
            session_id=session_id,
            scenario_id=req.scenario_id,
            scenario_title=agent_resp.get("scenario_title", ""),
            persona_name=agent_resp.get("persona_name", ""),
            opening_line=agent_resp.get("opening_line", ""),
            ws_url=ws_url,
        )

    @router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
    async def get_session(session_id: str):
        s = session_store.get(session_id)
        if not s:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        return SessionStatusResponse(
            session_id=s.session_id,
            status=s.status.value,
            scenario_id=s.scenario_id,
            scenario_title=s.scenario_title,
            current_phase=s.current_phase,
            total_turns=s.total_turns,
            latest_composite=s.latest_composite,
        )

    @router.get("/sessions/{session_id}/scores")
    async def get_scores(session_id: str):
        s = session_store.get(session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found.")
        try:
            return await agent_client.get_scores(session_id)
        except AgentServiceError as e:
            raise HTTPException(status_code=e.status_code, detail=str(e))

    @router.post("/sessions/{session_id}/end")
    async def end_session(session_id: str):
        s = session_store.get(session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found.")
        try:
            result = await agent_client.end_session(session_id)
        except AgentServiceError as e:
            result = {"error": str(e)}
        session_store.update_status(session_id, SessionStatus.ENDED)
        return {"session_id": session_id, "status": "ended", **result}

    @router.get("/sessions")
    async def list_sessions(request: Request):
        user_id = _resolve_user(request)
        sessions = session_store.list_for_user(user_id)
        return [s.to_dict() for s in sessions]

    return router
