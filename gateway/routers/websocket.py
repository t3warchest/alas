"""
gateway/routers/websocket.py

WebSocket handler for the gateway.

Connection flow:
  1.  Client opens  ws://host:8001/ws/{session_id}?token=<jwt>
  2.  Gateway authenticates the token (or allows demo if REQUIRE_AUTH=false).
  3.  Gateway verifies the session_id exists in the session store.
  4.  Gateway sends  connected  event with avatar opening line.
  5.  Client sends   message    frames; gateway relays each through AgentServiceClient,
      streaming tokens back and emitting score / phase_change / session_end events.
  6.  Session ends on: client "end" frame, session_ended flag from agent, or disconnect.

WebSocket message protocol (shared/contracts/events.py and messages.py):

  Client → Gateway
    {"type": "message",   "content": "..."}
    {"type": "ping",      "timestamp": 1234}
    {"type": "end"}
    {"type": "reconnect", "session_id": "...", "last_turn_index": 3}

  Gateway → Client
    {"type": "auth_ok",      "user_id": "...",  "token_expires_in": 3600}
    {"type": "connected",    "session_id": "...", "opening_line": "...", ...}
    {"type": "token",        "content": "word",  "turn_index": 2}
    {"type": "turn_done",    "session_id": "...", "avatar_response": "...", "emotion": "...", ...}
    {"type": "score",        "turn_index": 2, "composite": 0.74, "rationale": "...", ...}
    {"type": "phase_change", "from_phase": "core", "to_phase": "escalation", ...}
    {"type": "session_end",  "session_id": "...", "summary": {...}}
    {"type": "pong",         "timestamp": 1234}
    {"type": "error",        "code": "...", "message": "...", "recoverable": true}
    {"type": "fatal",        "code": "...", "message": "..."}
"""

from __future__ import annotations

import json
import time
from typing import Any

try:
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError:
    WebSocket = object  # type: ignore
    class WebSocketDisconnect(Exception): pass  # type: ignore

from gateway.auth.jwt_handler import verify_token
from gateway.config import get_gateway_settings
from gateway.services.agent_client import AgentServiceClient, AgentServiceError
from gateway.session_store.store import SessionStore
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
    WS_MSG_END,
    WS_MSG_MESSAGE,
    WS_MSG_PING,
    WS_MSG_RECONNECT,
)
from shared.contracts.messages import parse_client_frame
from shared.contracts.session import SessionStatus
from shared.utils.logging import get_logger

log = get_logger(__name__)
_settings = get_gateway_settings()


# ---------------------------------------------------------------------------
# Public entry point — called from gateway/main.py
# ---------------------------------------------------------------------------

async def websocket_handler(
    websocket: WebSocket,
    session_id: str,
    token: str,
    session_store: SessionStore,
    agent_client: AgentServiceClient,
) -> None:
    """
    Main WebSocket dispatch loop.

    Accepts the connection, authenticates, then enters the message loop.
    All errors are caught and converted to structured error events before
    closing — the client always receives an explicit close reason.
    """
    await websocket.accept()
    log.info("ws_connect", session_id=session_id)

    # ── Step 1: Authenticate ──────────────────────────────────────────────
    user_id = await _authenticate(websocket, token, session_id)
    if user_id is None:
        return  # _authenticate already sent fatal + closed

    # ── Step 2: Validate session ──────────────────────────────────────────
    session = session_store.get(session_id)
    if session is None:
        await _fatal(websocket, "session_not_found",
                     f"Session '{session_id}' not found. Call POST /api/sessions first.")
        return

    if session.status == SessionStatus.ENDED:
        await _fatal(websocket, "session_ended",
                     "This session has already ended.")
        return

    # ── Step 3: Reconnect vs fresh connect ────────────────────────────────
    is_reconnect = session.status == SessionStatus.DISCONNECTED
    session_store.update_status(session_id, SessionStatus.ACTIVE)

    # ── Step 4: Send auth_ok + connected ─────────────────────────────────
    await _send(websocket, {
        "type":             WS_EVT_AUTH_OK,
        "user_id":          user_id,
        "token_expires_in": _settings.jwt_expiry_seconds,
    })

    await _send(websocket, {
        "type":           WS_EVT_CONNECTED,
        "session_id":     session_id,
        "scenario_id":    session.scenario_id,
        "scenario_title": session.scenario_title,
        "persona_name":   session.persona_name,
        "opening_line":   session.opening_line,
        "phase":          session.current_phase,
        "reconnected":    is_reconnect,
        "last_turn_index": session.last_turn_index if is_reconnect else 0,
    })

    log.info("ws_ready", session_id=session_id, user_id=user_id, reconnect=is_reconnect)

    # ── Step 5: Message loop ──────────────────────────────────────────────
    try:
        await _message_loop(websocket, session_id, user_id, session_store, agent_client)
    except WebSocketDisconnect:
        log.info("ws_disconnect", session_id=session_id)
        session_store.update_status(session_id, SessionStatus.DISCONNECTED)
    except Exception as exc:
        log.error("ws_unhandled_error", session_id=session_id, error=str(exc))
        session_store.update_status(session_id, SessionStatus.ERROR)
        try:
            await _fatal(websocket, "internal_error", "Unexpected server error.")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Message loop
# ---------------------------------------------------------------------------

async def _message_loop(
    websocket: WebSocket,
    session_id: str,
    user_id: str,
    session_store: SessionStore,
    agent_client: AgentServiceClient,
) -> None:
    """
    Reads client frames in a loop, dispatches each to the right handler.
    Exits cleanly when the session ends or the client disconnects.
    """
    while True:
        raw = await websocket.receive_text()

        frame = parse_client_frame(raw)
        if frame is None:
            await _error(websocket, "parse_error", "Message must be valid JSON.")
            continue

        event_type = frame.type

        # ── Heartbeat ─────────────────────────────────────────────────────
        if event_type == WS_MSG_PING:
            ts = getattr(frame, "timestamp", time.time())
            await _send(websocket, {"type": WS_EVT_PONG, "timestamp": ts})
            continue

        # ── Explicit end ──────────────────────────────────────────────────
        if event_type == WS_MSG_END:
            await _handle_end(websocket, session_id, session_store, agent_client)
            break

        # ── Reconnect ─────────────────────────────────────────────────────
        if event_type == WS_MSG_RECONNECT:
            last_turn = getattr(frame, "last_turn_index", 0)
            session = session_store.get(session_id)
            current_turn = session.last_turn_index if session else 0
            missed = max(0, current_turn - last_turn)
            await _send(websocket, {
                "type":            WS_EVT_CONNECTED,
                "session_id":      session_id,
                "reconnected":     True,
                "current_phase":   session.current_phase if session else "unknown",
                "last_turn_index": current_turn,
                "missed_turns":    missed,
            })
            continue

        # ── Student message ───────────────────────────────────────────────
        if event_type == WS_MSG_MESSAGE:
            content = getattr(frame, "content", "").strip()
            if not content:
                await _error(websocket, "empty_message", "Message content cannot be empty.")
                continue

            ended = await _handle_message(
                websocket, session_id, content, session_store, agent_client
            )
            if ended:
                break
            continue

        # ── Unknown ───────────────────────────────────────────────────────
        await _error(websocket, "unknown_event",
                     f"Unknown event type: '{event_type}'. "
                     f"Expected: message | ping | end | reconnect.")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def _handle_message(
    websocket: WebSocket,
    session_id: str,
    content: str,
    session_store: SessionStore,
    agent_client: AgentServiceClient,
) -> bool:
    """
    Route one student message through the agent service, stream the response.
    Returns True if the session ended this turn.
    """
    session = session_store.get(session_id)
    prev_phase = session.current_phase if session else "setup"

    session_ended = False

    try:
        async for event in agent_client.stream_turn(session_id, content):
            etype = event.get("type")

            if etype == WS_EVT_TOKEN:
                await _send(websocket, event)

            elif etype == WS_EVT_TURN_DONE:
                new_phase = event.get("phase", prev_phase)
                turn_idx  = event.get("turn_index", 0)
                session_ended = event.get("session_ended", False)

                await _send(websocket, event)

                # Emit phase_change if phase shifted
                if new_phase and new_phase != prev_phase:
                    await _send(websocket, {
                        "type":       WS_EVT_PHASE_CHANGE,
                        "from_phase": prev_phase,
                        "to_phase":   new_phase,
                        "turn_index": turn_idx,
                        "reason":     "branching_engine",
                    })
                    prev_phase = new_phase

                # Update session store
                session_store.update_turn(
                    session_id,
                    turn_index=turn_idx,
                    phase=new_phase,
                )

            elif etype == WS_EVT_SCORE:
                composite = event.get("composite")
                turn_idx  = event.get("turn_index", 0)
                await _send(websocket, event)
                session_store.update_turn(
                    session_id,
                    turn_index=turn_idx,
                    composite=composite,
                )

            elif etype == WS_EVT_SESSION_END:
                await _send(websocket, event)
                session_store.update_status(session_id, SessionStatus.ENDED)
                session_ended = True

            elif etype == "error":
                await _error(websocket, event.get("code", "agent_error"),
                             event.get("message", "Agent service error."))
                # recoverable — don't end session

    except AgentServiceError as e:
        await _error(websocket, "agent_unreachable", str(e))

    return session_ended


async def _handle_end(
    websocket: WebSocket,
    session_id: str,
    session_store: SessionStore,
    agent_client: AgentServiceClient,
) -> None:
    """Client explicitly ended the session."""
    log.info("ws_end_requested", session_id=session_id)
    try:
        result = await agent_client.end_session(session_id)
    except AgentServiceError as e:
        result = {"error": str(e)}

    session_store.update_status(session_id, SessionStatus.ENDED)
    await _send(websocket, {
        "type":       WS_EVT_SESSION_END,
        "session_id": session_id,
        "summary":    result.get("summary", {}),
    })


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

async def _authenticate(
    websocket: WebSocket,
    token: str,
    session_id: str,
) -> str | None:
    """
    Verify the token. Returns user_id on success, None on failure.
    In demo mode (REQUIRE_AUTH=false), missing tokens are accepted with a
    synthetic user_id derived from the session_id.
    """
    if not _settings.require_auth:
        if not token:
            # Demo mode: no token needed
            log.debug("ws_auth_skipped", session_id=session_id, reason="demo_mode")
            return f"demo-{session_id[:8]}"

    if not token:
        await _fatal(websocket, "missing_token",
                     "Authentication required. Connect with ?token=<jwt>")
        return None

    try:
        payload = verify_token(token)
        user_id = payload.get("sub", "")
        log.debug("ws_auth_ok", session_id=session_id, user_id=user_id)
        return user_id
    except ValueError as e:
        await _fatal(websocket, "invalid_token", str(e))
        return None


# ---------------------------------------------------------------------------
# Send helpers
# ---------------------------------------------------------------------------

async def _send(ws: WebSocket, payload: dict[str, Any]) -> None:
    """Serialise and send one event. Silently absorbs post-disconnect errors."""
    try:
        await ws.send_text(json.dumps(payload, default=str))
    except Exception:
        pass


async def _error(ws: WebSocket, code: str, message: str) -> None:
    """Send a recoverable error event."""
    await _send(ws, {
        "type":        WS_EVT_ERROR,
        "code":        code,
        "message":     message,
        "recoverable": True,
    })


async def _fatal(ws: WebSocket, code: str, message: str) -> None:
    """Send a non-recoverable error and close the connection."""
    await _send(ws, {
        "type":        WS_EVT_FATAL,
        "code":        code,
        "message":     message,
        "recoverable": False,
    })
    try:
        await ws.close(code=1008)  # 1008 = Policy Violation
    except Exception:
        pass
