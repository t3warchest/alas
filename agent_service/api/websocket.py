"""
alas/agent_service/api/websocket.py

WebSocket endpoint for real-time streaming conversations.

Protocol:
  Client → Server:  JSON  {"type": "message", "content": "<student text>"}
                    JSON  {"type": "end"}
                    JSON  {"type": "ping"}

  Server → Client:  JSON  {"type": "token",   "content": "<word>"}         streaming token
                    JSON  {"type": "result",  "data": { TurnResult }}       turn complete
                    JSON  {"type": "score",   "data": { TurnScore }}        eval score
                    JSON  {"type": "session_end", "data": { summary }}      session over
                    JSON  {"type": "error",   "message": "..."}             recoverable error
                    JSON  {"type": "pong"}                                   heartbeat reply

Connection lifecycle:
  1. Client connects to /ws/{session_id}?user_id={user_id}
  2. Server streams avatar responses turn by turn
  3. Client sends "end" or session exit conditions are met → server closes
"""

from __future__ import annotations

import json

from fastapi import WebSocket, WebSocketDisconnect

from agent_service.session import orchestrator
from agent_service.utils.logging import get_logger

log = get_logger(__name__)


async def websocket_session(websocket: WebSocket, session_id: str, user_id: str = ""):
    """
    Main WebSocket handler. Mounted at /ws/{session_id}.
    Manages the full duplex streaming loop for one session.
    """
    await websocket.accept()
    log.info("ws_connected", session_id=session_id, user_id=user_id)

    try:
        while True:
            # Wait for student message
            raw = await websocket.receive_text()

            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await _send(websocket, {"type": "error", "message": "Invalid JSON."})
                continue

            event_type = event.get("type", "message")

            # --- Heartbeat ---
            if event_type == "ping":
                await _send(websocket, {"type": "pong"})
                continue

            # --- Explicit end ---
            if event_type == "end":
                summary = await orchestrator.end_session(session_id)
                await _send(websocket, {"type": "session_end", "data": summary})
                break

            # --- Student message ---
            if event_type == "message":
                content = event.get("content", "").strip()
                if not content:
                    await _send(websocket, {"type": "error", "message": "Empty message."})
                    continue

                # Stream tokens as they arrive
                session_ended = False
                async for chunk in orchestrator.stream_message(
                    session_id=session_id,
                    user_id=user_id,
                    student_message=content,
                ):
                    await _send(websocket, chunk)

                    if chunk.get("type") == "result":
                        data = chunk.get("data", {})
                        # Forward the score as a separate event for easy client handling
                        if data.get("turn_score"):
                            await _send(websocket, {
                                "type": "score",
                                "data": data["turn_score"],
                            })
                        session_ended = data.get("session_ended", False)

                    if chunk.get("type") == "error":
                        break  # do not close — let client retry

                if session_ended:
                    await _send(websocket, {
                        "type": "session_end",
                        "data": {"session_id": session_id},
                    })
                    break

            else:
                await _send(websocket, {
                    "type": "error",
                    "message": f"Unknown event type: {event_type!r}",
                })

    except WebSocketDisconnect:
        log.info("ws_disconnected", session_id=session_id)
    except Exception as e:
        log.error("ws_error", session_id=session_id, error=str(e))
        try:
            await _send(websocket, {"type": "error", "message": "Internal server error."})
        except Exception:
            pass
    finally:
        log.info("ws_closed", session_id=session_id)


async def _send(ws: WebSocket, payload: dict) -> None:
    """JSON-encode and send a message, swallowing send errors after disconnect."""
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass
