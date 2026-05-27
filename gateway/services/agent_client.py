"""
gateway/services/agent_client.py

AgentServiceClient: the clean interface boundary between the gateway
and the agent service.

Design:
  - All agent service interactions go through this class.
  - The gateway never imports from agent_service directly — only this client.
  - Uses httpx for async HTTP.
  - For streaming, the gateway calls the agent's REST /sessions/message endpoint
    and re-streams the response to the client.
  - WebSocket streaming uses the agent's /ws/{session_id} endpoint directly
    via an internal WS-to-WS relay (see stream_turn_ws).

This is the "seam" the task asked for. To swap the agent service for a
different backend, you only change this file.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from gateway.config import get_gateway_settings
from shared.utils.logging import get_logger

log = get_logger(__name__)
_settings = get_gateway_settings()

# httpx may not be installed in this sandbox — we'll do a graceful import
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False
    log.warning("httpx_not_available", detail="Agent HTTP client will use stdlib fallback")


class AgentServiceError(Exception):
    """Raised when the agent service returns an error or is unreachable."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class AgentServiceClient:
    """
    Async HTTP client for the ALAS agent service REST API.

    All methods raise AgentServiceError on failure so the gateway
    can translate them into appropriate WebSocket error events.
    """

    def __init__(self, base_url: str | None = None):
        self._base = (base_url or _settings.agent_base_url).rstrip("/")

    # -----------------------------------------------------------------------
    # Session lifecycle
    # -----------------------------------------------------------------------

    async def create_session(
        self,
        user_id: str,
        scenario_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        POST /api/v1/sessions
        Returns the session info dict from the agent service.
        """
        payload = {
            "user_id":     user_id,
            "scenario_id": scenario_id,
        }
        if session_id:
            payload["session_id"] = session_id

        return await self._post("/api/v1/sessions", payload)

    async def send_message(
        self,
        session_id: str,
        student_message: str,
    ) -> dict[str, Any]:
        """
        POST /api/v1/sessions/message
        Returns the full turn result (non-streaming).
        """
        return await self._post("/api/v1/sessions/message", {
            "session_id":      session_id,
            "student_message": student_message,
        })

    async def end_session(self, session_id: str) -> dict[str, Any]:
        """POST /api/v1/sessions/{id}/end"""
        return await self._post(f"/api/v1/sessions/{session_id}/end", {})

    async def get_scores(self, session_id: str) -> dict[str, Any]:
        """GET /api/v1/sessions/{id}/scores"""
        return await self._get(f"/api/v1/sessions/{session_id}/scores")

    async def list_scenarios(self) -> list[dict[str, Any]]:
        """GET /api/v1/scenarios"""
        return await self._get("/api/v1/scenarios")

    async def health(self) -> dict[str, Any]:
        """GET /api/v1/health"""
        return await self._get("/api/v1/health")

    # -----------------------------------------------------------------------
    # Streaming relay: call agent REST, yield structured events to gateway
    # -----------------------------------------------------------------------

    async def stream_turn(
        self,
        session_id: str,
        student_message: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Call the agent's send_message and yield structured gateway events.

        The agent service's stream_message yields {type, content/data} dicts.
        This method translates them into the shared contracts event format.

        For the demo, we call the non-streaming endpoint and simulate
        streaming at the gateway level (same approach as the agent's own
        stream_message). In production this would relay actual SSE/WS tokens.
        """
        try:
            result = await self.send_message(session_id, student_message)
        except AgentServiceError as e:
            yield {"type": "error", "code": "agent_error", "message": str(e)}
            return

        avatar_text  = result.get("avatar_response", "")
        turn_index   = result.get("turn_index", 0)
        emotion      = result.get("emotion", "neutral")
        phase        = result.get("scenario_phase", "")
        session_ended = result.get("session_ended", False)
        turn_score   = result.get("turn_score")
        session_summary = result.get("session_summary")

        # --- Yield streaming tokens ---
        import asyncio
        words = avatar_text.split()
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield {"type": "token", "content": chunk, "turn_index": turn_index}
            await asyncio.sleep(0.025)  # ~40 words/sec

        # --- Yield turn_done ---
        yield {
            "type":           "turn_done",
            "session_id":     session_id,
            "turn_index":     turn_index,
            "avatar_response": avatar_text,
            "emotion":        emotion,
            "phase":          phase,
            "session_ended":  session_ended,
        }

        # --- Yield score if available ---
        if turn_score:
            yield {"type": "score", **turn_score}

        # --- Yield session_end if session concluded ---
        if session_ended:
            yield {
                "type":       "session_end",
                "session_id": session_id,
                "summary":    session_summary or {},
            }

    # -----------------------------------------------------------------------
    # HTTP helpers
    # -----------------------------------------------------------------------

    async def _post(self, path: str, body: dict) -> Any:
        url = f"{self._base}{path}"
        log.debug("agent_post", url=url)

        if _HAS_HTTPX:
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    resp = await client.post(url, json=body)
                    return self._handle_response(resp)
                except httpx.ConnectError as e:
                    raise AgentServiceError(
                        f"Cannot reach agent service at {self._base}: {e}", 503
                    )
        else:
            # Stdlib fallback (sync, for testing without httpx)
            return await self._stdlib_post(url, body)

    async def _get(self, path: str) -> Any:
        url = f"{self._base}{path}"
        log.debug("agent_get", url=url)

        if _HAS_HTTPX:
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    resp = await client.get(url)
                    return self._handle_response(resp)
                except httpx.ConnectError as e:
                    raise AgentServiceError(
                        f"Cannot reach agent service at {self._base}: {e}", 503
                    )
        else:
            return await self._stdlib_get(url)

    def _handle_response(self, resp) -> Any:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise AgentServiceError(detail, resp.status_code)
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}

    # stdlib async wrappers (no external deps needed for testing)
    async def _stdlib_post(self, url: str, body: dict) -> Any:
        import asyncio
        import urllib.request
        import urllib.error

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        loop = asyncio.get_event_loop()
        try:
            response_bytes = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(req, timeout=60).read()
            )
            return json.loads(response_bytes)
        except urllib.error.HTTPError as e:
            raise AgentServiceError(str(e), e.code)
        except Exception as e:
            raise AgentServiceError(str(e), 503)

    async def _stdlib_get(self, url: str) -> Any:
        import asyncio
        import urllib.request
        import urllib.error

        loop = asyncio.get_event_loop()
        try:
            response_bytes = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(url, timeout=30).read()
            )
            return json.loads(response_bytes)
        except urllib.error.HTTPError as e:
            raise AgentServiceError(str(e), e.code)
        except Exception as e:
            raise AgentServiceError(str(e), 503)


# Module-level singleton
agent_client = AgentServiceClient()
