"""
gateway/auth/dependencies.py

FastAPI dependency functions for authentication.

Two variants:
  - require_token()    — for REST endpoints (reads Authorization: Bearer header)
  - extract_ws_token() — for WebSocket connections (reads ?token= query param
                         since WS handshakes don't support custom headers easily)

Both return the decoded token payload dict so downstream handlers can
access user_id and roles without re-verifying.
"""

from __future__ import annotations

from typing import Any

from fastapi import Header, HTTPException, Query, WebSocket, status

from gateway.auth.jwt_handler import verify_token
from shared.utils.logging import get_logger

log = get_logger(__name__)


async def require_token(
    authorization: str = Header(default=""),
) -> dict[str, Any]:
    """
    FastAPI dependency for REST endpoints.

    Expects:  Authorization: Bearer <token>
    Returns:  Decoded payload dict {"sub": user_id, "roles": [...], ...}
    Raises:   HTTP 401 on any auth failure.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = verify_token(token)
        log.debug("auth_ok", user_id=payload.get("sub"))
        return payload
    except ValueError as e:
        log.warning("auth_failed", reason=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def extract_ws_token(
    websocket: WebSocket,
    token: str = Query(default=""),
) -> dict[str, Any] | None:
    """
    Extract and verify a JWT from a WebSocket connection's query string.

    Usage:  ws://host/ws/{session_id}?token=<jwt>

    Returns: Decoded payload dict, or None if no token provided.
    Does NOT close the connection — the WS handler decides what to do
    when None is returned (send auth_error and close, or proceed as demo).
    """
    if not token:
        return None
    try:
        return verify_token(token)
    except ValueError as e:
        log.warning("ws_auth_failed", reason=str(e))
        return None


def get_user_id(payload: dict[str, Any]) -> str:
    """Extract user_id from a verified token payload."""
    return payload.get("sub", "")


def has_role(payload: dict[str, Any], role: str) -> bool:
    """Check if a token payload includes a given role."""
    return role in payload.get("roles", [])
