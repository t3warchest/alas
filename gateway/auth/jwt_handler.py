"""
gateway/auth/jwt_handler.py

Demo-grade JWT authentication.

Design decisions:
  - Uses HS256 (HMAC-SHA256) with a shared secret from config.
    Production would use RS256 with a proper key pair.
  - Token payload carries: sub (user_id), roles, exp, iat, jti (unique ID).
  - Verification is strict: checks signature, expiry, and required claims.
  - A demo `issue_demo_token()` function lets the frontend get a token
    without a real identity provider — remove this for production.

No pydantic needed. Pure stdlib + python-jose (jose).
Falls back gracefully if jose is unavailable (test environments).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
import base64
from typing import Any

from gateway.config import get_gateway_settings
from shared.utils.logging import get_logger

log = get_logger(__name__)
_settings = get_gateway_settings()


# ---------------------------------------------------------------------------
# Minimal pure-stdlib JWT (HS256) — no external dependency
# This is intentionally simple for a demo. Use python-jose in production.
# ---------------------------------------------------------------------------

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    # Re-add padding
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _sign(header_b64: str, payload_b64: str, secret: str) -> str:
    msg = f"{header_b64}.{payload_b64}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).digest()
    return _b64url_encode(sig)


def create_token(
    user_id: str,
    roles: list[str] | None = None,
    expires_in: int | None = None,
) -> str:
    """
    Create a signed JWT for the given user.

    Args:
        user_id:    Opaque user identifier (sub claim).
        roles:      List of role strings (default: ["student"]).
        expires_in: Token lifetime in seconds (default: from settings).

    Returns:
        A signed JWT string.
    """
    exp_seconds = expires_in or _settings.jwt_expiry_seconds
    now = int(time.time())

    header  = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub":   user_id,
        "roles": roles or ["student"],
        "iat":   now,
        "exp":   now + exp_seconds,
        "jti":   str(uuid.uuid4()),
    }

    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    s = _sign(h, p, _settings.jwt_secret)

    return f"{h}.{p}.{s}"


def verify_token(token: str) -> dict[str, Any]:
    """
    Verify and decode a JWT.

    Returns the decoded payload dict on success.
    Raises ValueError on any failure (expired, bad signature, malformed).
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Malformed token: expected 3 dot-separated parts")

        h_b64, p_b64, given_sig = parts

        # Verify signature
        expected_sig = _sign(h_b64, p_b64, _settings.jwt_secret)
        if not hmac.compare_digest(given_sig, expected_sig):
            raise ValueError("Invalid token signature")

        # Decode payload
        payload = json.loads(_b64url_decode(p_b64))

        # Check expiry
        exp = payload.get("exp", 0)
        if int(time.time()) > exp:
            raise ValueError(f"Token expired at {exp}")

        # Check required claims
        if "sub" not in payload:
            raise ValueError("Token missing 'sub' claim")

        return payload

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Token verification failed: {e}") from e


def extract_user_id(token: str) -> str:
    """Convenience: verify token and return the user_id (sub claim)."""
    payload = verify_token(token)
    return payload["sub"]


# ---------------------------------------------------------------------------
# Demo-only: issue a token via a simple REST endpoint
# ---------------------------------------------------------------------------

def issue_demo_token(user_id: str, scenario_id: str = "") -> dict[str, Any]:
    """
    Issue a demo token without any password check.
    For a real system, this would be replaced by an OAuth2 / SSO flow.
    Only enabled when DEMO_MODE=true in settings.
    """
    if not _settings.demo_mode:
        raise PermissionError("Demo token issuance is disabled (DEMO_MODE=false)")

    token = create_token(user_id=user_id, roles=["student"])
    return {
        "access_token": token,
        "token_type":   "bearer",
        "expires_in":   _settings.jwt_expiry_seconds,
        "user_id":      user_id,
    }
