"""
gateway/routers/auth.py

Authentication endpoints.

POST /auth/token    — demo token issuance (DEMO_MODE=true only)
GET  /auth/verify   — verify a token and return its claims (useful for frontend)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from gateway.auth.dependencies import require_token
from gateway.auth.jwt_handler import issue_demo_token, verify_token
from shared.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    user_id: str
    scenario_id: str = ""


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_id: str


@router.post("/token", response_model=TokenResponse)
async def get_demo_token(req: TokenRequest):
    """
    Issue a demo JWT. Only works when DEMO_MODE=true.
    In production, redirect clients to your SSO/OAuth2 provider instead.
    """
    try:
        result = issue_demo_token(user_id=req.user_id, scenario_id=req.scenario_id)
        log.info("demo_token_issued", user_id=req.user_id)
        return TokenResponse(**result)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/verify")
async def verify_token_endpoint(payload: dict = Depends(require_token)):
    """
    Verify the token in the Authorization header and return its claims.
    Useful for the frontend to confirm a token is still valid.
    """
    return {
        "valid": True,
        "user_id": payload.get("sub"),
        "roles": payload.get("roles", []),
        "expires_at": payload.get("exp"),
    }
