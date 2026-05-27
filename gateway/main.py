"""
gateway/main.py

ALAS API Gateway — FastAPI application.

Responsibilities:
  - Single process entry point for all client traffic
  - Auth middleware (JWT, demo-mode passthrough)
  - REST routing  → gateway/routers/sessions.py + auth.py
  - WebSocket     → gateway/routers/websocket.py
  - Session store wiring (in-memory or SQLite, selected by config)
  - Periodic session TTL cleanup
  - Structured JSON logging for every request

Ports (defaults, all overridable via env):
  Gateway :  8001   (this process)
  Agent   :  8000   (agent_service/main.py — separate process)

Run locally:
  cd alas/
  uvicorn gateway.main:app --port 8001 --reload
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from gateway.config import get_gateway_settings
from gateway.middleware.logging import RequestLoggingMiddleware
from gateway.routers.auth import router as auth_router
from gateway.routers.sessions import make_sessions_router
from gateway.routers.websocket import websocket_handler
from gateway.services.agent_client import AgentServiceClient
from gateway.session_store.store import create_session_store
from shared.utils.logging import get_logger

log      = get_logger(__name__)
settings = get_gateway_settings()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    # ── Shared singletons ─────────────────────────────────────────────────
    session_store  = create_session_store(
        persist=settings.persist_sessions,
        db_path=settings.session_db_path,
    )
    agent_client = AgentServiceClient(base_url=settings.agent_base_url)

    # ── Lifespan ──────────────────────────────────────────────────────────
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info(
            "gateway_startup",
            host=settings.host,
            port=settings.port,
            agent_url=settings.agent_base_url,
            require_auth=settings.require_auth,
            demo_mode=settings.demo_mode,
            persist_sessions=settings.persist_sessions,
        )

        # Background TTL cleanup — evicts DISCONNECTED sessions every 10 min
        cleanup_task = asyncio.create_task(_ttl_cleanup(session_store))

        yield

        cleanup_task.cancel()
        log.info("gateway_shutdown")

    # ── App ───────────────────────────────────────────────────────────────
    app = FastAPI(
        title="ALAS API Gateway",
        description=(
            "Integration layer between the frontend and the ALAS Agent Service. "
            "Handles auth, session lifecycle, WebSocket streaming, and routing."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    # ── REST routers ──────────────────────────────────────────────────────
    app.include_router(auth_router)
    app.include_router(make_sessions_router(session_store))

    # ── Gateway health (independent of agent service) ─────────────────────
    @app.get("/health", tags=["infra"])
    async def gateway_health():
        return {
            "status":       "ok",
            "service":      "gateway",
            "version":      "0.1.0",
            "agent_url":    settings.agent_base_url,
            "demo_mode":    settings.demo_mode,
            "require_auth": settings.require_auth,
        }

    @app.get("/health/agent", tags=["infra"])
    async def agent_health():
        """Probe the agent service and relay its health response."""
        try:
            result = await agent_client.health()
            return {"gateway": "ok", "agent": result}
        except Exception as e:
            return {"gateway": "ok", "agent": "unreachable", "error": str(e)}

    # ── WebSocket ─────────────────────────────────────────────────────────
    @app.websocket("/ws/{session_id}")
    async def ws_endpoint(
        websocket: WebSocket,
        session_id: str,
        token: str = "",
    ):
        await websocket_handler(
            websocket=websocket,
            session_id=session_id,
            token=token,
            session_store=session_store,
            agent_client=agent_client,
        )

    return app


# ---------------------------------------------------------------------------
# TTL cleanup background task
# ---------------------------------------------------------------------------

async def _ttl_cleanup(session_store) -> None:
    """Every 10 minutes, evict disconnected sessions older than TTL."""
    while True:
        await asyncio.sleep(600)
        try:
            removed = session_store.expire_old_sessions(
                ttl_seconds=settings.session_ttl_seconds
            )
            if removed:
                log.info("ttl_cleanup", sessions_removed=removed)
        except Exception as e:
            log.warning("ttl_cleanup_error", error=str(e))


# ---------------------------------------------------------------------------
# Module-level app instance + dev entrypoint
# ---------------------------------------------------------------------------

app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_config=None,
    )
