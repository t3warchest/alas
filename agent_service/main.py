"""
alas/agent_service/main.py

FastAPI application entrypoint.

Mounts:
  REST  — /api/v1  (routes.py)
  WS    — /ws/{session_id}  (websocket.py)

Lifespan:
  startup  — ingest scenario knowledge into ChromaDB
  shutdown — nothing to clean up in this prototype
"""

from __future__ import annotations

# Load .env FIRST — before any module that reads os.getenv() at import time
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from agent_service.api.routes import router
from agent_service.api.websocket import websocket_session
from agent_service.session import startup
from agent_service.utils.config import get_settings
from agent_service.utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    log.info(f"app_startup host={settings.host} port={settings.port}")
    startup()
    yield
    # ---- shutdown ----
    log.info("app_shutdown")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="ALAS Agent Service",
        description=(
            "AI-powered conversational simulation platform. "
            "Implements LangGraph orchestration, multi-layer memory, "
            "and real-time rubric evaluation."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — wide open for demo; lock down in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # REST routes
    app.include_router(router, prefix="/api/v1")

    # WebSocket endpoint
    @app.websocket("/ws/{session_id}")
    async def ws_endpoint(websocket: WebSocket, session_id: str, user_id: str = ""):
        await websocket_session(websocket, session_id, user_id)

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "agent_service.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_config=None,  # use our structured logger
    )