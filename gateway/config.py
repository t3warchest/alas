"""
gateway/config.py

All gateway configuration, loaded from environment / .env file.
Nothing is hardcoded. Sane defaults make local dev work with zero setup.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in ("1", "true", "yes")

def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default

def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


@dataclass(frozen=True)
class GatewaySettings:
    # Network
    host:        str = field(default_factory=lambda: _env("GATEWAY_HOST", "0.0.0.0"))
    port:        int = field(default_factory=lambda: _env_int("GATEWAY_PORT", 8001))

    # Agent service endpoint (the existing FastAPI service)
    agent_base_url: str = field(default_factory=lambda: _env("AGENT_SERVICE_URL", "http://localhost:8000"))
    agent_ws_url:   str = field(default_factory=lambda: _env("AGENT_SERVICE_WS_URL", "ws://localhost:8000"))

    # Public URL advertised to clients for WebSocket connections
    # Set GATEWAY_PUBLIC_URL to your deployed domain (e.g. https://myserver.com)
    # when running behind a reverse proxy or on a cloud host.
    # Defaults to the local address so development works out of the box.
    gateway_public_url: str = field(default_factory=lambda: _env(
        "GATEWAY_PUBLIC_URL",
        f"http://localhost:{_env_int('GATEWAY_PORT', 8001)}",
    ))

    # Auth
    jwt_secret:         str  = field(default_factory=lambda: _env("JWT_SECRET", secrets.token_hex(32)))
    jwt_expiry_seconds: int  = field(default_factory=lambda: _env_int("JWT_EXPIRY_SECONDS", 3600))
    demo_mode:          bool = field(default_factory=lambda: _env_bool("DEMO_MODE", True))

    # CORS — GitHub Pages origin always allowed; extend via CORS_ORIGINS env var
    cors_origins: list[str] = field(default_factory=lambda: (
        list({
            "https://t3warchest.github.io",
            *_env("CORS_ORIGINS", "*").split(","),
        })
    ))

    # Session persistence
    persist_sessions:    bool = field(default_factory=lambda: _env_bool("PERSIST_SESSIONS", False))
    session_db_path:     str  = field(default_factory=lambda: _env("SESSION_DB_PATH", "./data/gateway_sessions.db"))
    session_ttl_seconds: int  = field(default_factory=lambda: _env_int("SESSION_TTL_SECONDS", 3600))

    # Logging
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # Feature flags
    require_auth:   bool = field(default_factory=lambda: _env_bool("REQUIRE_AUTH", False))
    # When False, unauthenticated connections are allowed (demo-friendly).
    # When True, every WS connection must carry a valid ?token= query param.


@lru_cache(maxsize=1)
def get_gateway_settings() -> GatewaySettings:
    return GatewaySettings()
