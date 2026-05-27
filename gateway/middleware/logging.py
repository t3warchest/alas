"""
gateway/middleware/logging.py

Request logging middleware.

Every HTTP request gets a unique request_id attached to the context.
The structured access log line includes: method, path, status, duration_ms,
user_id (if authenticated), and request_id.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from shared.utils.logging import get_logger

log = get_logger("gateway.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs every HTTP request as a structured JSON line.

    Attaches request_id to request.state so downstream handlers
    can include it in their own logs for correlation.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        start = time.perf_counter()

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start) * 1000, 1)

            log.info(
                "http_request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration_ms,
                client=request.client.host if request.client else "unknown",
            )
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            log.error(
                "http_error",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise
