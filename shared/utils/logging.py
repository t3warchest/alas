"""
shared/utils/logging.py

Structured logger used by both gateway and agent service.
Every record is a single JSON line.

Usage:
    log = get_logger(__name__)
    log.info("session_created", extra={"session_id": "abc", "user_id": "u1"})
    # or the shorthand used throughout the codebase:
    log.info("session_created", **{"session_id": "abc"})   ← wrapped below

We wrap the standard logger so callers can write:
    log.info("event_name", session_id="abc", user_id="u1")
instead of:
    log.info("event_name", extra={"session_id": "abc", "user_id": "u1"})
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_SKIP = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text",
    "filename", "funcName", "id", "levelname", "levelno",
    "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated",
    "stack_info", "thread", "threadName",
})


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in _SKIP:
                base[key] = val
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str)


class _StructuredLogger:
    """
    Thin wrapper that lets callers pass keyword arguments directly:
        log.info("event", session_id="abc")
    instead of the stdlib:
        log.info("event", extra={"session_id": "abc"})
    """

    def __init__(self, logger: logging.Logger):
        self._log = logger

    def _emit(self, level: int, msg: str, **kwargs: Any) -> None:
        self._log.log(level, msg, extra=kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        self._log.exception(msg, extra=kwargs)

    # Pass-through so isinstance checks and library integrations work
    @property
    def name(self) -> str:
        return self._log.name

    @property
    def handlers(self):
        return self._log.handlers


_wrapper_cache: dict[str, _StructuredLogger] = {}

def get_logger(name: str) -> _StructuredLogger:
    if name in _wrapper_cache:
        return _wrapper_cache[name]
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    wrapper = _StructuredLogger(logger)
    _wrapper_cache[name] = wrapper
    return wrapper
