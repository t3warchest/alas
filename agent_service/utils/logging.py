"""
alas/agent_service/utils/logging.py

Structured logging setup. Every log record is a JSON line so it can be
shipped to any log aggregator without additional parsing.

Usage:
    from agent_service.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("session_started", session_id="abc", user_id="u1")
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
    """Wraps stdlib Logger so callers can write log.info("event", key=val)."""

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

    @property
    def name(self) -> str:
        return self._log.name

    @property
    def handlers(self):
        return self._log.handlers


_cache: dict[str, _StructuredLogger] = {}


def get_logger(name: str) -> _StructuredLogger:
    if name in _cache:
        return _cache[name]
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    wrapper = _StructuredLogger(logger)
    _cache[name] = wrapper
    return wrapper