"""Structured logging setup.

Call configure_logging() once at application startup (main / API lifespan).
All other modules should use:

    import logging
    logger = logging.getLogger(__name__)
"""

import logging
import sys
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Single-line JSON log records for production log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise *record* as a single-line JSON string."""
        import json
        import traceback

        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        if record.exc_info:
            payload["exc"] = traceback.format_exception(*record.exc_info)

        extra_keys = set(record.__dict__) - {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key in extra_keys:
            payload[key] = record.__dict__[key]

        return json.dumps(payload, default=str)


class _DevFormatter(logging.Formatter):
    """Coloured human-readable output for local development."""

    _COLOURS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Render *record* as a coloured single-line string for terminal output."""
        colour = self._COLOURS.get(record.levelname, "")
        prefix = f"{colour}{record.levelname:<8}{self._RESET}"
        return f"{prefix} {record.name}: {record.getMessage()}"


def configure_logging(level: str = "INFO", *, json: bool = False) -> None:
    """Configure the root logger.  Call once at startup.

    Args:
        level: Log level string (DEBUG / INFO / WARNING / ERROR).
        json:  Emit JSON lines instead of coloured text (set True in production).
    """
    formatter: logging.Formatter = _JsonFormatter() if json else _DevFormatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "insightface"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
