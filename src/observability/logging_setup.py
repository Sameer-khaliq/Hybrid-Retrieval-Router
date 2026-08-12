"""
Structured JSON logging skeleton (Step 4, NFR-11 / FR-20 foundation).

Every module from here forward should log through get_logger(trace_id=...)
from its very first line, so trace-ID-tagged structured logs exist from
day one instead of being retrofitted later (that retrofit is exactly the
rework this step exists to avoid).

Usage:
    from src.observability.logging_setup import get_logger
    log = get_logger(trace_id="abc123")
    log.info("smoke_test", stage="setup")
"""

import json
import logging
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats each log record as a single line of JSON to stdout."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # trace_id and any other structured kwargs passed via extra= get
        # attached directly onto the record by logging's `extra` mechanism -
        # pull them back out here so they land as top-level JSON fields
        # rather than nested inside the message string.
        reserved = set(logging.LogRecord(
            "", 0, "", 0, "", (), None
        ).__dict__.keys()) | {"message", "asctime"}
        for key, value in record.__dict__.items():
            if key not in reserved:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class TraceLoggerAdapter(logging.LoggerAdapter):
    """
    Thin adapter that guarantees trace_id (and any other bound fields)
    are attached to every log call made through it, without the caller
    having to pass extra= manually each time.
    """

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        # Allow ad-hoc structured fields passed as kwargs, e.g.
        # log.info("stage_done", stage="retrieval", latency_ms=42)
        structured = {
            k: v for k, v in kwargs.items()
            if k not in ("exc_info", "stack_info", "stacklevel", "extra")
        }
        extra.update(structured)
        kwargs["extra"] = extra
        for k in list(kwargs.keys()):
            if k not in ("exc_info", "stack_info", "stacklevel", "extra"):
                del kwargs[k]
        return msg, kwargs


_configured = False


def _configure_root_once() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.handlers = [handler]
    _configured = True


def get_logger(trace_id: str, **bound_fields: Any) -> TraceLoggerAdapter:
    """
    Returns a logger adapter that stamps every subsequent call with
    trace_id (and any other bound_fields, e.g. stage="ingestion").
    """
    _configure_root_once()
    base_logger = logging.getLogger("hybrid_retrieval")
    return TraceLoggerAdapter(base_logger, {"trace_id": trace_id, **bound_fields})