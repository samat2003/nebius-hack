"""Structured logging with automatic secret redaction and request/trace correlation.

Every request-level log entry includes:
- `request_id`: individual HTTP attempt ID
- `operation_id`: logical idempotent operation ID
- `trace_id`: application correlation trace ID
- `component`: module/subsystem identifier
- `event`: action/event name
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

from alienese.contracts.context import RequestContext
from alienese.observability.redaction import redact_mapping


def _redact_processor(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Structlog processor that redacts sensitive keys and values before emission."""
    return redact_mapping(event_dict)


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON structured logging with mandatory redaction."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_request_logger(ctx: RequestContext, component: str) -> Any:
    """Return a structured logger bound to the request context and component name."""
    logger = structlog.get_logger()
    return logger.bind(
        request_id=ctx.request_id,
        operation_id=ctx.operation_id,
        trace_id=ctx.correlation_trace_id,
        component=component,
    )
