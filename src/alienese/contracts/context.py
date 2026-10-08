"""Request, operation, and trace correlation context contracts.

Separates:
- `request_id`: individual HTTP attempt correlation identifier.
- `operation_id`: stable logical operation identifier preserved across idempotent retries.
- `correlation_trace_id`: application correlation identifier (e.g., from X-Trace-ID).
- `traceparent`: W3C distributed tracing header used by OpenTelemetry.
"""

from __future__ import annotations

import re
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_W3C_TRACEPARENT_PATTERN = re.compile(
    r"^00-(?!0{32})[0-9a-f]{32}-(?!0{16})[0-9a-f]{16}-[0-9a-f]{2}$"
)


def generate_request_id() -> str:
    """Generate a unique identifier for a single HTTP request attempt."""
    return f"req_{uuid.uuid4().hex}"


def generate_operation_id() -> str:
    """Generate a unique identifier for a logical runtime operation."""
    return f"op_{uuid.uuid4().hex}"


def generate_correlation_trace_id() -> str:
    """Generate an application-level correlation trace identifier."""
    return f"trc_{uuid.uuid4().hex}"


def normalize_caller_id(raw_value: str | None, *, fallback_factory: object = None) -> str:
    """Validate and preserve a caller-supplied correlation identifier if safe."""
    if raw_value is not None:
        cleaned = raw_value.strip()
        if _SAFE_ID_PATTERN.match(cleaned):
            return cleaned
    if callable(fallback_factory):
        return str(fallback_factory())
    return generate_request_id()


def is_valid_traceparent(value: str | None) -> bool:
    """Return True if value matches the W3C Trace Context traceparent format."""
    if not value:
        return False
    return bool(_W3C_TRACEPARENT_PATTERN.match(value.strip().lower()))


class RequestContext(BaseModel):
    """Explicit immutable request context passed through every TurnEngine phase."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(default_factory=generate_request_id)
    operation_id: str = Field(default_factory=generate_operation_id)
    correlation_trace_id: str = Field(default_factory=generate_correlation_trace_id)
    traceparent: str | None = None
    idempotency_key: str | None = None
    deadline_monotonic: float | None = Field(default=None, gt=0.0)

    @field_validator("traceparent")
    @classmethod
    def _validate_traceparent(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if not is_valid_traceparent(cleaned):
            return None
        return cleaned

    def with_operation_id(self, operation_id: str) -> RequestContext:
        """Return a copy bound to an existing logical operation_id (for idempotent retries)."""
        return self.model_copy(update={"operation_id": operation_id})

    def with_deadline_monotonic(self, deadline_monotonic: float) -> RequestContext:
        """Return a copy bound to a monotonic turn deadline timestamp."""
        return self.model_copy(update={"deadline_monotonic": deadline_monotonic})
