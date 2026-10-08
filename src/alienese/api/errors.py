"""Typed internal error taxonomy and deliberate HTTP API error mapping.

Never exposes raw Python stack traces to external API callers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ApiErrorDetail(BaseModel):
    """OpenAI-compatible structured error body with correlation identifiers."""

    model_config = ConfigDict(frozen=True)

    message: str
    type: str
    code: str
    param: str | None = None
    request_id: str | None = None
    operation_id: str | None = None
    correlation_trace_id: str | None = None


class ApiErrorEnvelope(BaseModel):
    """Top-level error response envelope."""

    model_config = ConfigDict(frozen=True)

    error: ApiErrorDetail


class AlieneseError(Exception):
    """Base class for all typed Alienese runtime errors."""

    status_code: int = 500
    error_type: str = "internal_error"
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        if code is not None:
            self.code = code
        self.details = details or {}

    def to_envelope(
        self,
        *,
        request_id: str | None = None,
        operation_id: str | None = None,
        correlation_trace_id: str | None = None,
    ) -> ApiErrorEnvelope:
        """Convert this typed error into an external API error envelope."""
        return ApiErrorEnvelope(
            error=ApiErrorDetail(
                message=self.message,
                type=self.error_type,
                code=self.code,
                param=self.param,
                request_id=request_id,
                operation_id=operation_id,
                correlation_trace_id=correlation_trace_id,
            )
        )


class ProtocolError(AlieneseError):
    """Raised when the request violates message/tool ordering or protocol semantics."""

    status_code = 400
    error_type = "invalid_request_error"
    code = "protocol_error"


class CompatibilityError(AlieneseError):
    """Raised when a caller uses an unsupported OpenAI API feature or parameter."""

    status_code = 400
    error_type = "compatibility_error"
    code = "unsupported_feature"


class IdempotencyConflict(AlieneseError):
    """Raised when an Idempotency-Key is reused with a materially different payload."""

    status_code = 409
    error_type = "idempotency_conflict"
    code = "idempotency_key_conflict"


class ProviderError(AlieneseError):
    """Raised when an upstream or fake model provider fails explicitly."""

    status_code = 502
    error_type = "provider_error"
    code = "provider_error"


class ProviderTimeout(ProviderError):
    """Raised when a model provider call exceeds its deadline."""

    status_code = 504
    error_type = "provider_timeout"
    code = "provider_timeout"


class ProviderUnavailable(ProviderError):
    """Raised when a required model provider is unreachable or unavailable."""

    status_code = 503
    error_type = "provider_unavailable"
    code = "provider_unavailable"


class InvalidProviderResponse(ProviderError):
    """Raised when a model provider returns malformed or invalid contract data."""

    status_code = 502
    error_type = "invalid_provider_response"
    code = "invalid_provider_response"


class InvariantViolation(AlieneseError):
    """Raised when an internal architectural invariant is violated."""

    status_code = 500
    error_type = "invariant_violation"
    code = "invariant_violation"
