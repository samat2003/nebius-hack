"""Normalized remote-provider error taxonomy and retry-safety classification.

Maps upstream HTTP status codes and transport exceptions into Alienese typed
errors without leaking credentials, headers, or private payload contents.
"""

from __future__ import annotations

from enum import StrEnum

import httpx

from alienese.api.errors import (
    AlieneseError,
    InvalidProviderResponse,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
)
from alienese.observability.redaction import redact_string


class FailureCategory(StrEnum):
    """Operational failure category used for retry and circuit-breaker policy."""

    SAFE_RETRYABLE_UPSTREAM = "SAFE_RETRYABLE_UPSTREAM"
    AMBIGUOUS_COMPLETION = "AMBIGUOUS_COMPLETION"
    RATE_LIMITED = "RATE_LIMITED"
    LOCAL_CAPACITY = "LOCAL_CAPACITY"
    NON_RETRYABLE_CLIENT = "NON_RETRYABLE_CLIENT"
    NON_RETRYABLE_AUTH = "NON_RETRYABLE_AUTH"
    NON_RETRYABLE_CONTRACT = "NON_RETRYABLE_CONTRACT"
    DEADLINE_EXHAUSTED = "DEADLINE_EXHAUSTED"


def classify_http_status(status_code: int) -> FailureCategory:
    """Classify an upstream non-200 HTTP status code."""
    if status_code == 429:
        return FailureCategory.RATE_LIMITED
    if status_code in (408, 502, 503, 504):
        return FailureCategory.SAFE_RETRYABLE_UPSTREAM
    if status_code == 500:
        return FailureCategory.AMBIGUOUS_COMPLETION
    if status_code in (401, 403):
        return FailureCategory.NON_RETRYABLE_AUTH
    if status_code in (202, 400, 404, 422):
        return FailureCategory.NON_RETRYABLE_CLIENT
    if 300 <= status_code < 400:
        return FailureCategory.NON_RETRYABLE_CONTRACT
    return FailureCategory.NON_RETRYABLE_CLIENT


def classify_transport_exception(exc: BaseException) -> FailureCategory:
    """Classify an httpx transport or timeout exception for retry and circuit-breaker policy."""
    if isinstance(exc, httpx.PoolTimeout):
        return FailureCategory.LOCAL_CAPACITY
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return FailureCategory.SAFE_RETRYABLE_UPSTREAM
    if isinstance(
        exc,
        (
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
        ),
    ):
        return FailureCategory.AMBIGUOUS_COMPLETION
    if isinstance(exc, httpx.TimeoutException):
        return FailureCategory.AMBIGUOUS_COMPLETION
    return FailureCategory.NON_RETRYABLE_CLIENT


def should_trip_circuit_breaker(category: FailureCategory) -> bool:
    """Return True if a failure category represents an upstream infrastructure outage.

    - Local capacity exhaustion (`LOCAL_CAPACITY`) must NEVER trip the provider circuit breaker.
    - Rate limiting (`RATE_LIMITED`, HTTP 429) is treated as throttling rather than outage.
    - Client/contract validation errors (`NON_RETRYABLE_CLIENT`, `NON_RETRYABLE_CONTRACT`)
      do not trip the circuit breaker.
    """
    return category in (
        FailureCategory.SAFE_RETRYABLE_UPSTREAM,
        FailureCategory.AMBIGUOUS_COMPLETION,
    )


def map_http_status_to_error(
    *,
    provider_name: str,
    status_code: int,
    upstream_request_id: str | None = None,
) -> AlieneseError:
    """Map an upstream HTTP status code to a sanitized AlieneseError."""
    safe_req_id = redact_string(upstream_request_id) if upstream_request_id else None
    details = {"provider": provider_name, "upstream_status": status_code}
    if safe_req_id:
        details["upstream_request_id"] = safe_req_id

    if status_code == 202:
        return ProviderUnavailable(
            f"Provider '{provider_name}' returned HTTP 202 pending invocation; "
            "asynchronous polling is not supported.",
            code="provider_pending_invocation",
            status_code=503,
            details=details,
        )
    if 300 <= status_code < 400:
        return InvalidProviderResponse(
            f"Provider '{provider_name}' returned unexpected redirect HTTP {status_code}; "
            "cross-origin redirects are forbidden.",
            code="unexpected_redirect",
            status_code=502,
            details=details,
        )
    if status_code in (401, 403):
        return ProviderUnavailable(
            f"Provider '{provider_name}' rejected authentication or authorization "
            f"(HTTP {status_code}).",
            code="provider_auth_failed",
            status_code=503,
            details=details,
        )
    if status_code == 404:
        return ProviderUnavailable(
            f"Provider '{provider_name}' endpoint or model was not found (HTTP 404).",
            code="provider_endpoint_not_found",
            status_code=502,
            details=details,
        )
    if status_code == 429:
        return ProviderUnavailable(
            f"Provider '{provider_name}' rate limit exceeded (HTTP 429).",
            code="provider_rate_limited",
            status_code=503,
            details=details,
        )
    if status_code in (400, 422):
        return ProviderError(
            f"Provider '{provider_name}' rejected request parameters (HTTP {status_code}).",
            code="provider_request_rejected",
            status_code=502,
            details=details,
        )
    if status_code in (408, 504):
        return ProviderTimeout(
            f"Provider '{provider_name}' timed out upstream (HTTP {status_code}).",
            code="provider_timeout",
            status_code=504,
            details=details,
        )
    if status_code in (500, 502, 503):
        return ProviderUnavailable(
            f"Provider '{provider_name}' experienced an upstream server failure "
            f"(HTTP {status_code}).",
            code="provider_upstream_error",
            status_code=503,
            details=details,
        )
    return ProviderError(
        f"Provider '{provider_name}' returned unexpected HTTP {status_code}.",
        code="provider_unexpected_status",
        status_code=502,
        details=details,
    )


def map_transport_exception_to_error(
    *,
    provider_name: str,
    exc: BaseException,
) -> AlieneseError:
    """Map an httpx transport exception into a sanitized AlieneseError."""
    if isinstance(exc, AlieneseError):
        return exc
    if isinstance(exc, httpx.PoolTimeout):
        return ProviderUnavailable(
            f"Local HTTP connection pool exhausted for provider '{provider_name}'.",
            code="local_pool_exhausted",
            status_code=503,
            details={"provider": provider_name},
        )
    if isinstance(exc, httpx.ConnectTimeout):
        return ProviderTimeout(
            f"Timed out establishing connection to provider '{provider_name}'.",
            code="provider_connect_timeout",
            status_code=504,
            details={"provider": provider_name},
        )
    if isinstance(exc, httpx.ReadTimeout):
        return ProviderTimeout(
            f"Timed out reading response from provider '{provider_name}'.",
            code="provider_read_timeout",
            status_code=504,
            details={"provider": provider_name},
        )
    if isinstance(exc, httpx.WriteTimeout):
        return ProviderTimeout(
            f"Timed out sending request to provider '{provider_name}'.",
            code="provider_write_timeout",
            status_code=504,
            details={"provider": provider_name},
        )
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeout(
            f"Timed out communicating with provider '{provider_name}'.",
            code="provider_timeout",
            status_code=504,
            details={"provider": provider_name},
        )
    if isinstance(exc, httpx.ConnectError):
        return ProviderUnavailable(
            f"Failed to connect to provider '{provider_name}'.",
            code="provider_connect_error",
            status_code=503,
            details={"provider": provider_name},
        )
    if isinstance(exc, httpx.RemoteProtocolError):
        return ProviderUnavailable(
            f"Provider '{provider_name}' interrupted the HTTP protocol stream.",
            code="provider_protocol_error",
            status_code=502,
            details={"provider": provider_name},
        )
    return ProviderUnavailable(
        f"Transport failure communicating with provider '{provider_name}' ({type(exc).__name__}).",
        code="provider_transport_error",
        status_code=503,
        details={"provider": provider_name},
    )
