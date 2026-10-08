"""Shared remote-provider execution runtime for Alienese."""

from __future__ import annotations

from alienese.providers.runtime.circuit_breaker import CircuitState, ProviderCircuitBreaker
from alienese.providers.runtime.client import ProviderHttpClient, ProviderHttpResponse
from alienese.providers.runtime.concurrency import ProviderConcurrencyLimiter
from alienese.providers.runtime.deadlines import DeadlineBudget, ensure_context_deadline
from alienese.providers.runtime.errors import (
    FailureCategory,
    classify_http_status,
    classify_transport_exception,
    map_http_status_to_error,
    map_transport_exception_to_error,
    should_trip_circuit_breaker,
)
from alienese.providers.runtime.retry import (
    RetryConfig,
    compute_retry_delay_seconds,
    parse_retry_after_seconds,
    should_retry_failure,
)
from alienese.providers.runtime.telemetry import (
    build_provider_telemetry,
    extract_upstream_request_id,
    parse_usage_dict,
)

__all__ = [
    "CircuitState",
    "DeadlineBudget",
    "FailureCategory",
    "ProviderCircuitBreaker",
    "ProviderConcurrencyLimiter",
    "ProviderHttpClient",
    "ProviderHttpResponse",
    "RetryConfig",
    "build_provider_telemetry",
    "classify_http_status",
    "classify_transport_exception",
    "compute_retry_delay_seconds",
    "ensure_context_deadline",
    "extract_upstream_request_id",
    "map_http_status_to_error",
    "map_transport_exception_to_error",
    "parse_retry_after_seconds",
    "parse_usage_dict",
    "should_retry_failure",
    "should_trip_circuit_breaker",
]
