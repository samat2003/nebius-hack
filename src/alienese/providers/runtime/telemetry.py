"""Truthful telemetry construction and token/cost accounting for remote providers.

Ensures:
- Unknown token usage remains `None` (never fabricated as `0`).
- Attempt counts and failed attempt counts are explicitly recorded.
- Retried attempt token usage (if any failed response included usage metadata)
  is accumulated separately without losing visibility.
- `estimated_cost_usd` remains `None` unless a verified pricing schedule is supplied.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from alienese.contracts.decisions import ProviderCallTelemetry
from alienese.observability.redaction import redact_string


def _extract_non_negative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def parse_usage_dict(
    raw_usage: Any,
) -> tuple[int | None, int | None, int | None]:
    """Extract `(prompt_tokens, completion_tokens, total_tokens)` from an upstream `usage` block.

    Returns `(None, None, None)` if `raw_usage` is missing or not a mapping.
    """
    if not isinstance(raw_usage, Mapping):
        return None, None, None

    prompt_tok = _extract_non_negative_int(raw_usage.get("prompt_tokens"))
    completion_tok = _extract_non_negative_int(raw_usage.get("completion_tokens"))
    total_tok = _extract_non_negative_int(raw_usage.get("total_tokens"))
    if total_tok is None and prompt_tok is not None and completion_tok is not None:
        total_tok = prompt_tok + completion_tok

    return prompt_tok, completion_tok, total_tok


def extract_upstream_request_id(headers: Mapping[str, str]) -> str | None:
    """Extract a sanitized upstream request identifier from response headers if present."""
    for candidate_header in (
        "nvcf-reqid",
        "x-request-id",
        "x-amzn-requestid",
        "cf-ray",
    ):
        for key, value in headers.items():
            if key.lower() == candidate_header and value.strip():
                return redact_string(value.strip())[:128]
    return None


def build_provider_telemetry(
    *,
    latency_ms: float,
    request_attempt_id: str | None,
    upstream_request_id: str | None = None,
    attempt_count: int = 1,
    failed_attempt_count: int = 0,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    retried_prompt_tokens: int | None = None,
    retried_completion_tokens: int | None = None,
    estimated_cost_usd: float | None = None,
) -> ProviderCallTelemetry:
    """Construct a validated `ProviderCallTelemetry` instance."""
    return ProviderCallTelemetry(
        latency_ms=max(0.0, round(latency_ms, 3)),
        request_attempt_id=request_attempt_id,
        upstream_request_id=upstream_request_id,
        attempt_count=max(1, attempt_count),
        failed_attempt_count=max(0, failed_attempt_count),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        retried_prompt_tokens=retried_prompt_tokens,
        retried_completion_tokens=retried_completion_tokens,
        estimated_cost_usd=estimated_cost_usd,
    )
