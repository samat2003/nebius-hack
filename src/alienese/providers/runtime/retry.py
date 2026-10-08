"""Bounded, deadline-aware retry policy for remote provider calls.

Distinguishes:
- Safe-to-retry failures (`SAFE_RETRYABLE_UPSTREAM`, `RATE_LIMITED`).
- Ambiguous-completion failures (`AMBIGUOUS_COMPLETION`), which are disabled by
  default (`retry_ambiguous_failures=False`) to prevent unintended duplicate
  inference after partial or complete request transmission.
- Non-retryable failures (`LOCAL_CAPACITY`, `NON_RETRYABLE_CLIENT`,
  `NON_RETRYABLE_AUTH`, `NON_RETRYABLE_CONTRACT`).
"""

from __future__ import annotations

import math
import time
from datetime import UTC
from email.utils import parsedate_to_datetime

from pydantic import BaseModel, ConfigDict, Field

from alienese.api.errors import ProviderTimeout
from alienese.providers.runtime.deadlines import DeadlineBudget
from alienese.providers.runtime.errors import FailureCategory


class RetryConfig(BaseModel):
    """Configuration for bounded provider retry behavior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=2, ge=1, le=5)
    base_delay_seconds: float = Field(default=0.25, ge=0.0, le=30.0)
    max_delay_seconds: float = Field(default=2.0, ge=0.0, le=60.0)
    retry_ambiguous_failures: bool = False


def parse_retry_after_seconds(
    header_value: str | None,
    *,
    now_epoch: float | None = None,
) -> float | None:
    """Parse an HTTP `Retry-After` header (delta-seconds or RFC 7231 HTTP-date).

    Returns `None` if `header_value` is absent, malformed, non-finite, or negative.
    """
    if header_value is None:
        return None
    cleaned = header_value.strip()
    if not cleaned:
        return None

    try:
        numeric = float(cleaned)
        if math.isfinite(numeric) and numeric >= 0.0:
            return numeric
        return None
    except ValueError:
        pass

    try:
        parsed_dt = parsedate_to_datetime(cleaned)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

    if parsed_dt.tzinfo is None:
        parsed_dt = parsed_dt.replace(tzinfo=UTC)

    ref_epoch = time.time() if now_epoch is None else now_epoch
    delta = parsed_dt.timestamp() - ref_epoch
    if not math.isfinite(delta) or delta < 0.0:
        return 0.0 if math.isfinite(delta) else None
    return delta


def should_retry_failure(
    category: FailureCategory,
    *,
    attempt_no: int,
    config: RetryConfig,
) -> bool:
    """Return True if the failed attempt is eligible for retry under `config`."""
    if attempt_no >= config.max_attempts:
        return False
    if category in (
        FailureCategory.SAFE_RETRYABLE_UPSTREAM,
        FailureCategory.RATE_LIMITED,
    ):
        return True
    if category == FailureCategory.AMBIGUOUS_COMPLETION:
        return bool(config.retry_ambiguous_failures)
    return False


def compute_retry_delay_seconds(
    *,
    attempt_no: int,
    config: RetryConfig,
    retry_after_header: str | None,
    deadline: DeadlineBudget,
    provider_name: str,
    now_epoch: float | None = None,
) -> float:
    """Compute retry delay honoring `Retry-After` and the remaining turn `deadline`."""
    remaining = deadline.require_remaining(
        provider_name=provider_name,
        phase="retry_scheduling",
    )

    parsed_retry_after = parse_retry_after_seconds(retry_after_header, now_epoch=now_epoch)
    if parsed_retry_after is not None:
        if parsed_retry_after >= remaining:
            raise ProviderTimeout(
                f"Provider '{provider_name}' Retry-After ({parsed_retry_after:.2f}s) "
                f"exceeds remaining turn deadline ({remaining:.2f}s).",
                code="retry_after_exceeds_deadline",
                status_code=504,
            )
        return parsed_retry_after

    exponent = max(0, attempt_no - 1)
    delay: float = float(min(config.max_delay_seconds, config.base_delay_seconds * (2.0**exponent)))
    if delay >= remaining:
        raise ProviderTimeout(
            f"Provider '{provider_name}' retry backoff ({delay:.2f}s) "
            f"exceeds remaining turn deadline ({remaining:.2f}s).",
            code="turn_deadline_exhausted",
            status_code=504,
        )
    return delay
