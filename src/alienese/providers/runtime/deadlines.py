"""End-to-end monotonic turn deadline budget enforcement for remote provider calls.

Ensures a single monotonic deadline per incoming Alienese turn governs queue
admission waits, HTTP connection/read timeouts, retry eligibility, and backoff
sleeps without contaminating deterministic state or replay semantics.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from alienese.api.errors import ProviderTimeout
from alienese.contracts.context import RequestContext

DEFAULT_TURN_TIMEOUT_SECONDS = 30.0
_MIN_OPERATIONAL_BUDGET_SECONDS = 1e-6


def ensure_context_deadline(
    ctx: RequestContext,
    *,
    timeout_seconds: float = DEFAULT_TURN_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> RequestContext:
    """Ensure `ctx` carries a monotonic turn deadline timestamp."""
    if ctx.deadline_monotonic is not None:
        return ctx
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be > 0.0")
    return ctx.with_deadline_monotonic(clock() + timeout_seconds)


class DeadlineBudget:
    """Monotonic deadline tracker shared across admission, retries, and HTTP attempts."""

    def __init__(
        self,
        *,
        deadline_monotonic: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if deadline_monotonic <= 0.0:
            raise ValueError("deadline_monotonic must be > 0.0")
        self._deadline_monotonic = deadline_monotonic
        self._clock = clock

    @classmethod
    def from_context(
        cls,
        ctx: RequestContext,
        *,
        default_timeout_seconds: float = DEFAULT_TURN_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> DeadlineBudget:
        """Create a DeadlineBudget bound to `ctx.deadline_monotonic` or default timeout."""
        if ctx.deadline_monotonic is not None:
            return cls(deadline_monotonic=ctx.deadline_monotonic, clock=clock)
        if default_timeout_seconds <= 0.0:
            raise ValueError("default_timeout_seconds must be > 0.0")
        return cls(deadline_monotonic=clock() + default_timeout_seconds, clock=clock)

    @property
    def deadline_monotonic(self) -> float:
        """Return the absolute monotonic timestamp at which the budget expires."""
        return self._deadline_monotonic

    def remaining_seconds(self) -> float:
        """Return the remaining seconds before deadline expiry (clamped to >= 0.0)."""
        return max(0.0, self._deadline_monotonic - self._clock())

    def is_expired(self) -> bool:
        """Return True if the deadline has been reached or exceeded."""
        return (self._deadline_monotonic - self._clock()) <= _MIN_OPERATIONAL_BUDGET_SECONDS

    def require_remaining(
        self,
        *,
        provider_name: str,
        phase: str,
        min_seconds: float = _MIN_OPERATIONAL_BUDGET_SECONDS,
    ) -> float:
        """Return remaining seconds or raise `ProviderTimeout` if insufficient budget remains."""
        remaining = self._deadline_monotonic - self._clock()
        if remaining < max(min_seconds, _MIN_OPERATIONAL_BUDGET_SECONDS):
            raise ProviderTimeout(
                f"Provider '{provider_name}' turn deadline exhausted before {phase}.",
                code="turn_deadline_exhausted",
            )
        return remaining

    def attempt_timeout_seconds(
        self,
        configured_attempt_timeout: float | None = None,
        *,
        provider_name: str,
        phase: str = "http_attempt",
    ) -> float:
        """Return per-attempt timeout bounded by remaining turn deadline."""
        remaining = self.require_remaining(provider_name=provider_name, phase=phase)
        if configured_attempt_timeout is not None and configured_attempt_timeout > 0.0:
            return min(remaining, configured_attempt_timeout)
        return remaining
