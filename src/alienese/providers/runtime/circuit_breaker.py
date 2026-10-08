"""Per-provider circuit breaker protecting against upstream infrastructure outages.

Accounting semantics:
- Tracks individual HTTP attempts against the remote provider.
- Trips only on upstream availability/outage failures (`SAFE_RETRYABLE_UPSTREAM`,
  `AMBIGUOUS_COMPLETION`), never on local pool exhaustion (`LOCAL_CAPACITY`),
  rate limiting (`RATE_LIMITED`, HTTP 429), or client/schema validation errors.
- Enforces strict single-probe admission in `HALF_OPEN` state so simultaneous
  callers cannot stampede an upstream endpoint during recovery.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

from alienese.api.errors import ProviderUnavailable
from alienese.providers.runtime.errors import FailureCategory, should_trip_circuit_breaker


class CircuitState(StrEnum):
    """Lifecycle states of a provider circuit breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class ProviderCircuitBreaker:
    """Deterministic per-provider circuit breaker with bounded half-open probing."""

    def __init__(
        self,
        *,
        provider_name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        half_open_max_probes: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout_seconds <= 0.0:
            raise ValueError("recovery_timeout_seconds must be > 0.0")
        if half_open_max_probes < 1:
            raise ValueError("half_open_max_probes must be >= 1")

        self._provider_name = provider_name
        self._failure_threshold = failure_threshold
        self._recovery_timeout_seconds = recovery_timeout_seconds
        self._half_open_max_probes = half_open_max_probes
        self._clock = clock

        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._opened_at: float | None = None
        self._half_open_in_flight: int = 0

    @property
    def state(self) -> CircuitState:
        """Return effective state, transitioning OPEN -> HALF_OPEN if recovery elapsed."""
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and (self._clock() - self._opened_at) >= self._recovery_timeout_seconds
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_in_flight = 0
        return self._state

    @property
    def consecutive_failures(self) -> int:
        """Return current consecutive upstream outage attempt count."""
        return self._consecutive_failures

    @property
    def half_open_in_flight(self) -> int:
        """Return number of currently active HALF_OPEN probes."""
        return self._half_open_in_flight

    def before_attempt(self) -> bool:
        """Check whether an HTTP attempt may proceed; True if admitted as HALF_OPEN probe."""
        current = self.state
        if current == CircuitState.OPEN:
            raise ProviderUnavailable(
                f"Circuit breaker for provider '{self._provider_name}' is OPEN.",
                code="circuit_breaker_open",
                status_code=503,
            )
        if current == CircuitState.HALF_OPEN:
            if self._half_open_in_flight >= self._half_open_max_probes:
                raise ProviderUnavailable(
                    f"Circuit breaker for provider '{self._provider_name}' is HALF_OPEN "
                    "with maximum probe calls already in flight.",
                    code="circuit_breaker_open",
                    status_code=503,
                )
            self._half_open_in_flight += 1
            return True
        return False

    def record_success(self, *, was_half_open_probe: bool = False) -> None:
        """Record a successful HTTP attempt and close the circuit."""
        if was_half_open_probe and self._half_open_in_flight > 0:
            self._half_open_in_flight -= 1
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        self._half_open_in_flight = 0

    def record_failure(
        self,
        category: FailureCategory,
        *,
        was_half_open_probe: bool = False,
    ) -> None:
        """Record a failed HTTP attempt according to its FailureCategory."""
        if was_half_open_probe and self._half_open_in_flight > 0:
            self._half_open_in_flight -= 1

        if not should_trip_circuit_breaker(category):
            return

        self._consecutive_failures += 1
        if was_half_open_probe or self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()
            self._half_open_in_flight = 0
            return

        if self._consecutive_failures >= self._failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()
            self._half_open_in_flight = 0

    def release_aborted_probe(self, *, was_half_open_probe: bool) -> None:
        """Release in-flight probe slot if a HALF_OPEN attempt was cancelled before completion."""
        if was_half_open_probe and self._half_open_in_flight > 0:
            self._half_open_in_flight -= 1
