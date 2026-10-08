"""Per-provider bounded concurrency semaphore and queue admission control.

Prevents unbounded waiter accumulation and guarantees slot cleanup across
normal completion, exceptions, timeouts, and coroutine cancellation.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from alienese.api.errors import ProviderTimeout, ProviderUnavailable
from alienese.providers.runtime.deadlines import DeadlineBudget


class ProviderConcurrencyLimiter:
    """Bounded concurrency limiter with explicit waiter admission control."""

    def __init__(
        self,
        *,
        provider_name: str,
        max_concurrency: int = 8,
        max_queue_waiters: int = 16,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if max_queue_waiters < 0:
            raise ValueError("max_queue_waiters must be >= 0")
        self._provider_name = provider_name
        self._max_concurrency = max_concurrency
        self._max_queue_waiters = max_queue_waiters
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_count = 0
        self._waiting_count = 0

    @property
    def active_count(self) -> int:
        """Return the number of currently executing provider calls."""
        return self._active_count

    @property
    def waiting_count(self) -> int:
        """Return the number of coroutines currently waiting for a concurrency slot."""
        return self._waiting_count

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def max_queue_waiters(self) -> int:
        return self._max_queue_waiters

    @asynccontextmanager
    async def acquire(self, deadline: DeadlineBudget) -> AsyncIterator[None]:
        """Acquire a concurrency slot within `deadline` or fail closed."""
        remaining = deadline.require_remaining(
            provider_name=self._provider_name,
            phase="concurrency_admission",
        )

        if self._semaphore.locked() and self._waiting_count >= self._max_queue_waiters:
            raise ProviderUnavailable(
                f"Provider '{self._provider_name}' concurrency queue capacity exhausted "
                f"(active={self._active_count}, waiting={self._waiting_count}).",
                code="provider_overloaded",
                status_code=503,
            )

        self._waiting_count += 1
        acquired = False
        try:
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=remaining)
                acquired = True
            except TimeoutError as exc:
                raise ProviderTimeout(
                    f"Provider '{self._provider_name}' timed out waiting for a concurrency slot.",
                    code="turn_deadline_exhausted",
                ) from exc
        finally:
            self._waiting_count -= 1

        self._active_count += 1
        try:
            deadline.require_remaining(
                provider_name=self._provider_name,
                phase="post_admission_check",
            )
            yield
        finally:
            self._active_count -= 1
            if acquired:
                self._semaphore.release()
