"""Idempotency storage contract and bounded in-memory implementation.

Separates:
- `request_id`: per-HTTP-attempt correlation ID
- `operation_id`: stable logical operation ID mapped to the idempotency record

Concurrency & retention guarantees:
- Per-key mutual exclusion uses reference-counted locks (`_ManagedKeyLock`) that
  track both the active lock holder and all queued waiting coroutines. A lock is
  never evicted or replaced while any coroutine holds it or is waiting to
  acquire it—including the event-loop scheduling window between `release()` and
  a queued waiter waking up.
- When `max_entries` lock slots are simultaneously held/awaited by active
  in-flight operations, bounded admission control rejects additional concurrent
  keys with `ProviderUnavailable` (`code="idempotency_capacity_exceeded"`,
  HTTP 503) rather than breaking per-key mutual exclusion.
- In-memory idempotency guarantees are scoped to a single process lifetime and
  bounded by configured `max_entries` (LRU eviction) and `ttl_seconds`.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable
from types import TracebackType
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from alienese.api.errors import ProviderUnavailable
from alienese.api.models import ChatCompletionResponse


class IdempotencyRecord(BaseModel):
    """Completed logical operation stored for an Idempotency-Key."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    idempotency_key: str = Field(min_length=1)
    request_fingerprint: str = Field(min_length=64, max_length=64)
    operation_id: str = Field(min_length=1)
    first_request_id: str = Field(min_length=1)
    response: ChatCompletionResponse


@runtime_checkable
class IdempotencyStore(Protocol):
    """Abstract storage interface for idempotent request records."""

    async def get(self, key: str) -> IdempotencyRecord | None:
        """Retrieve an existing IdempotencyRecord for key, if present."""
        ...

    async def put(self, record: IdempotencyRecord) -> None:
        """Store a completed IdempotencyRecord."""
        ...

    def key_lock(self, key: str) -> asyncio.Lock:
        """Return an asyncio.Lock scoped to the given idempotency key."""
        ...


class _ManagedKeyLock(asyncio.Lock):
    """An `asyncio.Lock` that tracks active holders and queued waiters explicitly.

    In Python `asyncio`, `Lock.release()` marks `_locked = False` and schedules
    the next waiter via `call_soon()` before that waiter coroutine resumes.
    Tracking `active_refs` synchronously around `__aenter__` / `__aexit__`
    ensures the store never treats a lock with queued waiters as idle.
    """

    def __init__(self, key: str, store: InMemoryIdempotencyStore) -> None:
        super().__init__()
        self.key = key
        self._store = store
        self.active_refs: int = 0

    @property
    def is_in_use(self) -> bool:
        """Return True if held or if any coroutine is queued waiting to acquire."""
        return self.active_refs > 0 or self.locked()

    async def __aenter__(self) -> None:
        self.active_refs += 1
        try:
            await super().__aenter__()
        except BaseException:
            self.active_refs -= 1
            self._store._on_lock_released(self.key)
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            await super().__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self.active_refs -= 1
            self._store._on_lock_released(self.key)


class InMemoryIdempotencyStore:
    """Bounded in-memory IdempotencyStore supporting concurrent duplicate coordination.

    Note: Retention is bounded by `max_entries`, `ttl_seconds`, and the current
    OS process lifetime.
    """

    def __init__(
        self,
        *,
        max_entries: int = 1024,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._records: OrderedDict[str, tuple[float, IdempotencyRecord]] = OrderedDict()
        self._locks: OrderedDict[str, _ManagedKeyLock] = OrderedDict()

    def _on_lock_released(self, key: str) -> None:
        """Clean up an unreferenced lock if its key has no stored record and we are at capacity."""
        lock = self._locks.get(key)
        if lock is None or lock.is_in_use:
            return
        if key not in self._records and len(self._locks) > self._max_entries:
            self._locks.pop(key, None)

    def _prune_expired(self, now: float) -> None:
        expired_keys = [
            key
            for key, (stored_at, _) in self._records.items()
            if (now - stored_at) >= self._ttl_seconds
        ]
        for key in expired_keys:
            self._records.pop(key, None)
            lock = self._locks.get(key)
            if lock is not None and not lock.is_in_use:
                self._locks.pop(key, None)

    def _prune_idle_locks(self) -> None:
        if len(self._locks) < self._max_entries:
            return
        # 1. Evict strictly idle locks (no holder, no queued waiters) that have no cached record
        idle_without_record = [
            k for k, lk in self._locks.items() if not lk.is_in_use and k not in self._records
        ]
        for k in idle_without_record:
            self._locks.pop(k, None)
            if len(self._locks) < self._max_entries:
                return

        # 2. If still at capacity, evict oldest strictly idle locks (never evicting in-use locks)
        for k, lk in list(self._locks.items()):
            if not lk.is_in_use:
                self._locks.pop(k, None)
                if len(self._locks) < self._max_entries:
                    return

    async def get(self, key: str) -> IdempotencyRecord | None:
        """Return the stored IdempotencyRecord for key if present and not expired."""
        now = self._clock()
        self._prune_expired(now)
        entry = self._records.get(key)
        if entry is None:
            return None
        self._records.move_to_end(key)
        return entry[1]

    async def put(self, record: IdempotencyRecord) -> None:
        """Save a completed IdempotencyRecord, evicting expired/oldest entries as needed."""
        now = self._clock()
        self._prune_expired(now)
        key = record.idempotency_key
        if key in self._records:
            self._records.move_to_end(key)
        else:
            while len(self._records) >= self._max_entries:
                evicted_key, _ = self._records.popitem(last=False)
                lk = self._locks.get(evicted_key)
                if lk is not None and not lk.is_in_use:
                    self._locks.pop(evicted_key, None)
        self._records[key] = (now, record)

    def key_lock(self, key: str) -> asyncio.Lock:
        """Return or create a reference-counted lock for the specified idempotency key.

        Never evicts or replaces a lock that is currently held or has queued
        waiters. Raises `ProviderUnavailable` (HTTP 503) if all `max_entries`
        lock slots are simultaneously in use by active operations.
        """
        self._prune_expired(self._clock())
        existing = self._locks.get(key)
        if existing is not None:
            self._locks.move_to_end(key)
            return existing

        self._prune_idle_locks()
        if len(self._locks) >= self._max_entries:
            raise ProviderUnavailable(
                "Idempotency store concurrency capacity exhausted; all lock slots have "
                "active in-flight operations or queued waiters.",
                param="Idempotency-Key",
                code="idempotency_capacity_exceeded",
            )

        lock = _ManagedKeyLock(key, self)
        self._locks[key] = lock
        return lock

    @property
    def size(self) -> int:
        """Return current number of active cached idempotency records."""
        self._prune_expired(self._clock())
        return len(self._records)

    @property
    def lock_count(self) -> int:
        """Return current number of tracked key locks."""
        return len(self._locks)
