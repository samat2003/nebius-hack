"""Idempotency storage contract and bounded in-memory implementation.

Separates:
- `request_id`: per-HTTP-attempt correlation ID
- `operation_id`: stable logical operation ID mapped to the idempotency record

Enforces bounded memory growth via configurable TTL expiration and LRU eviction,
including pruning idle per-key locks.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

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


class InMemoryIdempotencyStore:
    """Bounded in-memory IdempotencyStore supporting concurrent duplicate coordination."""

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
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()

    def _prune_expired(self, now: float) -> None:
        expired_keys = [
            key
            for key, (stored_at, _) in self._records.items()
            if (now - stored_at) >= self._ttl_seconds
        ]
        for key in expired_keys:
            self._records.pop(key, None)
            lock = self._locks.get(key)
            if lock is not None and not lock.locked():
                self._locks.pop(key, None)

    def _prune_idle_locks(self) -> None:
        if len(self._locks) < self._max_entries:
            return
        idle_keys = [
            k for k, lk in self._locks.items() if not lk.locked() and k not in self._records
        ]
        for k in idle_keys:
            self._locks.pop(k, None)
        while len(self._locks) >= self._max_entries:
            evicted = False
            for k, lk in list(self._locks.items()):
                if not lk.locked():
                    self._locks.pop(k, None)
                    evicted = True
                    break
            if not evicted:
                break

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
                if lk is not None and not lk.locked():
                    self._locks.pop(evicted_key, None)
        self._records[key] = (now, record)

    def key_lock(self, key: str) -> asyncio.Lock:
        """Return or create an asyncio.Lock for the specified idempotency key."""
        self._prune_expired(self._clock())
        lock = self._locks.get(key)
        if lock is not None:
            self._locks.move_to_end(key)
            return lock
        self._prune_idle_locks()
        lock = asyncio.Lock()
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
