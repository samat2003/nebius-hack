"""Idempotency storage contract and in-memory implementation.

Separates:
- `request_id`: per-HTTP-attempt correlation ID
- `operation_id`: stable logical operation ID mapped to the idempotency record
"""

from __future__ import annotations

import asyncio
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
    """In-memory IdempotencyStore supporting concurrent duplicate coordination."""

    def __init__(self) -> None:
        self._records: dict[str, IdempotencyRecord] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, key: str) -> IdempotencyRecord | None:
        """Return the stored IdempotencyRecord for key, if any."""
        return self._records.get(key)

    async def put(self, record: IdempotencyRecord) -> None:
        """Save a completed IdempotencyRecord."""
        self._records[record.idempotency_key] = record

    def key_lock(self, key: str) -> asyncio.Lock:
        """Return or create an asyncio.Lock for the specified idempotency key."""
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock
