"""Replay trace storage with mandatory pre-persistence secret redaction.

Enforces Refinement 9 & Invariant 14:
- Secret redaction always occurs BEFORE a ReplayArtifact is persisted or serialized.
- Runtime traces are stored in memory (or git-ignored local directory only when
  explicitly requested).
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from alienese.contracts.traces import ReplayArtifact
from alienese.observability.redaction import redact_mapping


def sanitize_replay_artifact(artifact: ReplayArtifact) -> ReplayArtifact:
    """Redact all secret keys and string patterns across the entire ReplayArtifact."""
    raw_dict = artifact.model_dump(mode="json")
    redacted_dict = redact_mapping(raw_dict)
    return ReplayArtifact.model_validate(redacted_dict)


def dump_replay_artifact_json(artifact: ReplayArtifact) -> str:
    """Serialize a sanitized ReplayArtifact to canonical deterministic JSON."""
    sanitized = sanitize_replay_artifact(artifact)
    return json.dumps(
        sanitized.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )


def load_replay_artifact_json(raw_json: str) -> ReplayArtifact:
    """Deserialize and sanitize a ReplayArtifact from JSON."""
    parsed = json.loads(raw_json)
    redacted = redact_mapping(parsed)
    return ReplayArtifact.model_validate(redacted)


@runtime_checkable
class TraceStore(Protocol):
    """Protocol for storing and retrieving sanitized ReplayArtifacts."""

    async def save(self, artifact: ReplayArtifact) -> ReplayArtifact:
        """Sanitize and persist a ReplayArtifact."""
        ...

    async def get_by_operation_id(self, operation_id: str) -> ReplayArtifact | None:
        """Retrieve a stored ReplayArtifact by its logical operation_id."""
        ...


class InMemoryTraceStore:
    """In-memory TraceStore that enforces secret redaction prior to persistence."""

    def __init__(self) -> None:
        self._by_operation_id: dict[str, ReplayArtifact] = {}

    async def save(self, artifact: ReplayArtifact) -> ReplayArtifact:
        """Sanitize and store a ReplayArtifact keyed by operation_id."""
        sanitized = sanitize_replay_artifact(artifact)
        self._by_operation_id[sanitized.telemetry.operation_id] = sanitized
        return sanitized

    async def get_by_operation_id(self, operation_id: str) -> ReplayArtifact | None:
        """Return the stored sanitized ReplayArtifact for operation_id, if present."""
        return self._by_operation_id.get(operation_id)

    def list_all(self) -> tuple[ReplayArtifact, ...]:
        """Return all stored sanitized ReplayArtifacts in insertion order."""
        return tuple(self._by_operation_id.values())
