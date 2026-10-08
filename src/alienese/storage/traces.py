"""Trace artifact storage with explicit telemetry vs replay separation and bounded retention.

Implements two explicit modes:
1. `TraceMode.METADATA_ONLY` (enabled by default, `replayable=False`):
   Retains only digests, counts, timings, action types/dispositions, model
   identifiers, and decision metrics; retains no raw user prompts, repository
   content, tool arguments, or tool outputs.
2. `TraceMode.FULL_FIDELITY_REPLAY` (disabled by default, `replayable=True`):
   Explicitly enabled only for controlled local development/evaluation; retains
   exact reconstruction inputs (`normalized_events`, `available_tools`,
   `candidates`, `decision_semantics`, `generation_job`, `generation_semantics`,
   `logical_response`).

Never mutates semantic event content via redaction while claiming `replayable=True`.
If `sanitize_replay_artifact` is invoked in default mode or detects secret material,
it converts the artifact to `TraceMode.METADATA_ONLY` with `replayable=False`.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from alienese.api.errors import InvariantViolation
from alienese.contracts.state import WorkingState
from alienese.contracts.traces import ReplayArtifact, ReplaySemantics, TraceMode
from alienese.engine.reconstruct import reconstruct
from alienese.observability.redaction import contains_sensitive_material


def to_metadata_only(artifact: ReplayArtifact) -> ReplayArtifact:
    """Convert any ReplayArtifact into a METADATA_ONLY artifact (`replayable=False`)."""
    if (
        artifact.semantics.trace_mode == TraceMode.METADATA_ONLY
        and not artifact.semantics.replayable
    ):
        return artifact

    sem = artifact.semantics
    metadata_semantics = ReplaySemantics(
        trace_mode=TraceMode.METADATA_ONLY,
        replayable=False,
        versions=sem.versions,
        state_digest=sem.state_digest,
        event_count=sem.event_count or len(sem.normalized_events),
        tool_count=sem.tool_count or len(sem.available_tools),
        candidate_count=sem.candidate_count or len(sem.candidates),
        event_digests=sem.event_digests,
        candidate_digests=sem.candidate_digests,
        decision_semantics=sem.decision_semantics,
        finish_reason=sem.finish_reason,
        response_sha256=sem.response_sha256,
        normalized_events=(),
        available_tools=(),
        candidates=(),
        generation_job=None,
        generation_semantics=None,
        logical_response=None,
    )
    return ReplayArtifact(
        semantics=metadata_semantics,
        telemetry=artifact.telemetry,
    )


def sanitize_replay_artifact(
    artifact: ReplayArtifact,
    *,
    allow_full_fidelity: bool = False,
) -> ReplayArtifact:
    """Prepare a ReplayArtifact for persistence without corrupting replay invariants.

    - When `allow_full_fidelity=False` (default) or `artifact` is already
      `METADATA_ONLY`, strips all raw content and marks `replayable=False`
      (`TraceMode.METADATA_ONLY`).
    - When `allow_full_fidelity=True`, preserves `TraceMode.FULL_FIDELITY_REPLAY`
      (`replayable=True`) ONLY if no secret material is present. If secret
      material is detected, downgrades to `TraceMode.METADATA_ONLY`
      (`replayable=False`) rather than mutating semantic events and invalidating
      `state_digest`.
    """
    if not allow_full_fidelity or artifact.trace_mode == TraceMode.METADATA_ONLY:
        return to_metadata_only(artifact)

    raw_dict = artifact.model_dump(mode="json")
    if contains_sensitive_material(raw_dict):
        return to_metadata_only(artifact)

    return artifact


def reconstruct_from_replay_artifact(artifact: ReplayArtifact) -> WorkingState:
    """Reconstruct and verify WorkingState from a FULL_FIDELITY_REPLAY artifact.

    Raises `InvariantViolation` if `artifact.replayable` is False or if the
    reconstructed `state_digest` does not match `artifact.semantics.state_digest`.
    """
    if not artifact.replayable or artifact.trace_mode != TraceMode.FULL_FIDELITY_REPLAY:
        raise InvariantViolation(
            f"Cannot reconstruct WorkingState from non-replayable trace artifact "
            f"(trace_mode={artifact.trace_mode.value}, replayable={artifact.replayable})."
        )
    state = reconstruct(
        artifact.semantics.normalized_events,
        available_tools=artifact.semantics.available_tools,
    )
    if state.state_digest != artifact.semantics.state_digest:
        raise InvariantViolation(
            f"Replay state digest mismatch: expected '{artifact.semantics.state_digest}', "
            f"got '{state.state_digest}'."
        )
    return state


def dump_replay_artifact_json(
    artifact: ReplayArtifact,
    *,
    allow_full_fidelity: bool = False,
) -> str:
    """Serialize a ReplayArtifact to canonical deterministic JSON."""
    prepared = sanitize_replay_artifact(
        artifact,
        allow_full_fidelity=allow_full_fidelity,
    )
    return json.dumps(
        prepared.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )


def load_replay_artifact_json(
    raw_json: str,
    *,
    allow_full_fidelity: bool = True,
) -> ReplayArtifact:
    """Deserialize a ReplayArtifact from JSON and enforce fidelity/secret invariants."""
    parsed = json.loads(raw_json)
    artifact = ReplayArtifact.model_validate(parsed)
    return sanitize_replay_artifact(artifact, allow_full_fidelity=allow_full_fidelity)


@runtime_checkable
class TraceStore(Protocol):
    """Protocol for storing and retrieving trace artifacts."""

    async def save(self, artifact: ReplayArtifact) -> ReplayArtifact:
        """Persist a ReplayArtifact according to configured telemetry/replay mode."""
        ...

    async def get_by_operation_id(self, operation_id: str) -> ReplayArtifact | None:
        """Retrieve a stored ReplayArtifact by its logical operation_id."""
        ...


class InMemoryTraceStore:
    """Bounded in-memory TraceStore with configurable fidelity, capacity, and TTL."""

    def __init__(
        self,
        *,
        allow_full_fidelity: bool = False,
        max_entries: int = 1024,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._allow_full_fidelity = allow_full_fidelity
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._by_operation_id: OrderedDict[str, tuple[float, ReplayArtifact]] = OrderedDict()

    def _prune_expired(self, now: float) -> None:
        expired_ids = [
            op_id
            for op_id, (stored_at, _) in self._by_operation_id.items()
            if (now - stored_at) >= self._ttl_seconds
        ]
        for op_id in expired_ids:
            self._by_operation_id.pop(op_id, None)

    async def save(self, artifact: ReplayArtifact) -> ReplayArtifact:
        """Sanitize/prepare and store a ReplayArtifact keyed by operation_id."""
        now = self._clock()
        self._prune_expired(now)
        prepared = sanitize_replay_artifact(
            artifact,
            allow_full_fidelity=self._allow_full_fidelity,
        )
        op_id = prepared.telemetry.operation_id
        if op_id in self._by_operation_id:
            self._by_operation_id.move_to_end(op_id)
        else:
            while len(self._by_operation_id) >= self._max_entries:
                self._by_operation_id.popitem(last=False)
        self._by_operation_id[op_id] = (now, prepared)
        return prepared

    async def get_by_operation_id(self, operation_id: str) -> ReplayArtifact | None:
        """Return the stored ReplayArtifact for operation_id, if present and not expired."""
        now = self._clock()
        self._prune_expired(now)
        entry = self._by_operation_id.get(operation_id)
        if entry is None:
            return None
        self._by_operation_id.move_to_end(operation_id)
        return entry[1]

    def list_all(self) -> tuple[ReplayArtifact, ...]:
        """Return all stored ReplayArtifacts in insertion/LRU order."""
        self._prune_expired(self._clock())
        return tuple(art for _, art in self._by_operation_id.values())

    @property
    def size(self) -> int:
        """Return current number of stored trace artifacts."""
        self._prune_expired(self._clock())
        return len(self._by_operation_id)
