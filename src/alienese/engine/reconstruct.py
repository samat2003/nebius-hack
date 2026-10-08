"""Deterministic WorkingState reconstruction from normalized event sequences.

Invariants enforced:
- WorkingState is a pure projection over NormalizedEvent history + available tools.
- Process-local prior state is never required for correctness.
- Prefix checkpoints produce the exact same WorkingState and state_digest as full replay.
- `MODEL_GENERATED` and `UNTRUSTED_EXTERNAL` content can never enter `trusted_system_instructions`.
- Calling a mutation or test tool sets `mutation_attempted` / `verification_attempted`,
  never `mutation_confirmed` / `verification_confirmed` without explicit confirmation evidence.
- `state_digest` is computed via canonical JSON serialization over semantic state fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from alienese.api.errors import InvariantViolation, ProtocolError
from alienese.contracts.events import EventKind, EventProvenance, NormalizedEvent, TrustLevel
from alienese.contracts.state import (
    CanonicalCapability,
    ExternalToolBinding,
    MutationVerificationState,
    RecordedAction,
    ToolObservation,
    WorkingState,
)
from alienese.engine.normalize import resolve_tool_capability

_MUTATION_CAPABILITIES: frozenset[CanonicalCapability] = frozenset(
    {
        CanonicalCapability.APPLY_PATCH,
        CanonicalCapability.WRITE_FILE,
    }
)

_VERIFICATION_CAPABILITIES: frozenset[CanonicalCapability] = frozenset(
    {
        CanonicalCapability.RUN_TEST,
    }
)


def _canonicalize_json_value(value: Any) -> Any:
    """Recursively canonicalize mappings and sequences for deterministic hashing."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolError("Non-finite float (NaN or Infinity) is not permitted in state.")
    if isinstance(value, Mapping):
        return {str(k): _canonicalize_json_value(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_json_value(item) for item in value]
    return value


def compute_state_digest(
    *,
    event_count: int,
    last_sequence_no: int | None,
    initial_user_request: str | None,
    latest_user_request: str | None,
    user_messages: Sequence[str],
    trusted_system_instructions: Sequence[str],
    available_tools: Sequence[ExternalToolBinding],
    recent_actions: Sequence[RecordedAction],
    pending_tool_call_ids: Sequence[str],
    latest_tool_observation: ToolObservation | None,
    mutation_verification: MutationVerificationState,
    provenance_refs: Sequence[EventProvenance],
) -> str:
    """Compute a canonical SHA-256 digest over semantic WorkingState fields."""
    sorted_tools = sorted(available_tools, key=lambda t: t.external_name)
    semantic_payload = {
        "schema_version": "v1",
        "event_count": event_count,
        "last_sequence_no": last_sequence_no,
        "initial_user_request": initial_user_request,
        "latest_user_request": latest_user_request,
        "user_messages": list(user_messages),
        "trusted_system_instructions": list(trusted_system_instructions),
        "available_tools": [
            {
                "external_name": t.external_name,
                "description": t.description,
                "parameters_schema": _canonicalize_json_value(t.parameters_schema),
                "canonical_capability": t.canonical_capability.value,
            }
            for t in sorted_tools
        ],
        "recent_actions": [
            {
                "sequence_no": a.sequence_no,
                "tool_call_id": a.tool_call_id,
                "tool_name": a.tool_name,
                "canonical_capability": a.canonical_capability.value,
                "arguments": _canonicalize_json_value(a.arguments),
                "provenance": a.provenance.model_dump(mode="json"),
            }
            for a in recent_actions
        ],
        "pending_tool_call_ids": list(pending_tool_call_ids),
        "latest_tool_observation": (
            {
                "sequence_no": latest_tool_observation.sequence_no,
                "tool_call_id": latest_tool_observation.tool_call_id,
                "tool_name": latest_tool_observation.tool_name,
                "content": latest_tool_observation.content,
                "trust": latest_tool_observation.trust.value,
                "provenance": latest_tool_observation.provenance.model_dump(mode="json"),
            }
            if latest_tool_observation is not None
            else None
        ),
        "mutation_verification": mutation_verification.model_dump(mode="json"),
        "provenance_refs": [p.model_dump(mode="json") for p in provenance_refs],
    }
    encoded = json.dumps(
        semantic_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def reconstruct(
    events: Sequence[NormalizedEvent],
    available_tools: Sequence[ExternalToolBinding] = (),
) -> WorkingState:
    """Reconstruct WorkingState deterministically from the full event sequence."""
    return reconstruct_from_checkpoint(
        checkpoint=None,
        new_events=events,
        available_tools=available_tools,
    )


def reconstruct_from_checkpoint(
    checkpoint: WorkingState | None,
    new_events: Sequence[NormalizedEvent],
    available_tools: Sequence[ExternalToolBinding] = (),
) -> WorkingState:
    """Apply new events on top of an optional prefix WorkingState checkpoint."""
    effective_tools = (
        tuple(available_tools)
        if available_tools
        else (checkpoint.available_tools if checkpoint is not None else ())
    )
    tool_capability_map: dict[str, CanonicalCapability] = {
        binding.external_name: binding.canonical_capability for binding in effective_tools
    }

    if checkpoint is None:
        event_count = 0
        last_seq: int | None = None
        user_messages: list[str] = []
        trusted_instructions: list[str] = []
        recent_actions: list[RecordedAction] = []
        pending_ids: list[str] = []
        latest_observation: ToolObservation | None = None
        mv_state = MutationVerificationState()
        provenance_refs: list[EventProvenance] = []
    else:
        event_count = checkpoint.event_count
        last_seq = checkpoint.last_sequence_no
        user_messages = list(checkpoint.user_messages)
        trusted_instructions = list(checkpoint.trusted_system_instructions)
        recent_actions = list(checkpoint.recent_actions)
        pending_ids = list(checkpoint.pending_tool_call_ids)
        latest_observation = checkpoint.latest_tool_observation
        mv_state = checkpoint.mutation_verification
        provenance_refs = list(checkpoint.provenance_refs)

    seen_tool_call_ids: set[str] = {action.tool_call_id for action in recent_actions}

    mutation_attempted = mv_state.mutation_attempted
    mutation_confirmed = mv_state.mutation_confirmed
    last_mutation_attempt_seq = mv_state.last_mutation_attempt_seq
    last_mutation_confirmed_seq = mv_state.last_mutation_confirmed_seq

    verification_attempted = mv_state.verification_attempted
    verification_confirmed = mv_state.verification_confirmed
    last_verification_attempt_seq = mv_state.last_verification_attempt_seq
    last_verification_confirmed_seq = mv_state.last_verification_confirmed_seq

    for event in new_events:
        expected_seq = 0 if last_seq is None else last_seq + 1
        if event.sequence_no != expected_seq:
            raise ProtocolError(
                f"Non-monotonic event sequence_no: expected {expected_seq}, "
                f"got {event.sequence_no}."
            )
        last_seq = event.sequence_no
        event_count += 1
        provenance_refs.append(event.provenance)

        if event.kind == EventKind.SYSTEM_MESSAGE:
            if event.trust != TrustLevel.SYSTEM_TRUSTED:
                raise InvariantViolation(
                    f"Cannot promote event with trust {event.trust} into "
                    "trusted_system_instructions."
                )
            trusted_instructions.append(event.content)

        elif event.kind == EventKind.USER_MESSAGE:
            if event.trust != TrustLevel.USER:
                raise InvariantViolation(
                    f"USER_MESSAGE event must have TrustLevel.USER, got {event.trust}."
                )
            user_messages.append(event.content)

        elif event.kind == EventKind.ASSISTANT_MESSAGE:
            if event.trust != TrustLevel.MODEL_GENERATED:
                raise InvariantViolation(
                    "ASSISTANT_MESSAGE event must have TrustLevel.MODEL_GENERATED, "
                    f"got {event.trust}."
                )

        elif event.kind == EventKind.TOOL_CALL:
            if event.trust != TrustLevel.MODEL_GENERATED:
                raise InvariantViolation(
                    f"TOOL_CALL event must have TrustLevel.MODEL_GENERATED, got {event.trust}."
                )
            if not event.tool_call_id or not event.tool_name or event.tool_arguments is None:
                raise InvariantViolation("TOOL_CALL event missing required tool fields.")
            if event.tool_call_id in seen_tool_call_ids:
                raise ProtocolError(
                    f"Duplicate tool_call_id '{event.tool_call_id}' during state reconstruction."
                )
            seen_tool_call_ids.add(event.tool_call_id)
            pending_ids.append(event.tool_call_id)

            capability = tool_capability_map.get(
                event.tool_name,
                resolve_tool_capability(event.tool_name),
            )
            recent_actions.append(
                RecordedAction(
                    sequence_no=event.sequence_no,
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    canonical_capability=capability,
                    arguments=dict(event.tool_arguments),
                    provenance=event.provenance,
                )
            )
            if capability in _MUTATION_CAPABILITIES:
                mutation_attempted = True
                last_mutation_attempt_seq = event.sequence_no
            elif capability in _VERIFICATION_CAPABILITIES:
                verification_attempted = True
                last_verification_attempt_seq = event.sequence_no

        elif event.kind == EventKind.TOOL_RESULT:
            if event.trust != TrustLevel.UNTRUSTED_EXTERNAL:
                raise InvariantViolation(
                    f"TOOL_RESULT event must have TrustLevel.UNTRUSTED_EXTERNAL, got {event.trust}."
                )
            if not event.tool_call_id or event.tool_call_id not in pending_ids:
                raise ProtocolError(
                    f"Orphan or duplicate TOOL_RESULT for tool_call_id '{event.tool_call_id}'."
                )
            origin_actions = [a for a in recent_actions if a.tool_call_id == event.tool_call_id]
            if (
                origin_actions
                and event.tool_name is not None
                and event.tool_name != origin_actions[0].tool_name
            ):
                raise ProtocolError(
                    f"TOOL_RESULT tool_name '{event.tool_name}' does not match originating "
                    f"TOOL_CALL tool_name '{origin_actions[0].tool_name}'."
                )
            pending_ids.remove(event.tool_call_id)
            latest_observation = ToolObservation(
                sequence_no=event.sequence_no,
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                content=event.content,
                trust=TrustLevel.UNTRUSTED_EXTERNAL,
                provenance=event.provenance,
            )

        elif event.kind == EventKind.MUTATION:
            mutation_attempted = True
            mutation_confirmed = True
            last_mutation_confirmed_seq = event.sequence_no

        elif event.kind == EventKind.VERIFICATION:
            verification_attempted = True
            verification_confirmed = True
            last_verification_confirmed_seq = event.sequence_no

    updated_mv = MutationVerificationState(
        mutation_attempted=mutation_attempted,
        mutation_confirmed=mutation_confirmed,
        last_mutation_attempt_seq=last_mutation_attempt_seq,
        last_mutation_confirmed_seq=last_mutation_confirmed_seq,
        verification_attempted=verification_attempted,
        verification_confirmed=verification_confirmed,
        last_verification_attempt_seq=last_verification_attempt_seq,
        last_verification_confirmed_seq=last_verification_confirmed_seq,
    )

    initial_user_req = user_messages[0] if user_messages else None
    latest_user_req = user_messages[-1] if user_messages else None

    digest = compute_state_digest(
        event_count=event_count,
        last_sequence_no=last_seq,
        initial_user_request=initial_user_req,
        latest_user_request=latest_user_req,
        user_messages=user_messages,
        trusted_system_instructions=trusted_instructions,
        available_tools=effective_tools,
        recent_actions=recent_actions,
        pending_tool_call_ids=pending_ids,
        latest_tool_observation=latest_observation,
        mutation_verification=updated_mv,
        provenance_refs=provenance_refs,
    )

    return WorkingState(
        state_digest=digest,
        event_count=event_count,
        last_sequence_no=last_seq,
        initial_user_request=initial_user_req,
        latest_user_request=latest_user_req,
        user_messages=tuple(user_messages),
        trusted_system_instructions=tuple(trusted_instructions),
        available_tools=effective_tools,
        recent_actions=tuple(recent_actions),
        pending_tool_call_ids=tuple(pending_ids),
        latest_tool_observation=latest_observation,
        mutation_verification=updated_mv,
        provenance_refs=tuple(provenance_refs),
    )
