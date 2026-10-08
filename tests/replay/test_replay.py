"""Replay determinism tests separating privacy-safe telemetry from full-fidelity replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from alienese.api.errors import InvariantViolation
from alienese.api.models import ChatCompletionRequest
from alienese.contracts.context import RequestContext
from alienese.contracts.traces import TraceMode
from alienese.engine.reconstruct import reconstruct
from alienese.engine.turn import TurnEngine
from alienese.providers.fake import FakeController, FakeGenerator, FakeRetriever
from alienese.storage.traces import (
    dump_replay_artifact_json,
    load_replay_artifact_json,
    reconstruct_from_replay_artifact,
)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "synthetic_turn_replay.json"

SYNTHETIC_REQUEST_PAYLOAD = {
    "model": "alienese-default",
    "messages": [
        {
            "role": "system",
            "content": "Synthetic system policy for offline replay verification.",
        },
        {
            "role": "user",
            "content": "Synthetic user task: verify deterministic state and decision replay.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_synth_01",
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "arguments": '{"directory":"."}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_synth_01",
            "content": "src/example.py\ntests/test_example.py",
        },
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "Synthetic directory listing tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "default": "."},
                    },
                    "required": ["directory"],
                },
            },
        }
    ],
    "temperature": 0.0,
    "max_tokens": 64,
}


async def test_stored_synthetic_replay_fixture_matches_deterministic_execution() -> None:
    raw_fixture = FIXTURE_PATH.read_text(encoding="utf-8")
    stored_artifact = load_replay_artifact_json(raw_fixture, allow_full_fidelity=True)
    assert stored_artifact.trace_mode == TraceMode.FULL_FIDELITY_REPLAY
    assert stored_artifact.replayable is True

    # 1. Reconstruct WorkingState purely from stored normalized events + tools
    reconstructed_state = reconstruct_from_replay_artifact(stored_artifact)
    assert reconstructed_state.state_digest == stored_artifact.semantics.state_digest
    assert (
        reconstruct(
            stored_artifact.semantics.normalized_events,
            available_tools=stored_artifact.semantics.available_tools,
        ).state_digest
        == stored_artifact.semantics.state_digest
    )

    # 2. Execute full TurnEngine with completely different attempt/telemetry IDs and clock
    engine = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=FakeGenerator(),
        include_replay_content=True,
        clock=lambda: 1800000999,
    )
    request = ChatCompletionRequest.model_validate(SYNTHETIC_REQUEST_PAYLOAD)
    new_ctx = RequestContext(
        request_id="req_replay_verification_999",
        operation_id="op_replay_verification_999",
        correlation_trace_id="trc_replay_verification_999",
    )
    _response, fresh_artifact = await engine.execute_turn(new_ctx, request)

    # Telemetry (including created_timestamp and request_id) differs across runs,
    # but ReplaySemantics are 100% identical.
    assert fresh_artifact.telemetry != stored_artifact.telemetry
    assert fresh_artifact.is_semantically_equivalent(stored_artifact)
    assert fresh_artifact.semantics == stored_artifact.semantics


async def test_replay_artifact_json_roundtrip_and_default_metadata_mode() -> None:
    # Full-fidelity roundtrip when explicitly enabled
    replay_engine = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=FakeGenerator(),
        include_replay_content=True,
    )
    request = ChatCompletionRequest.model_validate(SYNTHETIC_REQUEST_PAYLOAD)
    ctx = RequestContext(
        request_id="req_fixture_seed",
        operation_id="op_fixture_seed",
        correlation_trace_id="trc_fixture_seed",
    )
    _resp, artifact = await replay_engine.execute_turn(ctx, request)
    assert artifact.replayable is True

    dumped = dump_replay_artifact_json(artifact, allow_full_fidelity=True)
    reloaded = load_replay_artifact_json(dumped, allow_full_fidelity=True)
    assert reloaded == artifact
    assert dump_replay_artifact_json(reloaded, allow_full_fidelity=True) == dumped

    # Default dump converts to METADATA_ONLY (replayable=False)
    metadata_dumped = dump_replay_artifact_json(artifact)
    metadata_loaded = load_replay_artifact_json(metadata_dumped)
    assert metadata_loaded.trace_mode == TraceMode.METADATA_ONLY
    assert metadata_loaded.replayable is False
    with pytest.raises(InvariantViolation):
        reconstruct_from_replay_artifact(metadata_loaded)
