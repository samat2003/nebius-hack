"""Replay determinism tests separating ReplaySemantics from ReplayTelemetry."""

from __future__ import annotations

from pathlib import Path

from alienese.api.models import ChatCompletionRequest
from alienese.contracts.context import RequestContext
from alienese.engine.reconstruct import reconstruct
from alienese.engine.turn import TurnEngine
from alienese.providers.fake import FakeController, FakeGenerator, FakeRetriever
from alienese.storage.traces import dump_replay_artifact_json, load_replay_artifact_json

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
    stored_artifact = load_replay_artifact_json(raw_fixture)

    # 1. Reconstruct WorkingState purely from stored normalized events + tools
    reconstructed_state = reconstruct(
        stored_artifact.semantics.normalized_events,
        available_tools=stored_artifact.semantics.available_tools,
    )
    assert reconstructed_state.state_digest == stored_artifact.semantics.state_digest

    # 2. Execute full TurnEngine with completely different attempt/telemetry IDs
    engine = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=FakeGenerator(),
    )
    request = ChatCompletionRequest.model_validate(SYNTHETIC_REQUEST_PAYLOAD)
    new_ctx = RequestContext(
        request_id="req_replay_verification_999",
        operation_id="op_replay_verification_999",
        correlation_trace_id="trc_replay_verification_999",
    )
    _response, fresh_artifact = await engine.execute_turn(new_ctx, request)

    # Telemetry differs across runs/attempts, but ReplaySemantics are 100% identical
    assert fresh_artifact.telemetry != stored_artifact.telemetry
    assert fresh_artifact.is_semantically_equivalent(stored_artifact)
    assert fresh_artifact.semantics == stored_artifact.semantics


async def test_replay_artifact_json_roundtrip_is_stable() -> None:
    engine = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=FakeGenerator(),
    )
    request = ChatCompletionRequest.model_validate(SYNTHETIC_REQUEST_PAYLOAD)
    ctx = RequestContext(
        request_id="req_fixture_seed",
        operation_id="op_fixture_seed",
        correlation_trace_id="trc_fixture_seed",
    )
    _resp, artifact = await engine.execute_turn(ctx, request)

    dumped = dump_replay_artifact_json(artifact)
    reloaded = load_replay_artifact_json(dumped)
    assert reloaded == artifact
    assert dump_replay_artifact_json(reloaded) == dumped
