"""Comprehensive regression tests for the Final Phase 0/1 Correctness Review (Items 1-8)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from alienese.api.app import create_app
from alienese.api.errors import CompatibilityError, InvariantViolation
from alienese.api.models import ChatCompletionRequest
from alienese.config import Settings
from alienese.contracts.candidates import RiskClass
from alienese.contracts.context import RequestContext
from alienese.contracts.decisions import ProviderCallTelemetry
from alienese.contracts.generation import (
    GenerationJob,
    GenerationResult,
    GenerationSemantics,
)
from alienese.contracts.providers import RetrievalRequest, RetrievalResult
from alienese.contracts.state import CanonicalCapability
from alienese.contracts.traces import TraceMode
from alienese.engine.turn import (
    MAX_SCHEMA_DEPTH,
    TurnEngine,
    classify_tool_risk,
    validate_tool_arguments_against_schema,
)
from alienese.providers.fake import FakeController, FakeGenerator, FakeRetriever
from alienese.storage.idempotency import IdempotencyRecord, InMemoryIdempotencyStore
from alienese.storage.traces import (
    InMemoryTraceStore,
    reconstruct_from_replay_artifact,
)


# ---------------------------------------------------------------------------
# 1. Preserve external protocol semantics (no redaction on outgoing API payloads)
# ---------------------------------------------------------------------------
class _LiteralGenerator:
    """Test generator returning exact caller-specified text to verify non-redaction."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def generate(self, ctx: RequestContext, job: GenerationJob) -> GenerationResult:
        return GenerationResult(
            semantics=GenerationSemantics(
                job_id=job.job_id,
                job_type=job.job_type,
                content=self._text,
                provider_name="literal_generator",
                model_id="literal-v1",
                model_revision="v1",
            ),
            telemetry=ProviderCallTelemetry(latency_ms=1.0, request_attempt_id=ctx.request_id),
        )


async def test_1_outgoing_tool_arguments_and_assistant_content_never_redacted() -> None:
    """Valid JSON tool arguments and assistant output survive serialization unchanged."""
    token_like_val = "sk-" + "proj_legitimate_test_fixture_token_123456"
    bearer_like_val = "Bearer " + "eyJhbGciOiJIUzI1NiJ9.legitimate_test_token"

    tool_with_sensitive_looking_defaults: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file with auth header metadata",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "configs/auth_fixture.json"},
                    "token": {"type": "string", "default": token_like_val},
                    "authorization": {"type": "string", "default": bearer_like_val},
                },
                "required": ["path", "token", "authorization"],
            },
        },
    }

    engine = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=FakeGenerator(),
    )
    req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "Read the auth fixture"}],
            "tools": [tool_with_sensitive_looking_defaults],
            "tool_choice": {"type": "function", "function": {"name": "read_file"}},
        }
    )
    resp, _artifact = await engine.execute_turn(RequestContext(), req)
    assert resp.choices[0].finish_reason == "tool_calls"
    assert resp.choices[0].message.tool_calls is not None
    raw_args = resp.choices[0].message.tool_calls[0].function.arguments
    parsed_args = json.loads(raw_args)
    assert parsed_args == {
        "path": "configs/auth_fixture.json",
        "token": token_like_val,
        "authorization": bearer_like_val,
    }
    assert "[REDACTED]" not in raw_args

    # Legitimate assistant output containing token-like syntax must also not be mutated
    expected_assistant_text = f"Example header format: Authorization: {bearer_like_val}"
    engine_with_literal_gen = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=_LiteralGenerator(expected_assistant_text),
    )
    resp2, _ = await engine_with_literal_gen.execute_turn(
        RequestContext(),
        ChatCompletionRequest.model_validate(
            {
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Show header example"}],
            }
        ),
    )
    assert resp2.choices[0].message.content == expected_assistant_text


async def test_1_fake_generator_is_non_reflective_by_default() -> None:
    """FakeGenerator must produce synthetic output rather than echoing sensitive input."""
    engine = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=FakeGenerator(),
    )
    unique_marker = "UNIQUE_PRIVATE_USER_PROMPT_MARKER_987654321"
    resp, _ = await engine.execute_turn(
        RequestContext(),
        ChatCompletionRequest.model_validate(
            {
                "model": "alienese-default",
                "messages": [{"role": "user", "content": f"Please echo {unique_marker}"}],
            }
        ),
    )
    content = resp.choices[0].message.content or ""
    assert unique_marker not in content
    assert content.startswith("[fake:answer] Synthetic response for job_")


# ---------------------------------------------------------------------------
# 2. Separate privacy-safe telemetry from replayable content
# ---------------------------------------------------------------------------
async def test_2_default_telemetry_retains_no_private_content_and_marks_non_replayable() -> None:
    """Default telemetry retains zero raw prompts/tool outputs and marks replayable=False."""
    private_prompt = "CONFIDENTIAL_SOURCE_CODE_SNIPPET_ALPHA_777"
    private_tool_output = "INTERNAL_FILE_CONTENTS_BETA_888"

    trace_store = InMemoryTraceStore()
    engine = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=FakeGenerator(),
        trace_store=trace_store,
        include_replay_content=False,
    )
    req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [
                {"role": "system", "content": "INTERNAL_SYSTEM_POLICY_GAMMA_999"},
                {"role": "user", "content": private_prompt},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_priv_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"secret_dir/private.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_priv_1",
                    "content": private_tool_output,
                },
            ],
        }
    )
    ctx = RequestContext()
    _resp, artifact = await engine.execute_turn(ctx, req)

    assert artifact.trace_mode == TraceMode.METADATA_ONLY
    assert artifact.replayable is False
    assert artifact.semantics.normalized_events == ()
    assert artifact.semantics.available_tools == ()
    assert artifact.semantics.candidates == ()
    assert artifact.semantics.generation_job is None
    assert artifact.semantics.generation_semantics is None
    assert artifact.semantics.logical_response is None
    assert artifact.semantics.event_count == 4
    assert len(artifact.semantics.event_digests) == 4

    serialized = json.dumps(artifact.model_dump(mode="json"))
    assert private_prompt not in serialized
    assert private_tool_output not in serialized
    assert "INTERNAL_SYSTEM_POLICY_GAMMA_999" not in serialized
    assert "secret_dir/private.py" not in serialized

    with pytest.raises(InvariantViolation):
        reconstruct_from_replay_artifact(artifact)


async def test_2_full_fidelity_replay_preserves_state_and_secret_downgrades_to_metadata() -> None:
    """Full-fidelity replay preserves state digest; secret detection downgrades mode."""
    engine = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=FakeGenerator(),
        include_replay_content=True,
    )
    clean_req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "Clean local evaluation prompt"}],
        }
    )
    _resp, full_artifact = await engine.execute_turn(RequestContext(), clean_req)
    assert full_artifact.trace_mode == TraceMode.FULL_FIDELITY_REPLAY
    assert full_artifact.replayable is True

    reconstructed = reconstruct_from_replay_artifact(full_artifact)
    assert reconstructed.state_digest == full_artifact.semantics.state_digest

    # If secret material is present in the conversation, sanitize_replay_artifact
    # downgrades to METADATA_ONLY (replayable=False) rather than mutating events
    # and leaving a broken state_digest!
    secret_val = "sk-" + "live_secret_for_trace_downgrade_test_123456"
    secret_req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": f"Prompt containing {secret_val}"}],
        }
    )
    _resp2, downgraded_artifact = await engine.execute_turn(RequestContext(), secret_req)
    assert downgraded_artifact.trace_mode == TraceMode.METADATA_ONLY
    assert downgraded_artifact.replayable is False
    assert secret_val not in json.dumps(downgraded_artifact.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# 3. Fix the retrieval boundary
# ---------------------------------------------------------------------------
class _SpyRetriever:
    def __init__(self) -> None:
        self.call_count = 0

    async def rank(self, ctx: RequestContext, request: RetrievalRequest) -> RetrievalResult:
        self.call_count += 1
        raise AssertionError("Retriever.rank() must not be called in Phase 1 TurnEngine")


async def test_3_turn_engine_bypasses_unused_retrieval_in_phase_1() -> None:
    """TurnEngine must not invoke Retriever.rank() in Phase 1."""
    spy_retriever = _SpyRetriever()
    engine = TurnEngine(
        retriever=spy_retriever,
        controller=FakeController(),
        generator=FakeGenerator(),
    )
    req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "Check retrieval bypass"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "parameters": {
                            "type": "object",
                            "properties": {"directory": {"type": "string", "default": "."}},
                        },
                    },
                }
            ],
        }
    )
    resp, _ = await engine.execute_turn(RequestContext(), req)
    assert resp.choices[0].finish_reason == "stop"
    assert spy_retriever.call_count == 0


# ---------------------------------------------------------------------------
# 4. Make fake-mode action selection safe & classify tool risk conservatively
# ---------------------------------------------------------------------------
async def test_4_conservative_tool_risk_and_fake_action_selection() -> None:
    """Unknown/command/mutation tools are HIGH risk and never auto-selected."""
    assert classify_tool_risk(CanonicalCapability.READ_FILE) == RiskClass.LOW
    assert classify_tool_risk(CanonicalCapability.SEARCH_TEXT) == RiskClass.LOW
    assert classify_tool_risk(CanonicalCapability.LIST_FILES) == RiskClass.LOW
    assert classify_tool_risk(CanonicalCapability.RUN_TEST) == RiskClass.MEDIUM
    assert classify_tool_risk(CanonicalCapability.RUN_COMMAND) == RiskClass.HIGH
    assert classify_tool_risk(CanonicalCapability.WRITE_FILE) == RiskClass.HIGH
    assert classify_tool_risk(CanonicalCapability.APPLY_PATCH) == RiskClass.HIGH
    assert classify_tool_risk(CanonicalCapability.CUSTOM_TOOL) == RiskClass.HIGH
    assert classify_tool_risk(CanonicalCapability.UNKNOWN) == RiskClass.HIGH

    dangerous_tools_with_defaults: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Execute shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string", "default": "rm -rf /"}},
                    "required": ["cmd"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "default": "pwned.txt"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "deploy_arbitrary_custom_action",
                "description": "Unknown custom tool with default",
                "parameters": {
                    "type": "object",
                    "properties": {"env": {"type": "string", "default": "prod"}},
                    "required": ["env"],
                },
            },
        },
    ]

    engine = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=FakeGenerator(),
        include_replay_content=True,
    )

    # In 'auto' mode, FakeController must select cand_answer ('stop'), NEVER a tool with defaults
    resp, artifact = await engine.execute_turn(
        RequestContext(),
        ChatCompletionRequest.model_validate(
            {
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Run something"}],
                "tools": dangerous_tools_with_defaults,
                "tool_choice": "auto",
            }
        ),
    )
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.tool_calls is None
    for cand_digest in artifact.semantics.candidate_digests:
        if cand_digest.external_tool_name is not None:
            assert cand_digest.risk_class == RiskClass.HIGH

    # In generic 'required' mode, high-risk tools with defaults are still rejected
    with pytest.raises(CompatibilityError) as exc_info:
        await engine.execute_turn(
            RequestContext(),
            ChatCompletionRequest.model_validate(
                {
                    "model": "alienese-default",
                    "messages": [{"role": "user", "content": "Force any tool"}],
                    "tools": dangerous_tools_with_defaults,
                    "tool_choice": "required",
                }
            ),
        )
    assert exc_info.value.code == "ungrounded_required_tool_arguments"


# ---------------------------------------------------------------------------
# 5. Correct external model identity
# ---------------------------------------------------------------------------
async def test_5_strict_external_model_identity() -> None:
    """Expose only alienese-default on /v1/models and reject unknown/internal model IDs."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        models_resp = await client.get("/v1/models")
        assert models_resp.status_code == 200
        model_ids = [m["id"] for m in models_resp.json()["data"]]
        assert model_ids == ["alienese-default"]

        for unsupported_model in ("gpt-4o", "samatv256/mini-Jev", "nvidia/nemotron"):
            bad_resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": unsupported_model,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert bad_resp.status_code == 400
            err = bad_resp.json()["error"]
            assert err["type"] == "compatibility_error"
            assert err["code"] == "model_not_supported"
            assert err["param"] == "model"


# ---------------------------------------------------------------------------
# 6. Fix response timestamps and replay comparison
# ---------------------------------------------------------------------------
async def test_6_response_timestamps_and_idempotent_preservation() -> None:
    """Responses use real timestamps, preserve them on retry, and exclude them from semantics."""
    current_ts = 1760000100

    def _advancing_clock() -> int:
        nonlocal current_ts
        current_ts += 10
        return current_ts

    engine = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=FakeGenerator(),
        clock=_advancing_clock,
    )
    req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "Check timestamp behavior"}],
        }
    )
    r1, art1 = await engine.execute_turn(
        RequestContext(request_id="req_1", operation_id="op_1"), req
    )
    r2, art2 = await engine.execute_turn(
        RequestContext(request_id="req_2", operation_id="op_2"), req
    )

    assert r1.created == 1760000110
    assert r2.created == 1760000120
    assert r1.created != r2.created
    assert art1.telemetry.created_timestamp == 1760000110
    assert art2.telemetry.created_timestamp == 1760000120

    # ReplaySemantics equivalence is unaffected by volatile timestamps or request IDs
    assert art1.is_semantically_equivalent(art2)


# ---------------------------------------------------------------------------
# 7. Bound resource consumption (stores, body size, schema depth/keywords, loopback)
# ---------------------------------------------------------------------------
async def test_7_bounded_idempotency_and_trace_stores() -> None:
    """InMemoryIdempotencyStore and InMemoryTraceStore enforce max_entries, TTL, and lock bounds."""
    now = 100.0

    def _clock() -> float:
        return now

    idem_store = InMemoryIdempotencyStore(max_entries=2, ttl_seconds=10.0, clock=_clock)
    engine = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=FakeGenerator(),
    )
    req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "Store bounds test"}],
        }
    )
    sample_resp, sample_art = await engine.execute_turn(RequestContext(), req)

    for i in range(5):
        _ = idem_store.key_lock(f"lock_only_{i}")
    assert idem_store.lock_count <= 2

    await idem_store.put(
        IdempotencyRecord(
            idempotency_key="k1",
            request_fingerprint="a" * 64,
            operation_id="op_1",
            first_request_id="req_1",
            response=sample_resp,
        )
    )
    await idem_store.put(
        IdempotencyRecord(
            idempotency_key="k2",
            request_fingerprint="b" * 64,
            operation_id="op_2",
            first_request_id="req_2",
            response=sample_resp,
        )
    )
    await idem_store.put(
        IdempotencyRecord(
            idempotency_key="k3",
            request_fingerprint="c" * 64,
            operation_id="op_3",
            first_request_id="req_3",
            response=sample_resp,
        )
    )
    assert idem_store.size == 2
    assert await idem_store.get("k1") is None
    assert await idem_store.get("k2") is not None
    assert await idem_store.get("k3") is not None

    # Advance clock past TTL
    now += 15.0
    assert idem_store.size == 0
    assert await idem_store.get("k2") is None

    # TraceStore LRU + TTL
    trace_store = InMemoryTraceStore(max_entries=2, ttl_seconds=10.0, clock=_clock)
    for idx in range(3):
        art = sample_art.model_copy(
            update={
                "telemetry": sample_art.telemetry.model_copy(
                    update={"operation_id": f"op_trace_{idx}", "request_id": f"req_trace_{idx}"}
                )
            }
        )
        await trace_store.save(art)
    assert trace_store.size == 2
    assert await trace_store.get_by_operation_id("op_trace_0") is None
    assert await trace_store.get_by_operation_id("op_trace_1") is not None
    assert await trace_store.get_by_operation_id("op_trace_2") is not None

    now += 15.0
    assert trace_store.size == 0


async def test_7_request_body_size_limit_and_loopback_default() -> None:
    """Oversized request bodies return 413 and non-loopback hosts are rejected by default."""
    assert Settings().alienese_host == "127.0.0.1"
    with pytest.raises(CompatibilityError) as host_exc:
        Settings(alienese_host="0.0.0.0")
    assert host_exc.value.code == "non_loopback_binding_forbidden"

    # Explicit opt-in allows non-loopback when intended
    explicit_non_loopback = Settings(
        alienese_host="0.0.0.0",
        alienese_allow_non_loopback=True,
    )
    assert explicit_non_loopback.alienese_host == "0.0.0.0"

    small_body_settings = Settings(max_request_body_bytes=1024)
    app = create_app(settings=small_body_settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        oversized_resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "x" * 2048}],
            },
        )
        assert oversized_resp.status_code == 413
        assert oversized_resp.json()["error"]["code"] == "request_body_too_large"


def test_7_schema_recursion_depth_and_unsupported_keywords_rejected() -> None:
    """Deeply nested schemas and unsupported JSON Schema keywords fail closed."""
    for unsupported_kw in ("$ref", "oneOf", "anyOf", "allOf", "not", "patternProperties"):
        schema_with_unsupported = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            unsupported_kw: {},
        }
        ok, reason = validate_tool_arguments_against_schema(
            schema_with_unsupported,
            {"x": "hello"},
        )
        assert ok is False
        assert reason is not None
        assert unsupported_kw in reason

    # Construct nested schema deeper than MAX_SCHEMA_DEPTH
    deep_schema: dict[str, Any] = {"type": "string"}
    deep_value: Any = "leaf"
    for _ in range(MAX_SCHEMA_DEPTH + 3):
        deep_schema = {
            "type": "object",
            "properties": {"nested": deep_schema},
            "required": ["nested"],
        }
        deep_value = {"nested": deep_value}

    ok_deep, reason_deep = validate_tool_arguments_against_schema(deep_schema, deep_value)
    assert ok_deep is False
    assert reason_deep is not None
    assert "recursion depth" in reason_deep


# ---------------------------------------------------------------------------
# 8. Merge Gate 1: Idempotency concurrency, waiter scheduling, & admission control
# ---------------------------------------------------------------------------
async def test_8_idempotency_concurrency_waiter_scheduling_and_admission_control() -> None:
    """Locks with queued waiters are never evicted; capacity overflow uses admission control."""
    import asyncio

    from alienese.api.errors import ProviderUnavailable
    from alienese.contracts.state import ExternalToolBinding
    from alienese.engine.idempotency import IdempotencyCoordinator
    from alienese.engine.turn import extract_deterministic_tool_arguments

    _ = (extract_deterministic_tool_arguments, ExternalToolBinding)

    store = InMemoryIdempotencyStore(max_entries=2, ttl_seconds=60.0)
    coordinator = IdempotencyCoordinator(store)

    engine = TurnEngine(
        retriever=FakeRetriever(),
        controller=FakeController(),
        generator=FakeGenerator(),
    )
    req_alpha = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "Alpha operation"}],
        }
    )
    req_beta = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "Beta operation"}],
        }
    )
    req_gamma = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "Gamma operation"}],
        }
    )

    alpha_started = asyncio.Event()
    alpha_release = asyncio.Event()
    beta_started = asyncio.Event()
    beta_release = asyncio.Event()

    alpha_executions = 0
    alpha_in_flight = 0
    alpha_max_in_flight = 0

    async def _alpha_runner(turn_ctx: RequestContext) -> Any:
        nonlocal alpha_executions, alpha_in_flight, alpha_max_in_flight
        alpha_executions += 1
        alpha_in_flight += 1
        alpha_max_in_flight = max(alpha_max_in_flight, alpha_in_flight)
        alpha_started.set()
        try:
            await alpha_release.wait()
            resp, _ = await engine.execute_turn(turn_ctx, req_alpha)
            return resp
        finally:
            alpha_in_flight -= 1

    async def _beta_runner(turn_ctx: RequestContext) -> Any:
        beta_started.set()
        await beta_release.wait()
        resp, _ = await engine.execute_turn(turn_ctx, req_beta)
        return resp

    # 1. Start Holder (t1) for key "alpha"
    t1 = asyncio.create_task(
        coordinator.execute(
            RequestContext(request_id="req_a1", operation_id="op_a1", idempotency_key="alpha"),
            req_alpha,
            _alpha_runner,
        )
    )
    await alpha_started.wait()
    alpha_lock_initial = store.key_lock("alpha")

    # 2. Queue Waiter (t2) for key "alpha" and let it suspend on lock.__aenter__
    t2 = asyncio.create_task(
        coordinator.execute(
            RequestContext(request_id="req_a2", operation_id="op_a2", idempotency_key="alpha"),
            req_alpha,
            _alpha_runner,
        )
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # 3. Start Holder (t3) for key "beta", filling max_entries=2 active locks
    t3 = asyncio.create_task(
        coordinator.execute(
            RequestContext(request_id="req_b1", operation_id="op_b1", idempotency_key="beta"),
            req_beta,
            _beta_runner,
        )
    )
    await beta_started.wait()
    assert store.lock_count == 2

    # 4. While both "alpha" (holder + waiter) and "beta" (holder) are active,
    #    a 3rd concurrent key "gamma" triggers bounded admission control (HTTP 503)
    #    rather than evicting an active or waited lock.
    with pytest.raises(ProviderUnavailable) as cap_exc:
        await coordinator.execute(
            RequestContext(request_id="req_g1", operation_id="op_g1", idempotency_key="gamma"),
            req_gamma,
            _beta_runner,
        )
    assert cap_exc.value.status_code == 503
    assert cap_exc.value.code == "idempotency_capacity_exceeded"
    assert store.key_lock("alpha") is alpha_lock_initial

    # 5. Release "beta" so one slot becomes idle, then release "alpha" holder (t1)
    #    while simultaneously scheduling another duplicate "alpha" request (t4)
    #    and a new key "delta" (t5) during the waiter handoff window!
    beta_release.set()
    await t3

    alpha_release.set()
    # Immediately (before t2 wakes up on the next loop tick), request key_lock("delta")
    # which triggers _prune_idle_locks(). "alpha" has a queued waiter (t2) so its lock
    # MUST NOT be evicted even if _locked is momentarily False during handoff!
    _ = store.key_lock("delta")
    assert store.key_lock("alpha") is alpha_lock_initial

    t4 = asyncio.create_task(
        coordinator.execute(
            RequestContext(request_id="req_a3", operation_id="op_a3", idempotency_key="alpha"),
            req_alpha,
            _alpha_runner,
        )
    )

    res1, res2, res4 = await asyncio.gather(t1, t2, t4)
    assert alpha_executions == 1
    assert alpha_max_in_flight == 1
    assert res1[2] is False  # first execution
    assert res2[2] is True  # replayed to waiter t2
    assert res4[2] is True  # replayed to t4
    assert res1[1] == res2[1] == res4[1]


# ---------------------------------------------------------------------------
# 9. Merge Gate 2: Recursive fail-closed JSON Schema validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("bad_schema", "sample_args", "expected_fragment"),
    [
        (
            {
                "type": "object",
                "properties": {"name": {"type": "string", "pattern": "^[a-z]+$"}},
            },
            {"name": "alice"},
            "pattern",
        ),
        (
            {
                "type": "object",
                "properties": {"email": {"type": "string", "format": "email"}},
            },
            {"email": "a@b.com"},
            "format",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    }
                },
            },
            {"tags": ["a", "b"]},
            "uniqueItems",
        ),
        (
            {
                "type": "object",
                "properties": {"score": {"type": "number", "exclusiveMinimum": 0}},
            },
            {"score": 1.5},
            "exclusiveMinimum",
        ),
        # Nested unsupported keyword inside an OPTIONAL property omitted from arguments={}
        (
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "README.md"},
                    "unused_nested": {
                        "type": "object",
                        "properties": {
                            "deep": {"type": "string", "pattern": ".*"},
                        },
                    },
                },
            },
            {"path": "README.md"},
            "pattern",
        ),
        # Nested unsupported keyword inside array `items` even when array is empty `[]`
        (
            {
                "type": "object",
                "properties": {
                    "items_list": {
                        "type": "array",
                        "default": [],
                        "items": {"type": "string", "format": "uri"},
                    }
                },
            },
            {"items_list": []},
            "format",
        ),
        # Nested unsupported keyword inside `additionalProperties`
        (
            {
                "type": "object",
                "additionalProperties": {"type": "integer", "exclusiveMaximum": 100},
            },
            {},
            "exclusiveMaximum",
        ),
        # Malformed schema structures
        (
            {
                "type": "object",
                "properties": {"bad_prop": "not_a_schema_object"},
            },
            {},
            "must be a JSON object",
        ),
        (
            {
                "type": "object",
                "properties": {"x": {"type": "string", "minLength": -1}},
            },
            {"x": "abc"},
            "minLength",
        ),
        (
            {
                "type": "object",
                "properties": {"x": {"type": "string", "minLength": 10, "maxLength": 3}},
            },
            {"x": "abcde"},
            "minLength > maxLength",
        ),
        (
            {
                "type": "object",
                "properties": {"x": {"type": "number", "minimum": 10, "maximum": 2}},
            },
            {"x": 5},
            "minimum > maximum",
        ),
        (
            {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x", "x"],
            },
            {"x": "ok"},
            "required",
        ),
    ],
)
async def test_9_json_schema_fail_closed_on_unsupported_keywords_and_malformed_schemas(
    bad_schema: dict[str, Any],
    sample_args: dict[str, Any],
    expected_fragment: str,
) -> None:
    """Unsupported keywords and malformed schemas fail closed recursively."""
    from alienese.contracts.state import ExternalToolBinding
    from alienese.engine.normalize import normalize_request
    from alienese.engine.reconstruct import reconstruct
    from alienese.engine.turn import (
        build_deterministic_candidates,
        extract_deterministic_tool_arguments,
    )

    ok, reason = validate_tool_arguments_against_schema(bad_schema, sample_args)
    assert ok is False
    assert reason is not None
    assert expected_fragment in reason

    binding = ExternalToolBinding(
        external_name="read_file",
        description="Tool with unsupported or malformed schema",
        parameters_schema=bad_schema,
        canonical_capability=CanonicalCapability.READ_FILE,
    )
    extracted, is_valid = extract_deterministic_tool_arguments(binding)
    assert extracted == {}
    assert is_valid is False

    # Ensure build_deterministic_candidates never produces an executable tool candidate
    req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "Test malformed tool schema"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": bad_schema,
                    },
                }
            ],
        }
    )
    events, tools = normalize_request(req)
    state = reconstruct(events, available_tools=tools)
    candidates = build_deterministic_candidates(state, req)
    tool_cands = [c for c in candidates if c.external_tool_name == "read_file"]
    assert len(tool_cands) == 1
    assert tool_cands[0].arguments_complete is False

    # And explicit tool_choice on a tool with an unsupported/malformed schema fails closed
    req_forced = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "Force malformed tool schema"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": bad_schema,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "read_file"}},
        }
    )
    with pytest.raises(CompatibilityError):
        build_deterministic_candidates(state, req_forced)


def test_9_json_schema_boolean_vs_integer_strict_enum_and_const() -> None:
    """Boolean values must not match integer enum/const values (True != 1, False != 0)."""
    int_enum_schema = {
        "type": "object",
        "properties": {"flag": {"enum": [0, 1]}},
        "required": ["flag"],
    }
    ok_bool, _ = validate_tool_arguments_against_schema(int_enum_schema, {"flag": True})
    assert ok_bool is False
    ok_int, _ = validate_tool_arguments_against_schema(int_enum_schema, {"flag": 1})
    assert ok_int is True

    int_const_schema = {
        "type": "object",
        "properties": {"flag": {"const": 1}},
        "required": ["flag"],
    }
    ok_const_bool, _ = validate_tool_arguments_against_schema(int_const_schema, {"flag": True})
    assert ok_const_bool is False
