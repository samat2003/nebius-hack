"""Contract tests for Alienese core domain models and provider protocols."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from alienese.api.errors import InvariantViolation
from alienese.config import Settings
from alienese.contracts import (
    CandidateAction,
    CandidateDisposition,
    CanonicalCapability,
    Controller,
    CostClass,
    DecisionResult,
    DecisionSemantics,
    EventKind,
    EventProvenance,
    ExternalToolBinding,
    GenerationJob,
    GenerationJobType,
    GenerationResult,
    GenerationSemantics,
    Generator,
    MutationVerificationState,
    NormalizedEvent,
    ProviderCallTelemetry,
    RequestContext,
    Retriever,
    RiskClass,
    SourceRole,
    ToolObservation,
    TrustLevel,
    WorkingState,
)
from alienese.observability.redaction import REDACTED_PLACEHOLDER
from alienese.providers.fake import FakeController, FakeGenerator, FakeRetriever


def test_event_trust_classifications_are_enforced() -> None:
    sys_event = NormalizedEvent(
        sequence_no=0,
        event_id="evt_0",
        kind=EventKind.SYSTEM_MESSAGE,
        trust=TrustLevel.SYSTEM_TRUSTED,
        provenance=EventProvenance(message_index=0, source_role=SourceRole.SYSTEM),
        content="Follow repository conventions.",
    )
    assert sys_event.trust == TrustLevel.SYSTEM_TRUSTED

    dev_event = NormalizedEvent(
        sequence_no=1,
        event_id="evt_1",
        kind=EventKind.SYSTEM_MESSAGE,
        trust=TrustLevel.SYSTEM_TRUSTED,
        provenance=EventProvenance(message_index=1, source_role=SourceRole.DEVELOPER),
        content="Prefer minimal diffs.",
    )
    assert dev_event.provenance.source_role == SourceRole.DEVELOPER
    assert dev_event.trust == TrustLevel.SYSTEM_TRUSTED

    assistant_event = NormalizedEvent(
        sequence_no=2,
        event_id="evt_2",
        kind=EventKind.ASSISTANT_MESSAGE,
        trust=TrustLevel.MODEL_GENERATED,
        provenance=EventProvenance(message_index=2, source_role=SourceRole.ASSISTANT),
        content="I will inspect the test suite.",
    )
    assert assistant_event.trust == TrustLevel.MODEL_GENERATED


def test_assistant_message_cannot_have_user_or_system_trust() -> None:
    with pytest.raises(InvariantViolation):
        NormalizedEvent(
            sequence_no=0,
            event_id="evt_bad",
            kind=EventKind.ASSISTANT_MESSAGE,
            trust=TrustLevel.USER,
            provenance=EventProvenance(message_index=0, source_role=SourceRole.ASSISTANT),
            content="Assistant message masquerading as user",
        )

    with pytest.raises(InvariantViolation):
        NormalizedEvent(
            sequence_no=0,
            event_id="evt_bad_sys",
            kind=EventKind.ASSISTANT_MESSAGE,
            trust=TrustLevel.SYSTEM_TRUSTED,
            provenance=EventProvenance(message_index=0, source_role=SourceRole.ASSISTANT),
            content="Assistant message masquerading as system",
        )


def test_tool_observation_enforces_untrusted_external() -> None:
    obs = ToolObservation(
        sequence_no=2,
        tool_call_id="call_1",
        tool_name="read_file",
        content="file contents",
        provenance=EventProvenance(
            message_index=2,
            source_role=SourceRole.TOOL,
            external_tool_call_id="call_1",
            tool_name="read_file",
        ),
    )
    assert obs.trust == TrustLevel.UNTRUSTED_EXTERNAL

    with pytest.raises(InvariantViolation):
        ToolObservation(
            sequence_no=2,
            tool_call_id="call_1",
            tool_name="read_file",
            content="attempting trust escalation",
            trust=TrustLevel.SYSTEM_TRUSTED,
            provenance=EventProvenance(message_index=2, source_role=SourceRole.TOOL),
        )


def test_candidate_action_disposition_invariants() -> None:
    internal_cand = CandidateAction(
        candidate_id="cand_expand",
        canonical_intent=CanonicalCapability.EXPAND_SEARCH,
        disposition=CandidateDisposition.INTERNAL_TRANSITION,
        risk_class=RiskClass.LOW,
        cost_class=CostClass.FREE,
    )
    assert internal_cand.disposition == CandidateDisposition.INTERNAL_TRANSITION

    # EXPAND_SEARCH cannot be declared as an EXTERNAL_TOOL
    with pytest.raises(InvariantViolation):
        CandidateAction(
            candidate_id="cand_bad_expand",
            canonical_intent=CanonicalCapability.EXPAND_SEARCH,
            disposition=CandidateDisposition.EXTERNAL_TOOL,
            external_tool_name="expand_search",
        )

    # INTERNAL_TRANSITION cannot bind an external_tool_name
    with pytest.raises(InvariantViolation):
        CandidateAction(
            candidate_id="cand_bad_internal",
            canonical_intent=CanonicalCapability.EXPAND_SEARCH,
            disposition=CandidateDisposition.INTERNAL_TRANSITION,
            external_tool_name="some_external_tool",
        )

    # EXTERNAL_TOOL requires external_tool_name
    with pytest.raises(InvariantViolation):
        CandidateAction(
            candidate_id="cand_missing_tool",
            canonical_intent=CanonicalCapability.READ_FILE,
            disposition=CandidateDisposition.EXTERNAL_TOOL,
            external_tool_name=None,
        )


def test_working_state_and_contracts_are_immutable() -> None:
    state = WorkingState(
        state_digest="a" * 64,
        event_count=1,
        last_sequence_no=0,
        initial_user_request="Fix test",
        latest_user_request="Fix test",
        user_messages=("Fix test",),
        trusted_system_instructions=("System rule",),
        available_tools=(
            ExternalToolBinding(
                external_name="custom_checker",
                description="Custom tool",
                parameters_schema={"type": "object"},
            ),
        ),
        mutation_verification=MutationVerificationState(),
    )
    assert state.available_tools[0].canonical_capability == CanonicalCapability.CUSTOM_TOOL
    with pytest.raises(ValidationError):
        state.event_count = 2  # type: ignore[misc]


def test_decision_and_generation_separate_semantics_from_telemetry() -> None:
    semantics = DecisionSemantics(
        selected_candidate_id="cand_answer",
        scores=(),
        provider_name="fake_controller",
        model_id="samatv256/mini-Jev",
    )
    d1 = DecisionResult(
        semantics=semantics,
        telemetry=ProviderCallTelemetry(latency_ms=1.2, request_attempt_id="req_1"),
    )
    d2 = DecisionResult(
        semantics=semantics,
        telemetry=ProviderCallTelemetry(latency_ms=48.9, request_attempt_id="req_2"),
    )
    assert d1.semantics == d2.semantics
    assert d1.telemetry != d2.telemetry

    gen_job = GenerationJob(
        job_id="job_1",
        job_type=GenerationJobType.ANSWER,
        candidate_id="cand_answer",
        initial_user_request="Hello",
        latest_user_request="Hello",
    )
    gen_sem = GenerationSemantics(
        job_id=gen_job.job_id,
        job_type=gen_job.job_type,
        content="Response text",
        provider_name="fake_generator",
        model_id="nvidia/nemotron",
    )
    g1 = GenerationResult(
        semantics=gen_sem,
        telemetry=ProviderCallTelemetry(latency_ms=5.0, request_attempt_id="req_1"),
    )
    g2 = GenerationResult(
        semantics=gen_sem,
        telemetry=ProviderCallTelemetry(latency_ms=25.0, request_attempt_id="req_2"),
    )
    assert g1.semantics == g2.semantics


def test_fake_providers_satisfy_protocols() -> None:
    assert isinstance(FakeRetriever(), Retriever)
    assert isinstance(FakeController(), Controller)
    assert isinstance(FakeGenerator(), Generator)


def test_settings_safe_dump_masks_secret_keys() -> None:
    ret_secret = "sk-" + "retriever-secret-key-123456"
    ctrl_secret = "sk-" + "controller-secret-key-123456"
    gen_secret = "nvapi-" + "generator-secret-key-123456"
    settings = Settings(
        RETRIEVER_API_KEY=SecretStr(ret_secret),
        CONTROLLER_API_KEY=SecretStr(ctrl_secret),
        GENERATOR_API_KEY=SecretStr(gen_secret),
    )
    dumped = settings.safe_dump()
    assert dumped["retriever_api_key"] == REDACTED_PLACEHOLDER
    assert dumped["controller_api_key"] == REDACTED_PLACEHOLDER
    assert dumped["generator_api_key"] == REDACTED_PLACEHOLDER
    assert "retriever-secret" not in str(dumped)
    assert "generator-secret" not in str(dumped)


def test_request_context_traceparent_validation() -> None:
    valid_tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    ctx = RequestContext(traceparent=valid_tp)
    assert ctx.traceparent == valid_tp

    invalid_ctx = RequestContext(traceparent="not-a-valid-w3c-traceparent")
    assert invalid_ctx.traceparent is None
