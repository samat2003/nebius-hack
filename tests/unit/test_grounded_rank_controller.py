"""Unit tests for GroundedRankController."""

from __future__ import annotations

import pytest

from alienese.api.errors import InvalidProviderResponse
from alienese.contracts.candidates import (
    CandidateAction,
    CandidateDisposition,
    CostClass,
    RiskClass,
)
from alienese.contracts.context import RequestContext
from alienese.contracts.generation import GenerationJobType
from alienese.contracts.state import CanonicalCapability, WorkingState
from alienese.grounding.controller import GroundedRankController


@pytest.mark.asyncio
async def test_grounded_rank_controller_selects_candidate_zero() -> None:
    controller = GroundedRankController()
    ctx = RequestContext(request_id="test-rank-ctx", operation_id="op-rank-1")
    state = WorkingState(
        state_digest="0" * 64,
        event_count=1,
        initial_user_request="test request",
        latest_user_request="test request",
    )

    c0 = CandidateAction(
        candidate_id="cand_top_grounded",
        canonical_intent=CanonicalCapability.READ_FILE,
        disposition=CandidateDisposition.EXTERNAL_TOOL,
        external_tool_name="read_file",
        arguments={"path": "src/app.py"},
        arguments_complete=True,
        risk_class=RiskClass.LOW,
        cost_class=CostClass.LOW,
        rationale="Top grounded candidate",
    )
    c1 = CandidateAction(
        candidate_id="cand_answer",
        canonical_intent=CanonicalCapability.RESPOND,
        disposition=CandidateDisposition.GENERATION_JOB,
        requires_generation=True,
        generation_job_type=GenerationJobType.ANSWER,
        risk_class=RiskClass.LOW,
        cost_class=CostClass.LOW,
        rationale="Fallback answer",
    )

    result = await controller.decide(ctx, state, [c0, c1])
    assert result.semantics.selected_candidate_id == "cand_top_grounded"
    assert result.semantics.provider_name == "grounded_rank_controller"
    assert len(result.semantics.scores) == 2
    assert result.semantics.scores[0].score == 1.0


@pytest.mark.asyncio
async def test_grounded_rank_controller_empty_candidates_handled() -> None:
    controller = GroundedRankController()
    ctx = RequestContext(request_id="test-rank-ctx-2", operation_id="op-rank-2")
    state = WorkingState(
        state_digest="1" * 64,
        event_count=1,
        initial_user_request="test request",
        latest_user_request="test request",
    )

    with pytest.raises(InvalidProviderResponse):
        await controller.decide(ctx, state, [])
