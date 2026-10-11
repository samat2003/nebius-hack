"""Deterministic rank-argmax controller for reliable candidate selection."""

from __future__ import annotations

from collections.abc import Sequence

from alienese.contracts.candidates import CandidateAction
from alienese.contracts.context import RequestContext
from alienese.contracts.decisions import (
    CandidateScore,
    DecisionResult,
    DecisionSemantics,
    GuardMetadata,
    ProviderCallTelemetry,
)
from alienese.contracts.state import WorkingState
from alienese.providers.fake import FakeController


class GroundedRankController:
    """Controller that selects candidate_0 (the deterministically top-ranked grounded candidate).

    Falls back safely to FakeController behavior if candidate set is empty or non-standard.
    """

    def __init__(self, *, model_id: str = "alienese/grounded-ranker-v1") -> None:
        self.provider_name = "grounded_rank_controller"
        self.model_id = model_id
        self.model_revision = "v1"

    async def decide(
        self,
        ctx: RequestContext,
        state: WorkingState,
        candidates: Sequence[CandidateAction],
    ) -> DecisionResult:
        """Select top-ranked candidate deterministically."""
        if not candidates:
            fake = FakeController()
            return await fake.decide(ctx, state, candidates)

        # candidate_0 is the argmax of deterministic multi-factor ranking
        selected = candidates[0]
        scores: list[CandidateScore] = []
        for idx, cand in enumerate(candidates):
            score = (
                1.0
                if cand.candidate_id == selected.candidate_id
                else round(max(0.01, 0.8 - idx * 0.1), 4)
            )
            scores.append(CandidateScore(candidate_id=cand.candidate_id, score=score))

        verification_needed = (
            state.mutation_verification.mutation_attempted
            and not state.mutation_verification.verification_confirmed
        )

        return DecisionResult(
            semantics=DecisionSemantics(
                selected_candidate_id=selected.candidate_id,
                scores=tuple(scores),
                provider_name=self.provider_name,
                model_id=self.model_id,
                model_revision=self.model_revision,
                guard_metadata=GuardMetadata(
                    bypassed_controller=False,
                    verification_required=verification_needed,
                ),
            ),
            telemetry=ProviderCallTelemetry(
                latency_ms=0.5,
                request_attempt_id=ctx.request_id,
                attempt_count=1,
            ),
        )
