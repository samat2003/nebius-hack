"""Deterministic fake providers for Retriever, Controller, and Generator.

Operates completely offline with zero external API keys, network calls, or GPUs.
Supports deterministic fault injection (`TIMEOUT`, `UNAVAILABLE`, `MALFORMED_RESPONSE`).

Safety & protocol invariants:
- `FakeController` uses a conservative default selection policy: in `auto` mode
  it prefers `GENERATION_JOB` / `ASSISTANT_RESPONSE` (`cand_answer`) rather than
  automatically executing external commands, file edits, or custom tools merely
  because their JSON Schemas contain defaults.
- `FakeGenerator` emits deterministic synthetic output without echoing raw user
  prompts or tool outputs.
"""

from __future__ import annotations

from collections.abc import Sequence

from alienese.api.errors import InvalidProviderResponse
from alienese.contracts.candidates import CandidateAction, CandidateDisposition, RiskClass
from alienese.contracts.context import RequestContext
from alienese.contracts.decisions import (
    CandidateScore,
    DecisionResult,
    DecisionSemantics,
    GuardMetadata,
    ProviderCallTelemetry,
)
from alienese.contracts.generation import (
    GenerationJob,
    GenerationResult,
    GenerationSemantics,
)
from alienese.contracts.providers import (
    RankedItem,
    RetrievalRequest,
    RetrievalResult,
    RetrievalSemantics,
)
from alienese.contracts.state import WorkingState
from alienese.providers.base import ProviderFaultMode, raise_for_fault_mode


class FakeRetriever:
    """Deterministic offline implementation of the Retriever protocol."""

    def __init__(
        self,
        *,
        model_id: str = "google/embeddinggemma-2",
        model_revision: str = "fake-v1",
        fault_mode: ProviderFaultMode = ProviderFaultMode.NONE,
    ) -> None:
        self.provider_name = "fake_retriever"
        self.model_id = model_id
        self.model_revision = model_revision
        self.fault_mode = fault_mode

    async def rank(
        self,
        ctx: RequestContext,
        request: RetrievalRequest,
    ) -> RetrievalResult:
        """Deterministically rank candidate items preserving mandatory items first."""
        raise_for_fault_mode(self.provider_name, self.fault_mode)

        query_terms = set(request.query.lower().split())
        scored: list[tuple[bool, float, int, str]] = []

        for idx, item in enumerate(request.items):
            if item.mandatory:
                score = 1.0
            else:
                item_terms = set(item.content.lower().split())
                overlap = len(query_terms & item_terms) if query_terms else 0
                base = round(min(0.95, 0.5 + 0.1 * overlap), 4)
                score = round(max(0.1, base - (idx * 0.01)), 4)
            scored.append((item.mandatory, score, -idx, item.item_id))

        scored.sort(reverse=True)
        limited = scored[: request.max_results]
        ranked_items = tuple(
            RankedItem(item_id=item_id, score=score)
            for _mandatory, score, _neg_idx, item_id in limited
        )

        return RetrievalResult(
            semantics=RetrievalSemantics(
                ranked_items=ranked_items,
                provider_name=self.provider_name,
                model_id=self.model_id,
                model_revision=self.model_revision,
            ),
            telemetry=ProviderCallTelemetry(
                latency_ms=1.0,
                request_attempt_id=ctx.request_id,
            ),
        )


class FakeController:
    """Deterministic offline implementation of the Controller protocol."""

    def __init__(
        self,
        *,
        model_id: str = "samatv256/mini-Jev",
        model_revision: str = "fake-v1",
        fault_mode: ProviderFaultMode = ProviderFaultMode.NONE,
        forced_candidate_id: str | None = None,
    ) -> None:
        self.provider_name = "fake_controller"
        self.model_id = model_id
        self.model_revision = model_revision
        self.fault_mode = fault_mode
        self.forced_candidate_id = forced_candidate_id

    async def decide(
        self,
        ctx: RequestContext,
        state: WorkingState,
        candidates: Sequence[CandidateAction],
    ) -> DecisionResult:
        """Deterministically score and choose a candidate action."""
        if self.fault_mode == ProviderFaultMode.MALFORMED_RESPONSE:
            return DecisionResult(
                semantics=DecisionSemantics(
                    selected_candidate_id="__malformed_nonexistent_candidate__",
                    scores=(),
                    provider_name=self.provider_name,
                    model_id=self.model_id,
                    model_revision=self.model_revision,
                ),
                telemetry=ProviderCallTelemetry(
                    latency_ms=1.0,
                    request_attempt_id=ctx.request_id,
                ),
            )

        raise_for_fault_mode(self.provider_name, self.fault_mode)

        if not candidates:
            raise InvalidProviderResponse("Controller received an empty candidate set.")

        if self.forced_candidate_id is not None:
            selected_id = self.forced_candidate_id
        else:
            selected_id = self._select_default_candidate(state, candidates)

        scores: list[CandidateScore] = []
        for idx, cand in enumerate(candidates):
            score = (
                0.95 if cand.candidate_id == selected_id else round(max(0.05, 0.5 - idx * 0.1), 4)
            )
            scores.append(CandidateScore(candidate_id=cand.candidate_id, score=score))

        return DecisionResult(
            semantics=DecisionSemantics(
                selected_candidate_id=selected_id,
                scores=tuple(scores),
                provider_name=self.provider_name,
                model_id=self.model_id,
                model_revision=self.model_revision,
                guard_metadata=GuardMetadata(
                    verification_required=state.mutation_verification.mutation_attempted
                    and not state.mutation_verification.verification_confirmed
                ),
            ),
            telemetry=ProviderCallTelemetry(
                latency_ms=1.5,
                request_attempt_id=ctx.request_id,
            ),
        )

    @staticmethod
    def _select_default_candidate(
        state: WorkingState,
        candidates: Sequence[CandidateAction],
    ) -> str:
        """Select the safest executable candidate deterministically.

        Conservative fake-controller policy:
        1. Prefer generating a bounded assistant response (`GENERATION_JOB` or
           `ASSISTANT_RESPONSE`) whenever present in `candidates` (e.g., `auto` mode),
           never automatically triggering shell commands, file mutations, or
           custom tools merely because their JSON Schema contains valid defaults.
        2. When the caller explicitly constrained the turn to tool candidates
           (`tool_choice='required'` or `NamedToolChoice`), prefer low-risk
           executable tools (`RiskClass.LOW`) with complete arguments.
        """
        _ = state
        for cand in candidates:
            if cand.disposition in (
                CandidateDisposition.GENERATION_JOB,
                CandidateDisposition.ASSISTANT_RESPONSE,
            ):
                return cand.candidate_id

        for cand in candidates:
            if (
                cand.disposition == CandidateDisposition.EXTERNAL_TOOL
                and cand.arguments_complete
                and cand.risk_class == RiskClass.LOW
            ):
                return cand.candidate_id

        for cand in candidates:
            if cand.disposition == CandidateDisposition.EXTERNAL_TOOL and cand.arguments_complete:
                return cand.candidate_id

        return candidates[0].candidate_id


class FakeGenerator:
    """Deterministic, non-reflective offline implementation of the Generator protocol."""

    def __init__(
        self,
        *,
        model_id: str = "nvidia/nemotron",
        model_revision: str = "fake-v1",
        fault_mode: ProviderFaultMode = ProviderFaultMode.NONE,
        static_response: str | None = None,
    ) -> None:
        self.provider_name = "fake_generator"
        self.model_id = model_id
        self.model_revision = model_revision
        self.fault_mode = fault_mode
        self.static_response = static_response

    async def generate(
        self,
        ctx: RequestContext,
        job: GenerationJob,
    ) -> GenerationResult:
        """Deterministically synthesize a non-reflective response for a typed GenerationJob."""
        raise_for_fault_mode(self.provider_name, self.fault_mode)

        if self.static_response is not None:
            content = self.static_response
        else:
            content = f"[fake:{job.job_type.value.lower()}] Synthetic response for {job.job_id}"
            if job.selected_evidence:
                content += f" (with {len(job.selected_evidence)} evidence item(s))"

        prompt_source = (job.latest_user_request or job.initial_user_request or "").strip()
        prompt_tokens = max(1, len(prompt_source.split())) if prompt_source else 1
        completion_tokens = max(1, len(content.split()))

        return GenerationResult(
            semantics=GenerationSemantics(
                job_id=job.job_id,
                job_type=job.job_type,
                content=content,
                structured_arguments=None,
                provider_name=self.provider_name,
                model_id=self.model_id,
                model_revision=self.model_revision,
            ),
            telemetry=ProviderCallTelemetry(
                latency_ms=2.0,
                request_attempt_id=ctx.request_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )
