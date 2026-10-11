"""Deterministic CandidateAction builder and grounding pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from alienese.api.models import NamedToolChoice
from alienese.contracts.candidates import (
    CandidateAction,
    CandidateDisposition,
    CostClass,
    RiskClass,
)
from alienese.contracts.events import NormalizedEvent
from alienese.contracts.generation import GenerationJobType
from alienese.contracts.state import (
    CanonicalCapability,
    WorkingState,
)
from alienese.engine.turn import classify_tool_risk
from alienese.grounding.argument_resolution import GroundedArgumentResolver
from alienese.grounding.evidence import EvidenceCategory, GroundingEvidence
from alienese.grounding.extractors import extract_all_evidence
from alienese.grounding.policy import enforce_tool_choice_policy
from alienese.grounding.ranking import DEFAULT_MAX_CANDIDATES, rank_and_bound_candidates


def deterministic_candidate_id(
    tool_name: str,
    canonical_intent: CanonicalCapability,
    arguments: dict[str, Any],
) -> str:
    """Generate a stable, deterministic candidate ID from tool name and arguments."""
    canonical_args = json.dumps(
        arguments,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    seed = f"{tool_name}:{canonical_intent.value}:{canonical_args}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"cand_{digest}"


class GroundingDiagnostics(BaseModel):
    """Immutable diagnostic report for grounding and candidate construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_evidence_count: int = 0
    evidence_counts_by_category: dict[str, int] = Field(default_factory=dict)
    raw_candidates_generated: int = 0
    executable_candidates_count: int = 0
    final_candidates_count: int = 0
    deduplicated_candidates_count: int = 0
    rejected_incomplete_count: int = 0
    abstention_reasons: tuple[str, ...] = ()


class CandidateBuilder:
    """Deterministic candidate construction engine."""

    def __init__(
        self,
        resolver: GroundedArgumentResolver | None = None,
        max_k: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        if max_k < 1:
            raise ValueError(f"max_k must be at least 1, got {max_k}")
        self._resolver = resolver or GroundedArgumentResolver()
        self._max_k = max_k

    def build_candidates(
        self,
        events: Sequence[NormalizedEvent],
        state: WorkingState,
        tool_choice: str | NamedToolChoice | None = None,
    ) -> tuple[tuple[CandidateAction, ...], tuple[GroundingEvidence, ...], GroundingDiagnostics]:
        """Construct, ground, rank, and bound candidates deterministically."""
        evidence_items = extract_all_evidence(events, state, state.available_tools)

        category_counts: dict[str, int] = {}
        for evi in evidence_items:
            category_counts[evi.category.value] = category_counts.get(evi.category.value, 0) + 1

        raw_candidates: list[CandidateAction] = []
        candidates_by_sig: dict[tuple[str, str], CandidateAction] = {}
        dedup_count = 0
        rejected_incomplete = 0
        abstention_reasons: list[str] = []

        # Always prepare standard assistant response candidate
        answer_cand = CandidateAction(
            candidate_id="cand_answer",
            canonical_intent=CanonicalCapability.RESPOND,
            disposition=CandidateDisposition.GENERATION_JOB,
            requires_generation=True,
            generation_job_type=GenerationJobType.ANSWER,
            risk_class=RiskClass.LOW,
            cost_class=CostClass.LOW,
            rationale="Synthesize a direct assistant response to the latest user request.",
        )

        for binding in state.available_tools:
            tool_risk = classify_tool_risk(binding.canonical_capability)

            # 1. Base resolution with schema defaults
            def_args, def_complete, def_evi = self._resolver.resolve_arguments(
                binding,
                evidence_items,
                state,
                primary_evidence=None,
            )
            # Use canonical cand_tool_{name} for schema-default base candidates
            base_cand_id = f"cand_tool_{binding.external_name}"
            action_sig = (
                binding.external_name,
                json.dumps(def_args, sort_keys=True, separators=(",", ":")),
            )
            base_action = CandidateAction(
                candidate_id=base_cand_id,
                canonical_intent=binding.canonical_capability,
                disposition=CandidateDisposition.EXTERNAL_TOOL,
                external_tool_name=binding.external_name,
                arguments=def_args,
                arguments_complete=def_complete,
                evidence_refs=tuple(e.source_provenance for e in def_evi),
                requires_generation=False,
                risk_class=tool_risk,
                cost_class=CostClass.LOW,
                rationale=f"Candidate for external tool '{binding.external_name}'.",
            )
            candidates_by_sig[action_sig] = base_action
            raw_candidates.append(base_action)

            # 2. Targeted resolution for each relevant evidence item
            for evi in evidence_items:
                is_relevant = False
                if (
                    (
                        binding.canonical_capability
                        in (
                            CanonicalCapability.READ_FILE,
                            CanonicalCapability.LIST_FILES,
                            CanonicalCapability.WRITE_FILE,
                            CanonicalCapability.APPLY_PATCH,
                        )
                        and evi.category == EvidenceCategory.FILE_PATH
                    )
                    or (
                        binding.canonical_capability == CanonicalCapability.RUN_TEST
                        and evi.category
                        in (
                            EvidenceCategory.TEST_TARGET,
                            EvidenceCategory.TEST_COMMAND,
                        )
                    )
                    or (
                        binding.canonical_capability == CanonicalCapability.SEARCH_TEXT
                        and evi.category
                        in (
                            EvidenceCategory.SEARCH_PATTERN,
                            EvidenceCategory.SYMBOL,
                        )
                    )
                    or (
                        binding.canonical_capability == CanonicalCapability.RUN_COMMAND
                        and evi.category == EvidenceCategory.TEST_COMMAND
                    )
                ):
                    is_relevant = True

                if not is_relevant:
                    continue

                args, is_complete, bound_evi = self._resolver.resolve_arguments(
                    binding,
                    evidence_items,
                    state,
                    primary_evidence=evi,
                )
                if not is_complete:
                    rejected_incomplete += 1

                action_sig = (
                    binding.external_name,
                    json.dumps(args, sort_keys=True, separators=(",", ":")),
                )
                if action_sig in candidates_by_sig:
                    # Deduplicate while preserving merged evidence provenance
                    existing = candidates_by_sig[action_sig]
                    existing_refs = set(existing.evidence_refs)
                    new_refs = {e.source_provenance for e in bound_evi}
                    combined_refs = tuple(sorted(existing_refs.union(new_refs), key=str))
                    # Update existing candidate with merged evidence refs
                    updated = existing.model_copy(update={"evidence_refs": combined_refs})
                    candidates_by_sig[action_sig] = updated
                    # Replace in raw_candidates list
                    for idx, c in enumerate(raw_candidates):
                        if c.candidate_id == existing.candidate_id:
                            raw_candidates[idx] = updated
                            break
                    dedup_count += 1
                    continue

                # Grounded candidate with bound evidence gets content-addressed ID
                cand_id = deterministic_candidate_id(
                    binding.external_name,
                    binding.canonical_capability,
                    args,
                )
                grounded_cand = CandidateAction(
                    candidate_id=cand_id,
                    canonical_intent=binding.canonical_capability,
                    disposition=CandidateDisposition.EXTERNAL_TOOL,
                    external_tool_name=binding.external_name,
                    arguments=args,
                    arguments_complete=is_complete,
                    evidence_refs=tuple(e.source_provenance for e in bound_evi),
                    requires_generation=False,
                    risk_class=tool_risk,
                    cost_class=CostClass.LOW,
                    rationale=(
                        f"Grounded candidate for tool '{binding.external_name}' "
                        f"bound to {evi.category.value} '{evi.value}'."
                    ),
                )
                candidates_by_sig[action_sig] = grounded_cand
                raw_candidates.append(grounded_cand)

        raw_candidates.append(answer_cand)
        raw_count = len(raw_candidates)

        # 3. Apply tool_choice eligibility FIRST before truncation
        try:
            eligible_candidates = enforce_tool_choice_policy(
                raw_candidates,
                tool_choice=tool_choice,
                available_tools=state.available_tools,
            )
        except Exception as exc:
            # Capture abstention reason for diagnostics and re-raise
            abstention_reasons.append(str(exc))
            raise

        # 4. Rank and bound within the eligible candidate set
        user_request = state.latest_user_request or state.initial_user_request
        preserve_answer = tool_choice in ("auto", None)
        final_candidates = rank_and_bound_candidates(
            eligible_candidates,
            state,
            evidence_items,
            user_request=user_request,
            max_k=self._max_k,
            preserve_answer=preserve_answer,
        )

        executable_count = sum(1 for c in final_candidates if c.arguments_complete)

        diagnostics = GroundingDiagnostics(
            total_evidence_count=len(evidence_items),
            evidence_counts_by_category=category_counts,
            raw_candidates_generated=raw_count,
            executable_candidates_count=executable_count,
            final_candidates_count=len(final_candidates),
            deduplicated_candidates_count=dedup_count,
            rejected_incomplete_count=rejected_incomplete,
            abstention_reasons=tuple(abstention_reasons),
        )

        return final_candidates, evidence_items, diagnostics
