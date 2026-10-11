"""Deterministic candidate ranking and bounded set selection."""

from __future__ import annotations

from collections.abc import Sequence

from alienese.contracts.candidates import (
    CandidateAction,
    CostClass,
    RiskClass,
)
from alienese.contracts.events import TrustLevel
from alienese.contracts.state import CanonicalCapability, WorkingState
from alienese.grounding.evidence import EvidenceStatus, GroundingEvidence

DEFAULT_MAX_CANDIDATES = 8


def score_candidate(
    candidate: CandidateAction,
    state: WorkingState,
    evidence_items: Sequence[GroundingEvidence],
    user_request: str | None,
) -> float:
    """Compute a multi-factor deterministic ranking score for a candidate action."""
    score = 0.0

    # 1. Base completeness bonus: complete tool candidates are executable
    if candidate.canonical_intent == CanonicalCapability.RESPOND:
        score += 100.0
    elif candidate.arguments_complete:
        score += 200.0
    else:
        score += 20.0

    # 2. Trust tier bonus from supporting evidence provenance
    trust_weights = {
        TrustLevel.SYSTEM_TRUSTED: 100.0,
        TrustLevel.USER: 75.0,
        TrustLevel.MODEL_GENERATED: 50.0,
        TrustLevel.UNTRUSTED_EXTERNAL: 25.0,
    }
    if candidate.evidence_refs:
        ref_provenances = set(candidate.evidence_refs)
        max_trust = 0.0
        for evi in evidence_items:
            if evi.source_provenance in ref_provenances:
                t_score = trust_weights.get(evi.trust, 0.0)
                if t_score > max_trust:
                    max_trust = t_score
        score += max_trust

    # 3. User request alignment: path or tool explicitly in user prompt
    if user_request:
        user_lower = user_request.lower()
        if candidate.external_tool_name and candidate.external_tool_name.lower() in user_lower:
            score += 100.0
        for val in candidate.arguments.values():
            if isinstance(val, str) and len(val) > 3 and val.lower() in user_lower:
                score += 100.0

    # 4. Active failure / traceback alignment
    has_active_failure = any(e.status == EvidenceStatus.FAILED for e in evidence_items)
    if has_active_failure:
        for val in candidate.arguments.values():
            if isinstance(val, str):
                for e in evidence_items:
                    if e.status == EvidenceStatus.FAILED and e.value == val:
                        score += 80.0
                        break

    # 5. Mutation & Verification obligation
    mut_state = state.mutation_verification
    if mut_state.mutation_attempted and not mut_state.verification_confirmed:
        if candidate.canonical_intent == CanonicalCapability.RUN_TEST:
            score += 70.0
        elif candidate.canonical_intent == CanonicalCapability.FINISH:
            # Penalize finishing before verification
            score -= 100.0

    # 6. Redundancy & repetition penalty: avoid repeating exact recent actions
    for past_act in state.recent_actions:
        if (
            past_act.tool_name == candidate.external_tool_name
            and past_act.arguments == candidate.arguments
        ):
            score -= 60.0
            break

    # 7. Risk adjustments
    if candidate.risk_class == RiskClass.HIGH:
        score -= 40.0
    elif candidate.risk_class == RiskClass.MEDIUM:
        score -= 10.0

    # 8. Cost adjustments
    if candidate.cost_class == CostClass.HIGH:
        score -= 10.0

    return score


def rank_and_bound_candidates(
    candidates: Sequence[CandidateAction],
    state: WorkingState,
    evidence_items: Sequence[GroundingEvidence],
    user_request: str | None = None,
    max_k: int = DEFAULT_MAX_CANDIDATES,
    preserve_answer: bool = True,
) -> tuple[CandidateAction, ...]:
    """Deterministically score, rank, and bound candidates to top K."""
    if not candidates:
        return ()

    scored: list[tuple[float, str, CandidateAction]] = []
    for cand in candidates:
        score = score_candidate(cand, state, evidence_items, user_request)
        scored.append((score, cand.candidate_id, cand))

    # Sort descending by score, tiebreak ascending by candidate_id
    scored.sort(key=lambda item: (-item[0], item[1]))

    selected: list[CandidateAction] = [item[2] for item in scored[:max_k]]

    # Ensure assistant answer is preserved if requested and was in original pool
    if preserve_answer and not any(
        c.canonical_intent == CanonicalCapability.RESPOND for c in selected
    ):
        answer_candidates = [
            c for c in candidates if c.canonical_intent == CanonicalCapability.RESPOND
        ]
        if answer_candidates:
            if len(selected) >= max_k:
                selected.pop()  # Drop lowest ranked candidate
            selected.append(answer_candidates[0])

    return tuple(selected)
