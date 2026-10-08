"""`samatv256/mini-Jev` finite-choice Controller adapter boundary.

Targets the pinned checkpoint `samatv256/mini-Jev` at revision `step-010626`
(commit `3a1d1d19d85e9863146307fe4b769e8bbe242c4e`, `Qwen/Qwen3-0.6B` backbone
+ PEFT LoRA adapter + `_SetHead` `decision_head.safetensors`).

Enforces:
- Unique, non-empty candidate ID validation and bounded candidate-count policy
  (normal operational target 4-8 candidates; hard cap default 8).
- Deterministic single-candidate bypass without network invocation when only 1
  eligible candidate exists.
- Explicit preservation of `CandidateDisposition` (`external_tool`,
  `generation_job`, `assistant_response`, `internal_transition`) and separation
  between trusted system instructions and untrusted tool observations.
- Strict finite-choice response verification: exact candidate-ID set match
  (no duplicates, missing, or unknown IDs), finite probabilities in `[0.0, 1.0]`,
  probability simplex normalization within `1e-3`, and `selected_id` consistency
  with the argmax option.

Note: Option probabilities produced by `mini-Jev` are uncalibrated finite-choice
selection scores and must never be interpreted as calibrated domain confidence.
Hosting status in Phase 2: `BLOCKED_HOSTING` (verified offline via `httpx.MockTransport`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from alienese.api.errors import CompatibilityError, InvalidProviderResponse
from alienese.contracts.candidates import CandidateAction, CandidateDisposition
from alienese.contracts.context import RequestContext
from alienese.contracts.decisions import (
    CandidateScore,
    DecisionResult,
    DecisionSemantics,
    GuardMetadata,
    ProviderCallTelemetry,
)
from alienese.contracts.state import WorkingState
from alienese.providers.runtime.client import ProviderHttpClient

MINI_JEV_DEFAULT_MODEL = "samatv256/mini-Jev"
MINI_JEV_PINNED_REVISION = "step-010626"
MINI_JEV_PINNED_COMMIT_SHA = "3a1d1d19d85e9863146307fe4b769e8bbe242c4e"
MINI_JEV_DEFAULT_QUESTION = (
    "Given the current state and available options,\nwhich option should be selected?"
)
DEFAULT_MAX_CONTROLLER_CANDIDATES = 8
_PROB_SUM_TOLERANCE = 1e-3
_ARGMAX_TIE_TOLERANCE = 1e-6


def _disposition_to_option_type(disposition: CandidateDisposition) -> str:
    if disposition == CandidateDisposition.EXTERNAL_TOOL:
        return "external_tool"
    if disposition == CandidateDisposition.GENERATION_JOB:
        return "generation_job"
    if disposition == CandidateDisposition.ASSISTANT_RESPONSE:
        return "assistant_response"
    return "internal_transition"


def serialize_mini_jev_option(candidate: CandidateAction) -> dict[str, Any]:
    """Serialize a `CandidateAction` into `mini-Jev`'s `answer_options` schema.

    Preserves exact candidate disposition so internal transitions or generation
    jobs are never misrepresented as executable external tools.
    """
    option_type = _disposition_to_option_type(candidate.disposition)
    if candidate.disposition == CandidateDisposition.EXTERNAL_TOOL and candidate.external_tool_name:
        label = candidate.external_tool_name
    else:
        label = f"{option_type}:{candidate.canonical_intent.value.lower()}"

    option: dict[str, Any] = {
        "id": candidate.candidate_id,
        "type": option_type,
        "label": label,
        "description": candidate.rationale or f"Action {candidate.candidate_id}",
    }
    if candidate.disposition == CandidateDisposition.EXTERNAL_TOOL and candidate.arguments:
        option["schema"] = candidate.arguments
    return option


def serialize_mini_jev_state(state: WorkingState) -> dict[str, Any]:
    """Serialize `WorkingState` into `mini-Jev`'s structured state mapping.

    Preserves the trust boundary between `trusted_system_instructions` (`system`),
    user requests (`user_goal`), and untrusted external tool observations (`history`).
    """
    system_text = (
        "\n\n".join(state.trusted_system_instructions) if state.trusted_system_instructions else ""
    )
    user_goal = state.latest_user_request or state.initial_user_request or ""

    history: list[dict[str, str]] = []
    for msg in state.user_messages:
        history.append({"role": "user", "content": msg})
    if state.latest_tool_observation is not None:
        obs = state.latest_tool_observation
        history.append(
            {
                "role": "tool",
                "content": f"[untrusted_tool:{obs.tool_name}] {obs.content}",
            }
        )

    pending_count = len(state.pending_tool_call_ids)
    return {
        "system": system_text,
        "user_goal": user_goal,
        "summary": f"turn_events={state.event_count};pending_tool_calls={pending_count}",
        "environment": {
            "available_tool_count": len(state.available_tools),
            "mutation_attempted": state.mutation_verification.mutation_attempted,
            "verification_confirmed": state.mutation_verification.verification_confirmed,
        },
        "history": history,
    }


class MiniJevController:
    """Typed remote Controller adapter for `samatv256/mini-Jev` (`step-010626`)."""

    def __init__(
        self,
        *,
        http_client: ProviderHttpClient,
        model_id: str = MINI_JEV_DEFAULT_MODEL,
        expected_revision: str = MINI_JEV_PINNED_REVISION,
        max_candidates: int = DEFAULT_MAX_CONTROLLER_CANDIDATES,
    ) -> None:
        self._provider_name = "mini_jev"
        cleaned_model = model_id.strip()
        if not cleaned_model:
            raise CompatibilityError(
                "MiniJevController requires a non-empty model_id.",
                param="controller_model",
                code="invalid_controller_model",
            )
        cleaned_rev = expected_revision.strip()
        if not cleaned_rev:
            raise CompatibilityError(
                "MiniJevController requires a non-empty expected_revision.",
                param="expected_revision",
                code="invalid_controller_revision",
            )
        if max_candidates < 2 or max_candidates > 16:
            raise CompatibilityError(
                "MiniJevController max_candidates must be between 2 and 16 "
                "(recommended operational target is 4-8).",
                param="max_candidates",
                code="invalid_controller_max_candidates",
            )

        self._http = http_client
        self._model_id = cleaned_model
        self._expected_revision = cleaned_rev
        self._max_candidates = max_candidates

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def expected_revision(self) -> str:
        return self._expected_revision

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    def _validate_candidates(
        self,
        candidates: Sequence[CandidateAction],
    ) -> list[CandidateAction]:
        if not candidates:
            raise InvalidProviderResponse(
                "MiniJevController received an empty candidate set.",
                code="empty_candidate_set",
            )
        if len(candidates) > self._max_candidates:
            raise CompatibilityError(
                f"Candidate count ({len(candidates)}) exceeds MiniJevController "
                f"bounded candidate policy ({self._max_candidates}).",
                param="candidates",
                code="too_many_controller_candidates",
            )

        seen_ids: set[str] = set()
        validated: list[CandidateAction] = []
        for cand in candidates:
            cid = cand.candidate_id.strip()
            if not cid:
                raise CompatibilityError(
                    "CandidateAction.candidate_id must be non-empty.",
                    param="candidates",
                    code="empty_candidate_id",
                )
            if cid in seen_ids:
                raise CompatibilityError(
                    f"Duplicate CandidateAction.candidate_id '{cid}'.",
                    param="candidates",
                    code="duplicate_candidate_id",
                )
            seen_ids.add(cid)
            validated.append(cand)
        return validated

    def _parse_and_validate_decision(
        self,
        data: Mapping[str, Any],
        expected_candidate_ids: Sequence[str],
    ) -> tuple[str, tuple[CandidateScore, ...], str]:
        """Validate `mini-Jev` finite-choice response and return `(selected_id, scores, rev)`."""
        reported_model = data.get("model")
        if (
            isinstance(reported_model, str)
            and reported_model.strip()
            and reported_model.strip() != self._model_id
        ):
            raise InvalidProviderResponse(
                f"Provider '{self._provider_name}' returned model '{reported_model.strip()}', "
                f"expected '{self._model_id}'.",
                code="provider_model_mismatch",
            )

        reported_rev = data.get("model_revision") or data.get("revision")
        if isinstance(reported_rev, str) and reported_rev.strip():
            clean_rev = reported_rev.strip()
            if clean_rev not in {self._expected_revision, MINI_JEV_PINNED_COMMIT_SHA}:
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' returned unexpected revision "
                    f"'{clean_rev}'; expected '{self._expected_revision}'.",
                    code="provider_revision_mismatch",
                )
            revision = clean_rev
        else:
            revision = self._expected_revision

        raw_options = data.get("options")
        if not isinstance(raw_options, list) or len(raw_options) != len(expected_candidate_ids):
            actual = len(raw_options) if isinstance(raw_options, list) else "non-list"
            raise InvalidProviderResponse(
                f"Provider '{self._provider_name}' options count ({actual}) does not match "
                f"expected candidate count ({len(expected_candidate_ids)}).",
                code="invalid_controller_options_count",
            )

        expected_id_set = set(expected_candidate_ids)
        probs_by_id: dict[str, float] = {}

        for opt in raw_options:
            if not isinstance(opt, Mapping):
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' option entry must be a JSON object.",
                    code="invalid_controller_option",
                )
            opt_id = opt.get("id")
            if not isinstance(opt_id, str) or not opt_id.strip():
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' option entry is missing string 'id'.",
                    code="invalid_controller_option_id",
                )
            if opt_id not in expected_id_set:
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' returned unknown candidate id '{opt_id}'.",
                    code="unknown_candidate_id",
                )
            if opt_id in probs_by_id:
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' returned duplicate candidate id '{opt_id}'.",
                    code="duplicate_candidate_id",
                )

            raw_prob = opt.get("probability")
            if isinstance(raw_prob, bool) or not isinstance(raw_prob, (int, float)):
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' option '{opt_id}' probability "
                    "is not numeric.",
                    code="invalid_candidate_probability",
                )
            prob = float(raw_prob)
            if not math.isfinite(prob) or prob < 0.0 or prob > 1.0:
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' option '{opt_id}' probability {prob} "
                    "must be finite and within [0.0, 1.0].",
                    code="invalid_candidate_probability",
                )
            probs_by_id[opt_id] = prob

        if set(probs_by_id.keys()) != expected_id_set:
            raise InvalidProviderResponse(
                f"Provider '{self._provider_name}' options did not cover all expected candidates.",
                code="missing_candidate_id",
            )

        prob_sum = sum(probs_by_id.values())
        if not math.isfinite(prob_sum) or abs(prob_sum - 1.0) > _PROB_SUM_TOLERANCE:
            raise InvalidProviderResponse(
                f"Provider '{self._provider_name}' option probabilities sum to {prob_sum:.6f}, "
                f"violating simplex normalization tolerance ({_PROB_SUM_TOLERANCE}).",
                code="unnormalized_candidate_probabilities",
            )

        selected_id = data.get("selected_id")
        if not isinstance(selected_id, str) or selected_id not in expected_id_set:
            raise InvalidProviderResponse(
                f"Provider '{self._provider_name}' returned unknown or missing selected_id "
                f"'{selected_id}'.",
                code="invalid_selected_candidate_id",
            )

        max_prob = max(probs_by_id.values())
        if (max_prob - probs_by_id[selected_id]) > _ARGMAX_TIE_TOLERANCE:
            raise InvalidProviderResponse(
                f"Provider '{self._provider_name}' selected_id '{selected_id}' "
                f"(p={probs_by_id[selected_id]:.6f}) is inconsistent with argmax "
                f"probability ({max_prob:.6f}).",
                code="inconsistent_selected_candidate",
            )

        scores = tuple(
            CandidateScore(candidate_id=cid, score=round(probs_by_id[cid], 6))
            for cid in expected_candidate_ids
        )
        return selected_id, scores, revision

    async def decide(
        self,
        ctx: RequestContext,
        state: WorkingState,
        candidates: Sequence[CandidateAction],
    ) -> DecisionResult:
        """Score and select one `CandidateAction` from the finite candidate set."""
        validated = self._validate_candidates(candidates)
        verification_needed = (
            state.mutation_verification.mutation_attempted
            and not state.mutation_verification.verification_confirmed
        )

        # Single eligible candidate deterministic bypass (mini-Jev requires >= 2 options)
        if len(validated) == 1:
            only_cand = validated[0]
            return DecisionResult(
                semantics=DecisionSemantics(
                    selected_candidate_id=only_cand.candidate_id,
                    scores=(CandidateScore(candidate_id=only_cand.candidate_id, score=1.0),),
                    provider_name=self._provider_name,
                    model_id=self._model_id,
                    model_revision=self._expected_revision,
                    guard_metadata=GuardMetadata(
                        bypassed_controller=True,
                        fallback_used=False,
                        fallback_reason="single_eligible_candidate",
                        verification_required=verification_needed,
                    ),
                ),
                telemetry=ProviderCallTelemetry(
                    latency_ms=0.0,
                    request_attempt_id=ctx.request_id,
                    attempt_count=1,
                    failed_attempt_count=0,
                ),
            )

        expected_ids = [c.candidate_id for c in validated]
        payload: dict[str, Any] = {
            "model": self._model_id,
            "revision": self._expected_revision,
            "state": serialize_mini_jev_state(state),
            "question": MINI_JEV_DEFAULT_QUESTION,
            "question_type": "choice",
            "answer_options": [serialize_mini_jev_option(c) for c in validated],
        }

        http_resp = await self._http.post_json(ctx, "/predict", payload)
        selected_id, scores, revision = self._parse_and_validate_decision(
            http_resp.data,
            expected_ids,
        )

        return DecisionResult(
            semantics=DecisionSemantics(
                selected_candidate_id=selected_id,
                scores=scores,
                provider_name=self._provider_name,
                model_id=self._model_id,
                model_revision=revision,
                guard_metadata=GuardMetadata(
                    bypassed_controller=False,
                    fallback_used=False,
                    fallback_reason=None,
                    verification_required=verification_needed,
                ),
            ),
            telemetry=http_resp.telemetry,
        )
