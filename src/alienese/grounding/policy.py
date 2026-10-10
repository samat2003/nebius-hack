"""Tool-choice enforcement, candidate filtering, and guard policy."""

from __future__ import annotations

from collections.abc import Sequence

from alienese.api.errors import CompatibilityError
from alienese.api.models import NamedToolChoice
from alienese.contracts.candidates import (
    CandidateAction,
    CandidateDisposition,
    RiskClass,
)
from alienese.contracts.state import ExternalToolBinding


def enforce_tool_choice_policy(
    candidates: Sequence[CandidateAction],
    tool_choice: str | NamedToolChoice | None,
    available_tools: Sequence[ExternalToolBinding],
) -> tuple[CandidateAction, ...]:
    """Enforce caller tool_choice constraints on the candidate action set."""
    if tool_choice == "none":
        filtered = [
            c
            for c in candidates
            if c.disposition
            in (CandidateDisposition.GENERATION_JOB, CandidateDisposition.ASSISTANT_RESPONSE)
        ]
        return tuple(filtered)

    if isinstance(tool_choice, NamedToolChoice):
        target_name = tool_choice.function.name
        matched_tools = [t for t in available_tools if t.external_name == target_name]
        if not matched_tools:
            raise CompatibilityError(
                f"Requested tool '{target_name}' is not present in available tools.",
                param="tool_choice",
            )
        target_candidates = [
            c
            for c in candidates
            if c.disposition == CandidateDisposition.EXTERNAL_TOOL
            and c.external_tool_name == target_name
        ]
        complete_target = [c for c in target_candidates if c.arguments_complete]
        if not complete_target:
            raise CompatibilityError(
                f"Tool '{target_name}' requires arguments that cannot be deterministically "
                "grounded from observed evidence.",
                param="tool_choice",
                code="ungrounded_required_tool_arguments",
            )
        return tuple(complete_target)

    if tool_choice == "required":
        executable_tools = [
            c
            for c in candidates
            if c.disposition == CandidateDisposition.EXTERNAL_TOOL
            and c.arguments_complete
            and c.risk_class == RiskClass.LOW
        ]
        if not executable_tools:
            raise CompatibilityError(
                "tool_choice='required' was specified, but no low-risk external tool "
                "has complete grounded arguments.",
                param="tool_choice",
                code="ungrounded_required_tool_arguments",
            )
        return tuple(executable_tools)

    # In 'auto' mode or None: return all valid candidates
    return tuple(candidates)
