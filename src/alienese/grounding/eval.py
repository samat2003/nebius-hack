"""Offline evaluation harness and CLI metric reporter for candidate grounding."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alienese.api.errors import CompatibilityError
from alienese.api.models import ChatCompletionRequest, NamedToolChoice
from alienese.contracts.candidates import (
    CandidateAction,
    CandidateDisposition,
    CostClass,
    RiskClass,
)
from alienese.contracts.generation import GenerationJobType
from alienese.contracts.state import CanonicalCapability, ExternalToolBinding, WorkingState
from alienese.engine.normalize import normalize_request
from alienese.engine.reconstruct import reconstruct
from alienese.engine.turn import (
    _validate_schema_node,
    classify_tool_risk,
    extract_deterministic_tool_arguments,
    validate_tool_arguments_against_schema,
)
from alienese.grounding.candidate_builder import CandidateBuilder
from alienese.grounding.extractors import extract_all_evidence

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "tests"
    / "fixtures"
    / "grounding"
    / "decision_points.json"
)


def extract_schema_defaults(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Extract schema-defined defaults for properties."""
    defaults: dict[str, Any] = {}
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for prop_name, prop_def in properties.items():
            if isinstance(prop_def, Mapping) and "default" in prop_def:
                default_val = copy.deepcopy(prop_def["default"])
                ok, _ = _validate_schema_node(
                    prop_def,
                    default_val,
                    path=str(prop_name),
                    depth=1,
                )
                if ok:
                    defaults[str(prop_name)] = default_val
    return defaults


def _canonical_token(val: Any) -> str:
    """Deterministically serialize values (including dicts and lists) to safe string tokens."""
    if isinstance(val, (dict, list)):
        return json.dumps(val, sort_keys=True, separators=(",", ":"))
    return str(val)


def build_frozen_phase1_baseline(
    state: WorkingState,
    request: ChatCompletionRequest,
) -> tuple[CandidateAction, ...]:
    """Faithful frozen implementation of Phase 1 candidate construction.

    Uses schema defaults only, with no evidence grounding.
    """
    tool_choice = request.tool_choice
    candidates: list[CandidateAction] = []

    answer_candidate = CandidateAction(
        candidate_id="cand_answer",
        canonical_intent=CanonicalCapability.RESPOND,
        disposition=CandidateDisposition.GENERATION_JOB,
        requires_generation=True,
        generation_job_type=GenerationJobType.ANSWER,
        risk_class=RiskClass.LOW,
        cost_class=CostClass.LOW,
        rationale="Synthesize a direct assistant response to the latest user request.",
    )

    if tool_choice == "none" or not state.available_tools:
        return (answer_candidate,)

    if isinstance(tool_choice, NamedToolChoice):
        target_name = tool_choice.function.name
        matched = [t for t in state.available_tools if t.external_name == target_name]
        if not matched:
            raise CompatibilityError(
                f"Requested tool '{target_name}' is not present in available tools.",
                param="tool_choice",
            )
        binding = matched[0]
        args, is_complete = extract_deterministic_tool_arguments(binding)
        if not is_complete:
            raise CompatibilityError(
                f"Tool '{target_name}' requires arguments that cannot be deterministically "
                "grounded in Phase 1 without fabricating values.",
                param="tool_choice",
                code="ungrounded_required_tool_arguments",
            )
        tool_risk = classify_tool_risk(binding.canonical_capability)
        return (
            CandidateAction(
                candidate_id=f"cand_tool_{binding.external_name}",
                canonical_intent=binding.canonical_capability,
                disposition=CandidateDisposition.EXTERNAL_TOOL,
                external_tool_name=binding.external_name,
                arguments=args,
                arguments_complete=True,
                requires_generation=False,
                risk_class=tool_risk,
                cost_class=CostClass.LOW,
                rationale=f"Explicitly requested tool '{binding.external_name}'.",
            ),
        )

    for binding in state.available_tools:
        args, is_complete = extract_deterministic_tool_arguments(binding)
        tool_risk = classify_tool_risk(binding.canonical_capability)
        candidates.append(
            CandidateAction(
                candidate_id=f"cand_tool_{binding.external_name}",
                canonical_intent=binding.canonical_capability,
                disposition=CandidateDisposition.EXTERNAL_TOOL,
                external_tool_name=binding.external_name,
                arguments=args,
                arguments_complete=is_complete,
                requires_generation=False,
                risk_class=tool_risk,
                cost_class=CostClass.LOW,
                rationale=f"Candidate for external tool '{binding.external_name}'.",
            )
        )

    if tool_choice == "required":
        safe_executable_tools = [
            c for c in candidates if c.arguments_complete and c.risk_class == RiskClass.LOW
        ]
        if not safe_executable_tools:
            raise CompatibilityError(
                "tool_choice='required' was specified, but no low-risk tool has "
                "deterministically complete required arguments in Phase 1.",
                param="tool_choice",
                code="ungrounded_required_tool_arguments",
            )
        return tuple(safe_executable_tools)

    candidates.append(answer_candidate)
    return tuple(candidates)


def matches_oracle(candidate: CandidateAction, oracle: dict[str, Any]) -> bool:
    """Check if candidate is complete and matches oracle intent, tool, and exact arguments."""
    if not candidate.arguments_complete:
        return False

    expected_intent = oracle.get("canonical_intent")
    if expected_intent and candidate.canonical_intent.value != expected_intent:
        return False

    expected_tool = oracle.get("external_tool_name")
    if expected_tool is not None and candidate.external_tool_name != expected_tool:
        return False

    required_args: dict[str, Any] = oracle.get("required_arguments", {})
    for k, v in required_args.items():
        if k not in candidate.arguments:
            return False
        if candidate.arguments[k] != v:
            return False

    return True


def check_argument_presence(candidate: CandidateAction, oracle: dict[str, Any]) -> tuple[int, int]:
    """Count (present_args, total_required_args) for matching tool candidate."""
    required_args: dict[str, Any] = oracle.get("required_arguments", {})
    if not required_args:
        return 0, 0
    present = sum(1 for k in required_args if k in candidate.arguments)
    return present, len(required_args)


def evaluate_decision_points(
    decision_points: list[dict[str, Any]],
    use_phase1_baseline: bool = False,
    split: str | None = None,
) -> dict[str, Any]:
    """Run candidate evaluation across decision points with explicit metrics and denominators."""
    filtered_points = [dp for dp in decision_points if split is None or dp.get("split") == split]
    total_points = len(filtered_points)
    if total_points == 0:
        return {"total": 0}

    action_points = [dp for dp in filtered_points if dp["oracle"].get("expect_executable", True)]
    abstention_points = [
        dp for dp in filtered_points if not dp["oracle"].get("expect_executable", True)
    ]

    total_action = len(action_points)
    total_abstain = len(abstention_points)

    hits_at_1 = 0
    hits_at_4 = 0
    hits_at_8 = 0
    exact_arg_matches = 0

    total_req_args = 0
    present_req_args = 0

    correct_abstentions = 0

    total_executable_candidates = 0
    valid_executable_candidates = 0

    total_bound_arg_values = 0
    unsupported_arg_count = 0

    builder = CandidateBuilder()

    for dp in filtered_points:
        req_data = dp["request"]
        oracle = dp["oracle"]
        expect_exec = oracle.get("expect_executable", True)

        req = ChatCompletionRequest.model_validate(req_data)
        events, tools = normalize_request(req)
        state = reconstruct(events, available_tools=tools)

        tools_by_name: dict[str, ExternalToolBinding] = {t.external_name: t for t in tools}

        candidates: tuple[CandidateAction, ...] = ()
        failed_closed = False

        if use_phase1_baseline:
            try:
                candidates = build_frozen_phase1_baseline(state, req)
            except CompatibilityError:
                failed_closed = True
                candidates = ()
        else:
            try:
                candidates, _evidence, _diag = builder.build_candidates(
                    events=events,
                    state=state,
                    tool_choice=req.tool_choice,
                )
            except CompatibilityError:
                failed_closed = True
                candidates = ()

        # Collect evidence tokens for evidence-support proxy analysis
        all_evidence = extract_all_evidence(events, state, tools)
        evidence_tokens = {_canonical_token(e.value) for e in all_evidence}

        # Check executable candidates validity and evidence support
        for c in candidates:
            if c.arguments_complete and c.disposition == CandidateDisposition.EXTERNAL_TOOL:
                total_executable_candidates += 1
                binding = tools_by_name.get(c.external_tool_name or "")
                if binding:
                    valid, _ = validate_tool_arguments_against_schema(
                        binding.parameters_schema, c.arguments
                    )
                    if valid:
                        valid_executable_candidates += 1
                    tool_defaults = extract_schema_defaults(binding.parameters_schema)
                else:
                    tool_defaults = {}

                for arg_name, arg_val in c.arguments.items():
                    total_bound_arg_values += 1
                    val_token = _canonical_token(arg_val)
                    is_in_evidence = val_token in evidence_tokens
                    is_in_tool_defaults = (
                        arg_name in tool_defaults
                        and _canonical_token(tool_defaults[arg_name]) == val_token
                    )
                    if not (is_in_evidence or is_in_tool_defaults):
                        unsupported_arg_count += 1

        if not expect_exec:
            # Abstention case: expect no executable external tool
            has_no_executable = (
                failed_closed
                or not candidates
                or all(
                    not c.arguments_complete
                    for c in candidates
                    if c.disposition == CandidateDisposition.EXTERNAL_TOOL
                )
            )
            if has_no_executable:
                correct_abstentions += 1
            continue

        # Positive action case
        if not candidates:
            continue

        # Recall@1
        if len(candidates) >= 1 and matches_oracle(candidates[0], oracle):
            hits_at_1 += 1

        # Recall@4
        if any(matches_oracle(c, oracle) for c in candidates[:4]):
            hits_at_4 += 1

        # Recall@8
        if any(matches_oracle(c, oracle) for c in candidates[:8]):
            hits_at_8 += 1

        # Exact argument matches & argument presence
        matching_tools = [
            c for c in candidates if c.external_tool_name == oracle.get("external_tool_name")
        ]
        if matching_tools:
            top_matching = matching_tools[0]
            if matches_oracle(top_matching, oracle):
                exact_arg_matches += 1
            pres, req_cnt = check_argument_presence(top_matching, oracle)
            present_req_args += pres
            total_req_args += req_cnt
        else:
            req_args = oracle.get("required_arguments", {})
            total_req_args += len(req_args)

    return {
        "total": total_points,
        "total_action": total_action,
        "total_abstain": total_abstain,
        "hits_at_1": hits_at_1,
        "hits_at_4": hits_at_4,
        "hits_at_8": hits_at_8,
        "recall_at_1": round(hits_at_1 / total_action, 4) if total_action > 0 else 0.0,
        "recall_at_4": round(hits_at_4 / total_action, 4) if total_action > 0 else 0.0,
        "recall_at_8": round(hits_at_8 / total_action, 4) if total_action > 0 else 0.0,
        "exact_arg_matches": exact_arg_matches,
        "exact_arg_accuracy": round(exact_arg_matches / total_action, 4)
        if total_action > 0
        else 0.0,
        "present_req_args": present_req_args,
        "total_req_args": total_req_args,
        "arg_presence_rate": round(present_req_args / total_req_args, 4)
        if total_req_args > 0
        else 1.0,
        "correct_abstentions": correct_abstentions,
        "abstention_rate": round(correct_abstentions / total_abstain, 4)
        if total_abstain > 0
        else 1.0,
        "total_executable": total_executable_candidates,
        "valid_executable": valid_executable_candidates,
        "valid_executable_rate": None
        if total_executable_candidates == 0
        else round(valid_executable_candidates / total_executable_candidates, 4),
        "total_bound_args": total_bound_arg_values,
        "unsupported_arg_count": unsupported_arg_count,
        "evidence_support_rate": round(
            (total_bound_arg_values - unsupported_arg_count) / total_bound_arg_values, 4
        )
        if total_bound_arg_values > 0
        else 1.0,
        "fabrication_count": unsupported_arg_count,
        "fabrication_rate": 0.0
        if total_bound_arg_values == 0
        else round(unsupported_arg_count / total_bound_arg_values, 4),
    }


def format_rate(hits: int, total: int, rate: float | None) -> str:
    """Format metric with numerator, denominator, and percentage."""
    if rate is None or total == 0:
        return "N/A"
    return f"{hits}/{total} ({rate:.1%})"


def run_cli() -> None:
    """CLI entrypoint for running grounding offline evaluation."""
    parser = argparse.ArgumentParser(description="Alienese Grounding Offline Evaluation")
    parser.add_argument(
        "--fixture",
        type=str,
        default=str(DEFAULT_FIXTURE_PATH),
        help="Path to decision points JSON file",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        print(f"Fixture file not found: {fixture_path}")
        return

    with open(fixture_path, encoding="utf-8") as f:
        data = json.load(f)

    print("================================================================================")
    print(" Alienese Phase 3 — Offline Grounding Candidate Evaluation")
    print(f" Dataset: {fixture_path} ({len(data)} decision points)")
    print("================================================================================\n")

    for split in [None, "train", "dev", "test"]:
        split_name = "ALL (Combined)" if split is None else f"SPLIT: {split.upper()}"
        p1 = evaluate_decision_points(data, use_phase1_baseline=True, split=split)
        p3 = evaluate_decision_points(data, use_phase1_baseline=False, split=split)

        n_tot = p3["total"]
        n_act = p3["total_action"]
        n_abs = p3["total_abstain"]
        print(f"--- {split_name} (Total N={n_tot}, Action N={n_act}, Abstain N={n_abs}) ---")
        header = f"{'Metric':<28} {'Phase 1 Baseline':<22} {'Phase 3 Engine':<22} {'Lift':<10}"
        print(header)
        print("-" * 84)

        # Oracle Recalls
        p1_r1 = format_rate(p1["hits_at_1"], p1["total_action"], p1["recall_at_1"])
        p3_r1 = format_rate(p3["hits_at_1"], p3["total_action"], p3["recall_at_1"])
        r1_lift = f"{p3['recall_at_1'] - p1['recall_at_1']:+.1%}"
        print(f"{'Oracle Recall@1':<28} {p1_r1:<22} {p3_r1:<22} {r1_lift:<10}")

        p1_r4 = format_rate(p1["hits_at_4"], p1["total_action"], p1["recall_at_4"])
        p3_r4 = format_rate(p3["hits_at_4"], p3["total_action"], p3["recall_at_4"])
        r4_lift = f"{p3['recall_at_4'] - p1['recall_at_4']:+.1%}"
        print(f"{'Oracle Recall@4':<28} {p1_r4:<22} {p3_r4:<22} {r4_lift:<10}")

        p1_r8 = format_rate(p1["hits_at_8"], p1["total_action"], p1["recall_at_8"])
        p3_r8 = format_rate(p3["hits_at_8"], p3["total_action"], p3["recall_at_8"])
        r8_lift = f"{p3['recall_at_8'] - p1['recall_at_8']:+.1%}"
        print(f"{'Oracle Recall@8':<28} {p1_r8:<22} {p3_r8:<22} {r8_lift:<10}")

        # Exact Arg Accuracy
        p1_ea = format_rate(p1["exact_arg_matches"], p1["total_action"], p1["exact_arg_accuracy"])
        p3_ea = format_rate(p3["exact_arg_matches"], p3["total_action"], p3["exact_arg_accuracy"])
        ea_lift = f"{p3['exact_arg_accuracy'] - p1['exact_arg_accuracy']:+.1%}"
        print(f"{'Exact Arg Accuracy':<28} {p1_ea:<22} {p3_ea:<22} {ea_lift:<10}")

        # Arg Presence
        p1_ap = format_rate(p1["present_req_args"], p1["total_req_args"], p1["arg_presence_rate"])
        p3_ap = format_rate(p3["present_req_args"], p3["total_req_args"], p3["arg_presence_rate"])
        ap_lift = f"{p3['arg_presence_rate'] - p1['arg_presence_rate']:+.1%}"
        print(f"{'Arg Presence Rate':<28} {p1_ap:<22} {p3_ap:<22} {ap_lift:<10}")

        # Abstention Correctness
        p1_ab = format_rate(p1["correct_abstentions"], p1["total_abstain"], p1["abstention_rate"])
        p3_ab = format_rate(p3["correct_abstentions"], p3["total_abstain"], p3["abstention_rate"])
        ab_lift = f"{p3['abstention_rate'] - p1['abstention_rate']:+.1%}"
        print(f"{'Abstention Correctness':<28} {p1_ab:<22} {p3_ab:<22} {ab_lift:<10}")

        # Executable Validity
        p1_ev = format_rate(
            p1["valid_executable"], p1["total_executable"], p1["valid_executable_rate"]
        )
        p3_ev = format_rate(
            p3["valid_executable"], p3["total_executable"], p3["valid_executable_rate"]
        )
        print(f"{'Executable Validity':<28} {p1_ev:<22} {p3_ev:<22} {'0.0%':<10}")

        # Evidence Support (Proxy)
        p1_sup = format_rate(
            p1["total_bound_args"] - p1["unsupported_arg_count"],
            p1["total_bound_args"],
            p1["evidence_support_rate"],
        )
        p3_sup = format_rate(
            p3["total_bound_args"] - p3["unsupported_arg_count"],
            p3["total_bound_args"],
            p3["evidence_support_rate"],
        )
        sup_lift = f"{p3['evidence_support_rate'] - p1['evidence_support_rate']:+.1%}"
        print(f"{'Evidence Support (Proxy)':<28} {p1_sup:<22} {p3_sup:<22} {sup_lift:<10}")

        # Unsupported Arg Count
        p1_un = f"{p1['unsupported_arg_count']}/{p1['total_bound_args']}"
        p3_un = f"{p3['unsupported_arg_count']}/{p3['total_bound_args']}"
        print(f"{'Unsupported Arg Count':<28} {p1_un:<22} {p3_un:<22} {'0':<10}")
        print()


if __name__ == "__main__":
    run_cli()
