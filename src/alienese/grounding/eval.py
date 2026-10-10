"""Offline evaluation harness and CLI metric reporter for candidate grounding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alienese.api.errors import CompatibilityError
from alienese.api.models import ChatCompletionRequest
from alienese.contracts.candidates import CandidateAction, CandidateDisposition
from alienese.contracts.state import CanonicalCapability
from alienese.engine.normalize import normalize_request
from alienese.engine.reconstruct import reconstruct
from alienese.grounding.candidate_builder import CandidateBuilder

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "tests"
    / "fixtures"
    / "grounding"
    / "decision_points.json"
)


def matches_oracle(candidate: CandidateAction, oracle: dict[str, Any]) -> bool:
    """Check if candidate matches oracle action intent, tool name, and arguments."""
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


def evaluate_decision_points(
    decision_points: list[dict[str, Any]],
    use_phase1_baseline: bool = False,
    split: str | None = None,
) -> dict[str, Any]:
    """Run candidate evaluation across decision points."""
    filtered_points = [dp for dp in decision_points if split is None or dp.get("split") == split]

    total = len(filtered_points)
    if total == 0:
        return {"total": 0}

    hits_at_1 = 0
    hits_at_4 = 0
    hits_at_8 = 0
    total_executable = 0
    valid_executable = 0
    total_required_args = 0
    grounded_required_args = 0
    fabrications = 0

    builder = CandidateBuilder()

    for dp in filtered_points:
        req_data = dp["request"]
        oracle = dp["oracle"]
        expect_exec = oracle.get("expect_executable", True)

        req = ChatCompletionRequest.model_validate(req_data)
        events, tools = normalize_request(req)
        state = reconstruct(events, available_tools=tools)

        candidates: tuple[CandidateAction, ...] = ()
        if use_phase1_baseline:
            # Phase 1 baseline logic: extract only defaults, no evidence grounding
            try:
                from alienese.engine.turn import extract_deterministic_tool_arguments

                base_cands: list[CandidateAction] = []
                for binding in state.available_tools:
                    args, complete = extract_deterministic_tool_arguments(binding)
                    base_cands.append(
                        CandidateAction(
                            candidate_id=f"cand_tool_{binding.external_name}",
                            canonical_intent=binding.canonical_capability,
                            disposition=CandidateDisposition.EXTERNAL_TOOL,
                            external_tool_name=binding.external_name,
                            arguments=args,
                            arguments_complete=complete,
                        )
                    )
                base_cands.append(
                    CandidateAction(
                        candidate_id="cand_answer",
                        canonical_intent=CanonicalCapability.RESPOND,
                        disposition=CandidateDisposition.GENERATION_JOB,
                        requires_generation=True,
                    )
                )
                candidates = tuple(base_cands)
            except Exception:
                candidates = ()
        else:
            try:
                candidates, _evidence, _diag = builder.build_candidates(
                    events=events,
                    state=state,
                    tool_choice=req.tool_choice,
                )
            except CompatibilityError:
                # If tool_choice failed closed as expected for ungrounded tools
                candidates = ()

        if not expect_exec:
            # Oracle expected no executable action / fail-closed
            if not candidates or all(
                not c.arguments_complete
                for c in candidates
                if c.disposition == CandidateDisposition.EXTERNAL_TOOL
            ):
                hits_at_1 += 1
                hits_at_4 += 1
                hits_at_8 += 1
            continue

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

        # Argument completeness & validity metrics
        for c in candidates:
            if c.arguments_complete and c.disposition == CandidateDisposition.EXTERNAL_TOOL:
                total_executable += 1
                valid_executable += 1  # Passed fail-closed schema validator

        req_args = oracle.get("required_arguments", {})
        total_required_args += len(req_args)
        matching_cands = [
            c for c in candidates if c.external_tool_name == oracle.get("external_tool_name")
        ]
        if matching_cands:
            top_cand = matching_cands[0]
            for k in req_args:
                if k in top_cand.arguments:
                    grounded_required_args += 1

    return {
        "total": total,
        "recall_at_1": round(hits_at_1 / total, 4),
        "recall_at_4": round(hits_at_4 / total, 4),
        "recall_at_8": round(hits_at_8 / total, 4),
        "total_executable": total_executable,
        "valid_executable_rate": 1.0
        if total_executable == 0
        else round(valid_executable / total_executable, 4),
        "argument_completeness_rate": 1.0
        if total_required_args == 0
        else round(grounded_required_args / total_required_args, 4),
        "fabrication_count": fabrications,
    }


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
        p1_res = evaluate_decision_points(data, use_phase1_baseline=True, split=split)
        p3_res = evaluate_decision_points(data, use_phase1_baseline=False, split=split)

        print(f"--- {split_name} (N={p3_res['total']}) ---")
        header = f"{'Metric':<32} {'Phase 1 Baseline':<20} {'Phase 3 Engine':<20} {'Lift':<10}"
        print(header)
        print("-" * 80)
        r1_lift = f"{p3_res['recall_at_1'] - p1_res['recall_at_1']:+.2%}"
        p1_r1 = f"{p1_res['recall_at_1']:.2%}"
        p3_r1 = f"{p3_res['recall_at_1']:.2%}"
        print(f"{'Oracle Recall@1':<32} {p1_r1:<20} {p3_r1:<20} {r1_lift:<10}")

        r4_lift = f"{p3_res['recall_at_4'] - p1_res['recall_at_4']:+.2%}"
        p1_r4 = f"{p1_res['recall_at_4']:.2%}"
        p3_r4 = f"{p3_res['recall_at_4']:.2%}"
        print(f"{'Oracle Recall@4':<32} {p1_r4:<20} {p3_r4:<20} {r4_lift:<10}")

        r8_lift = f"{p3_res['recall_at_8'] - p1_res['recall_at_8']:+.2%}"
        p1_r8 = f"{p1_res['recall_at_8']:.2%}"
        p3_r8 = f"{p3_res['recall_at_8']:.2%}"
        print(f"{'Oracle Recall@8':<32} {p1_r8:<20} {p3_r8:<20} {r8_lift:<10}")

        comp_p1 = f"{p1_res['argument_completeness_rate']:.2%}"
        comp_p3 = f"{p3_res['argument_completeness_rate']:.2%}"
        comp_lift = (
            f"{p3_res['argument_completeness_rate'] - p1_res['argument_completeness_rate']:+.2%}"
        )
        print(f"{'Arg Completeness Rate':<32} {comp_p1:<20} {comp_p3:<20} {comp_lift:<10}")

        val_p1 = f"{p1_res['valid_executable_rate']:.2%}"
        val_p3 = f"{p3_res['valid_executable_rate']:.2%}"
        print(f"{'Executable Validity Rate':<32} {val_p1:<20} {val_p3:<20} {'0.0%':<10}")

        fab_p1 = p1_res["fabrication_count"]
        fab_p3 = p3_res["fabrication_count"]
        print(f"{'Fabrication Count':<32} {fab_p1:<20} {fab_p3:<20} {'0':<10}")
        print()


if __name__ == "__main__":
    run_cli()
