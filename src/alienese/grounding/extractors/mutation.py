"""Deterministic mutation and verification state extractor."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from alienese.contracts.events import EventKind, NormalizedEvent
from alienese.contracts.state import (
    CanonicalCapability,
    ExternalToolBinding,
    WorkingState,
)
from alienese.grounding.evidence import (
    EvidenceCategory,
    EvidenceStatus,
    GroundingEvidence,
    deterministic_evidence_id,
)
from alienese.grounding.extractors.base import MAX_EVENTS_SCANNED
from alienese.grounding.normalization import normalize_file_path

_MUTATION_CAPABILITIES = frozenset(
    {
        CanonicalCapability.APPLY_PATCH,
        CanonicalCapability.WRITE_FILE,
    }
)
_VERIFICATION_CAPABILITIES = frozenset(
    {
        CanonicalCapability.RUN_TEST,
    }
)

_PYTEST_PASSED_RE = re.compile(
    r"(?:^|\n)\s*={2,}\s+(?:(\d+)\s+passed(?:,\s+\d+\s+warnings?)?)\s+in\s+[\d\.]+s\s+={2,}",
    re.MULTILINE,
)
_UNITTEST_PASSED_RE = re.compile(r"(?:^|\n)\s*OK\s*(?:\([^\n\r]*\))?\s*$", re.MULTILINE)
_PYTEST_FAILED_RE = re.compile(
    r"(?:^|\n)\s*={2,}\s+.*(?:failed|error|FAILURES).*={2,}|FAILED\s+tests/|AssertionError\b",
    re.MULTILINE | re.IGNORECASE,
)
_ZERO_TESTS_RE = re.compile(r"collected\s+0\s+items|no\s+tests\s+ran", re.IGNORECASE)
_INTERRUPTED_RE = re.compile(
    r"KeyboardInterrupt|interrupted:|timed out|cancelled|terminated", re.IGNORECASE
)


def classify_verification_outcome(
    content: str,
    exit_code: int | None = None,
) -> tuple[str, EvidenceStatus]:
    """Classify verification output into PASSED, FAILED, or UNKNOWN.

    Only explicit recognized passed summaries yield PASSED and EvidenceStatus.CONFIRMED.
    """
    if not content or not content.strip():
        return "UNKNOWN", EvidenceStatus.INFERRED

    cleaned = content.strip()

    # 1. Non-zero exit code indicates explicit failure
    if exit_code is not None and exit_code != 0:
        return "FAILED", EvidenceStatus.FAILED

    if re.search(r"\bexit\s+(?:status|code)\s+([1-9]\d*)\b", cleaned, re.IGNORECASE):
        return "FAILED", EvidenceStatus.FAILED

    # 2. Interrupted or zero tests collected -> UNKNOWN (cannot infer success)
    if _INTERRUPTED_RE.search(cleaned):
        return "UNKNOWN", EvidenceStatus.INFERRED
    if _ZERO_TESTS_RE.search(cleaned):
        return "UNKNOWN", EvidenceStatus.INFERRED

    # 3. Explicit failed test indicators -> FAILED
    if _PYTEST_FAILED_RE.search(cleaned):
        return "FAILED", EvidenceStatus.FAILED
    if re.search(r"\bFAILED\s+\(failures=\d+|\bFAILED\s+\(errors=\d+", cleaned):
        return "FAILED", EvidenceStatus.FAILED

    # 4. Explicit passed test indicators -> PASSED (only if no failures detected)
    match_pytest = _PYTEST_PASSED_RE.search(cleaned)
    if match_pytest:
        num_passed = int(match_pytest.group(1))
        if num_passed > 0:
            return "PASSED", EvidenceStatus.CONFIRMED

    if _UNITTEST_PASSED_RE.search(cleaned):
        return "PASSED", EvidenceStatus.CONFIRMED

    # 5. Default to UNKNOWN for ambiguous, partial, or unrecognized output
    return "UNKNOWN", EvidenceStatus.INFERRED


def classify_mutation_outcome(
    content: str,
    target_file: str,
) -> tuple[bool, EvidenceStatus]:
    """Validate whether tool output positively confirms mutation of target_file.

    Returns:
        (is_confirmed, status)
        - (True, EvidenceStatus.CONFIRMED) if structured success or explicit positive text
        - (False, EvidenceStatus.FAILED) if explicit failure reported
        - (False, EvidenceStatus.OBSERVED) if empty, unrelated, wrong file, or unconfirmed
    """
    if not content or not content.strip():
        return False, EvidenceStatus.OBSERVED

    cleaned = content.strip()

    # 1. Check for explicit error / failure indicators
    lower = cleaned.lower()
    if any(
        err in lower
        for err in (
            "permission denied",
            "failed to write",
            "failed to apply",
            "patch failed",
            "error:",
            "conflict",
            "cannot open file",
            "no such file or directory",
        )
    ):
        return False, EvidenceStatus.FAILED

    # 2. Check structured JSON output
    if (cleaned.startswith("{") and cleaned.endswith("}")) or (
        cleaned.startswith("[") and cleaned.endswith("]")
    ):
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                # If explicit failure field
                if data.get("success") is False or data.get("status") in ("error", "failed"):
                    return False, EvidenceStatus.FAILED

                # Verify target path match if path is provided in JSON
                json_path = (
                    data.get("path")
                    or data.get("filepath")
                    or data.get("file_path")
                    or data.get("target_file")
                    or data.get("file")
                )
                if json_path and isinstance(json_path, str):
                    norm_json_path = normalize_file_path(json_path)
                    if norm_json_path and norm_json_path != target_file:
                        # Output confirms a different file, not target_file!
                        return False, EvidenceStatus.OBSERVED

                # Check for positive status or success flag
                if data.get("success") is True or data.get("status") in (
                    "ok",
                    "success",
                    "applied",
                    "written",
                    "saved",
                    "completed",
                ):
                    return True, EvidenceStatus.CONFIRMED
        except (ValueError, TypeError):
            pass

    # 3. Check for recognized positive text patterns matching target_file
    target_pattern = re.escape(target_file)
    # Target file must be mentioned in the output
    if not re.search(rf"\b{target_pattern}\b", cleaned):
        return False, EvidenceStatus.OBSERVED

    # Positive completion text patterns
    positive_patterns = (
        rf"\b(?:wrote|written|saved|created|updated|overwrote)\b.*?\b{target_pattern}\b",
        rf"\b{target_pattern}\b.*?\b(?:written|saved|created|updated|overwritten)\b",
        rf"\bapplied patch\b.*?\b{target_pattern}\b",
        rf"\bpatch applied\b.*?\b{target_pattern}\b",
        rf"\bsuccessfully\s+(?:wrote|updated|saved|created|applied|modified)\b.*?\b{target_pattern}\b",
        rf"\b{target_pattern}\b.*?\bsuccessfully\b",
        rf"Wrote\s+\d+\s+bytes\s+to\s+\b{target_pattern}\b",
        rf"File\s+['\"]?{target_pattern}['\"]?\s+(?:updated|created|written|saved)",
    )
    for pat in positive_patterns:
        if re.search(pat, cleaned, re.IGNORECASE):
            return True, EvidenceStatus.CONFIRMED

    # Unrecognized or ambiguous text
    return False, EvidenceStatus.OBSERVED


class MutationExtractor:
    """Extracts mutation targets and explicit verification results across event history."""

    def extract(
        self,
        events: Sequence[NormalizedEvent],
        state: WorkingState,
        tools: Sequence[ExternalToolBinding],
    ) -> Sequence[GroundingEvidence]:
        results: list[GroundingEvidence] = []
        tools_by_name = {t.external_name: t for t in tools}
        scanned_events = events[-MAX_EVENTS_SCANNED:]

        # Map tool_call_id to tool call events
        call_events_by_id: dict[str, NormalizedEvent] = {}
        for ev in scanned_events:
            if ev.kind == EventKind.TOOL_CALL and ev.tool_call_id:
                call_events_by_id[ev.tool_call_id] = ev

        for ev in scanned_events:
            # 1. Check tool calls for mutation attempts
            if ev.kind == EventKind.TOOL_CALL and ev.tool_name:
                binding = tools_by_name.get(ev.tool_name)
                capability = (
                    binding.canonical_capability if binding else CanonicalCapability.CUSTOM_TOOL
                )
                if capability in _MUTATION_CAPABILITIES and ev.tool_arguments:
                    target = (
                        ev.tool_arguments.get("path")
                        or ev.tool_arguments.get("filepath")
                        or ev.tool_arguments.get("file_path")
                        or ev.tool_arguments.get("target_file")
                    )
                    if isinstance(target, str):
                        norm_target = normalize_file_path(target)
                        if norm_target:
                            results.append(
                                GroundingEvidence(
                                    evidence_id=deterministic_evidence_id(
                                        EvidenceCategory.MUTATION_TARGET,
                                        norm_target,
                                        ev.provenance,
                                        ev.sequence_no,
                                        sub_id="mut_attempt",
                                    ),
                                    category=EvidenceCategory.MUTATION_TARGET,
                                    value=norm_target,
                                    source_provenance=ev.provenance,
                                    trust=ev.trust,
                                    sequence_no=ev.sequence_no,
                                    is_direct=True,
                                    status=EvidenceStatus.OBSERVED,
                                    tool_call_id=ev.tool_call_id,
                                )
                            )

            # 2. Check tool results for mutation confirmations and verification outcomes
            if ev.kind == EventKind.TOOL_RESULT and ev.tool_call_id:
                call_ev = call_events_by_id.get(ev.tool_call_id)
                tool_name = call_ev.tool_name if call_ev else ev.tool_name
                binding = tools_by_name.get(tool_name) if tool_name else None
                capability = (
                    binding.canonical_capability if binding else CanonicalCapability.CUSTOM_TOOL
                )

                # Check for mutation confirmation: successful write/patch execution
                if capability in _MUTATION_CAPABILITIES and call_ev and call_ev.tool_arguments:
                    target = (
                        call_ev.tool_arguments.get("path")
                        or call_ev.tool_arguments.get("filepath")
                        or call_ev.tool_arguments.get("file_path")
                        or call_ev.tool_arguments.get("target_file")
                    )
                    if isinstance(target, str):
                        norm_target = normalize_file_path(target)
                        if norm_target:
                            _is_confirmed, status = classify_mutation_outcome(
                                ev.content, norm_target
                            )
                            sub_id = (
                                "mut_confirmed"
                                if status == EvidenceStatus.CONFIRMED
                                else (
                                    "mut_failed"
                                    if status == EvidenceStatus.FAILED
                                    else "mut_unconfirmed"
                                )
                            )
                            results.append(
                                GroundingEvidence(
                                    evidence_id=deterministic_evidence_id(
                                        EvidenceCategory.MUTATION_TARGET,
                                        norm_target,
                                        ev.provenance,
                                        ev.sequence_no,
                                        sub_id=sub_id,
                                    ),
                                    category=EvidenceCategory.MUTATION_TARGET,
                                    value=norm_target,
                                    source_provenance=ev.provenance,
                                    trust=ev.trust,
                                    sequence_no=ev.sequence_no,
                                    is_direct=True,
                                    status=status,
                                    tool_call_id=ev.tool_call_id,
                                    context_snippet=ev.content[:256],
                                )
                            )

                # Check if this tool execution was a verification attempt
                is_verification = capability in _VERIFICATION_CAPABILITIES or (
                    tool_name
                    and any(
                        tool_name.lower().startswith(p)
                        for p in ("pytest", "run_test", "test_runner")
                    )
                )
                if is_verification:
                    val, status = classify_verification_outcome(ev.content)
                    results.append(
                        GroundingEvidence(
                            evidence_id=deterministic_evidence_id(
                                EvidenceCategory.VERIFICATION_RESULT,
                                val,
                                ev.provenance,
                                ev.sequence_no,
                                sub_id="verif_result",
                            ),
                            category=EvidenceCategory.VERIFICATION_RESULT,
                            value=val,
                            source_provenance=ev.provenance,
                            trust=ev.trust,
                            sequence_no=ev.sequence_no,
                            is_direct=True,
                            status=status,
                            tool_call_id=ev.tool_call_id,
                            context_snippet=ev.content[:256],
                        )
                    )

        return results
