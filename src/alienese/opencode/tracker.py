"""Alienese OpenCode Session Tracker.

Tracks evidence, enforces workspace containment and safety policies, derives
WorkingState with SHA-256 state digests, and produces redacted session traces
during interactive and headless OpenCode sessions.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alienese.contracts.events import (
    EventKind,
    EventProvenance,
    NormalizedEvent,
    SourceRole,
    TrustLevel,
)
from alienese.contracts.state import (
    CanonicalCapability,
    ExternalToolBinding,
    WorkingState,
)
from alienese.engine.reconstruct import reconstruct
from alienese.grounding.extractors.failures import FailureExtractor
from alienese.grounding.extractors.mutation import MutationExtractor
from alienese.grounding.extractors.paths import PathExtractor
from alienese.grounding.extractors.tests import TestExtractor
from alienese.grounding.normalization import normalize_file_path
from alienese.observability.redaction import redact_value

_TOOL_CAPABILITY_MAP: dict[str, CanonicalCapability] = {
    "read": CanonicalCapability.READ_FILE,
    "read_file": CanonicalCapability.READ_FILE,
    "view_file": CanonicalCapability.READ_FILE,
    "write": CanonicalCapability.WRITE_FILE,
    "write_file": CanonicalCapability.WRITE_FILE,
    "write_to_file": CanonicalCapability.WRITE_FILE,
    "edit": CanonicalCapability.APPLY_PATCH,
    "edit_file": CanonicalCapability.APPLY_PATCH,
    "replace_file_content": CanonicalCapability.APPLY_PATCH,
    "bash": CanonicalCapability.RUN_COMMAND,
    "run_command": CanonicalCapability.RUN_COMMAND,
    "command": CanonicalCapability.RUN_COMMAND,
}

_PROTECTED_BASENAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        "id_rsa",
        "id_ed25519",
    }
)


class OpenCodeSessionTracker:
    """Session tracker managing state derivation, safety policies, and evidence tracking."""

    def __init__(
        self,
        workspace: Path,
        session_id: str = "default",
        trace_file: Path | None = None,
        read_only_test_files: bool = False,
    ) -> None:
        self.workspace = workspace.resolve()
        self.session_id = session_id
        self.trace_file = (
            trace_file.resolve()
            if trace_file is not None
            else (self.workspace / ".alienese" / "opencode_trace.jsonl")
        )
        self.read_only_test_files = read_only_test_files

        self.sequence_no = 0
        self.events: list[NormalizedEvent] = []
        self.mutations: list[str] = []
        self.verification_obligation = False
        self.last_test_passed: bool | None = None
        self.state_digest = "0" * 64
        self.evidence_log: list[dict[str, Any]] = []

        self._failure_extractor = FailureExtractor()
        self._test_extractor = TestExtractor()
        self._path_extractor = PathExtractor()
        self._mutation_extractor = MutationExtractor()

        # Load persisted session state if present
        self._load_session_state()

    def _state_file(self) -> Path:
        state_dir = self.workspace / ".alienese" / "sessions"
        state_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize session_id for filesystem
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", self.session_id)
        return state_dir / f"{safe_id}.json"

    def _load_session_state(self) -> None:
        path = self._state_file()
        if not path.is_file():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.sequence_no = int(data.get("sequence_no", 0))
            self.mutations = list(data.get("mutations", []))
            self.verification_obligation = bool(data.get("verification_obligation", False))
            self.last_test_passed = data.get("last_test_passed")
            self.state_digest = str(data.get("state_digest", "0" * 64))
            self.evidence_log = list(data.get("evidence_log", []))

            # Reconstruct raw events
            raw_events = data.get("events", [])
            for rev in raw_events:
                self.events.append(
                    NormalizedEvent(
                        sequence_no=rev["sequence_no"],
                        event_id=rev["event_id"],
                        kind=EventKind(rev["kind"]),
                        trust=TrustLevel(rev["trust"]),
                        provenance=EventProvenance(**rev["provenance"]),
                        content=rev["content"],
                        tool_name=rev.get("tool_name"),
                        tool_call_id=rev.get("tool_call_id"),
                        tool_arguments=rev.get("tool_arguments"),
                    )
                )
        except Exception:
            # Corrupted session file; restart clean
            pass

    def _save_session_state(self) -> None:
        path = self._state_file()
        try:
            serialized_events = [
                {
                    "sequence_no": ev.sequence_no,
                    "event_id": ev.event_id,
                    "kind": ev.kind.value,
                    "trust": ev.trust.value,
                    "content": ev.content[:2048],
                    "provenance": ev.provenance.model_dump(mode="json"),
                    "tool_name": ev.tool_name,
                    "tool_call_id": ev.tool_call_id,
                    "tool_arguments": ev.tool_arguments,
                }
                for ev in self.events[-50:]  # keep recent 50
            ]
            data = {
                "session_id": self.session_id,
                "sequence_no": self.sequence_no,
                "mutations": self.mutations,
                "verification_obligation": self.verification_obligation,
                "last_test_passed": self.last_test_passed,
                "state_digest": self.state_digest,
                "evidence_log": self.evidence_log[-50:],
                "events": serialized_events,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def evaluate_pre_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        call_id: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate pre-tool safety guards and return an enforcement decision.

        Enforces:
        - Out-of-workspace containment rejection
        - Protected credential / sensitive file access rejection
        - Directory traversal suppression
        - Read-only protection for test files when enabled
        """
        self.sequence_no += 1
        cid = call_id or f"call_{len(self.events)}"
        raw_target = (
            args.get("filePath") or args.get("path") or args.get("file") or args.get("target")
        )

        decision: dict[str, Any] = {
            "allowed": True,
            "decision": "ALLOW",
            "enforced_guard": False,
            "guard_name": "none",
            "reason": f"Tool '{tool_name}' permitted by policy.",
            "grounded_path": None,
            "verification_obligation": self.verification_obligation,
            "call_id": cid,
        }

        # 1. File operation validation
        if raw_target and isinstance(raw_target, str):
            target_str = str(raw_target).strip()
            target_path = Path(target_str)

            # Check absolute vs relative resolution
            try:
                if target_path.is_absolute():
                    resolved = target_path.resolve()
                else:
                    resolved = (self.workspace / target_path).resolve()
            except Exception as exc:
                decision.update(
                    {
                        "allowed": False,
                        "decision": "REJECT",
                        "enforced_guard": True,
                        "guard_name": "workspace_containment",
                        "reason": f"Invalid path resolution: {exc}",
                    }
                )
                self._record_trace("PRE_TOOL", tool_name, args, decision)
                return decision

            # Check workspace containment
            try:
                is_contained = resolved.is_relative_to(self.workspace)
            except AttributeError:
                # Python <3.9 fallback
                try:
                    resolved.relative_to(self.workspace)
                    is_contained = True
                except ValueError:
                    is_contained = False

            if not is_contained:
                decision.update(
                    {
                        "allowed": False,
                        "decision": "REJECT",
                        "enforced_guard": True,
                        "guard_name": "workspace_containment",
                        "reason": (
                            f"Path '{target_str}' is outside repository workspace "
                            f"boundary ({self.workspace})."
                        ),
                    }
                )
                self._record_trace("PRE_TOOL", tool_name, args, decision)
                return decision

            # Compute repo-relative path and run Alienese normalization
            try:
                rel_path = resolved.relative_to(self.workspace)
            except ValueError:
                decision.update(
                    {
                        "allowed": False,
                        "decision": "REJECT",
                        "enforced_guard": True,
                        "guard_name": "workspace_containment",
                        "reason": f"Path '{target_str}' escapes workspace root.",
                    }
                )
                self._record_trace("PRE_TOOL", tool_name, args, decision)
                return decision

            rel_str = str(rel_path).replace("\\", "/")
            normalized = normalize_file_path(rel_str)
            if normalized is None and rel_str != ".":
                decision.update(
                    {
                        "allowed": False,
                        "decision": "REJECT",
                        "enforced_guard": True,
                        "guard_name": "path_normalization",
                        "reason": (
                            f"Path '{target_str}' contains invalid characters "
                            "or traversal segments."
                        ),
                    }
                )
                self._record_trace("PRE_TOOL", tool_name, args, decision)
                return decision

            decision["grounded_path"] = normalized

            # Check protected files (.env, .git)
            parts = rel_path.parts
            if any(p == ".git" for p in parts) or rel_path.name in _PROTECTED_BASENAMES:
                decision.update(
                    {
                        "allowed": False,
                        "decision": "REJECT",
                        "enforced_guard": True,
                        "guard_name": "protected_file",
                        "reason": (
                            f"Access to sensitive/protected target '{rel_str}' is strictly blocked."
                        ),
                    }
                )
                self._record_trace("PRE_TOOL", tool_name, args, decision)
                return decision

            # Check read-only test modification if configured
            is_mutation_tool = tool_name in {"write", "edit", "write_file", "edit_file"}
            is_test_file = any(p == "tests" for p in parts) or rel_path.name.startswith("test_")
            if self.read_only_test_files and is_mutation_tool and is_test_file:
                decision.update(
                    {
                        "allowed": False,
                        "decision": "REJECT",
                        "enforced_guard": True,
                        "guard_name": "read_only_test_policy",
                        "reason": (f"Modifying test '{rel_str}' disallowed by policy."),
                    }
                )
                self._record_trace("PRE_TOOL", tool_name, args, decision)
                return decision

        # 2. Bash command inspection
        elif tool_name in {"bash", "run_command", "command"}:
            cmd = str(args.get("command", "")).strip()
            # Explicit inspection: detect obvious traversal attempts
            if "/etc/passwd" in cmd or "/etc/shadow" in cmd:
                decision.update(
                    {
                        "allowed": False,
                        "decision": "REJECT",
                        "enforced_guard": True,
                        "guard_name": "sensitive_system_path",
                        "reason": "Execution targeting sensitive system files (/etc) is blocked.",
                    }
                )
                self._record_trace("PRE_TOOL", tool_name, args, decision)
                return decision

            # Unsupported general shell execution is described as observed / deferred
            decision.update(
                {
                    "decision": "OBSERVE",
                    "enforced_guard": False,
                    "guard_name": "shell_observation",
                    "reason": (
                        "Shell execution deferred to OpenCode; evidence observed by Alienese."
                    ),
                }
            )

        if decision["allowed"]:
            seq = len(self.events)
            tool_call_event = NormalizedEvent(
                sequence_no=seq,
                event_id=f"ev_tool_call_{seq}",
                kind=EventKind.TOOL_CALL,
                trust=TrustLevel.MODEL_GENERATED,
                provenance=EventProvenance(
                    message_index=seq,
                    sub_index=0,
                    source_role=SourceRole.ASSISTANT,
                    external_tool_call_id=cid,
                    tool_name=tool_name,
                ),
                content="",
                tool_name=tool_name,
                tool_call_id=cid,
                tool_arguments=redact_value(args),
            )
            self.events.append(tool_call_event)
            self._save_session_state()

        self._record_trace("PRE_TOOL", tool_name, args, decision)
        return decision

    def process_post_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        output_data: Any,
        call_id: str | None = None,
    ) -> dict[str, Any]:
        """Process observed tool execution outcome, extract evidence, update obligations."""
        self.sequence_no += 1
        output_str = ""
        if isinstance(output_data, str):
            output_str = output_data
        elif isinstance(output_data, dict):
            output_str = str(output_data.get("output", output_data.get("title", "")))
        else:
            output_str = str(output_data)

        cid = call_id or f"call_{len(self.events)}"
        # Ensure preceding TOOL_CALL exists for protocol invariant
        has_tool_call = any(
            ev.kind == EventKind.TOOL_CALL and ev.tool_call_id == cid for ev in self.events
        )
        if not has_tool_call:
            seq_call = len(self.events)
            tool_call_event = NormalizedEvent(
                sequence_no=seq_call,
                event_id=f"ev_tool_call_{seq_call}",
                kind=EventKind.TOOL_CALL,
                trust=TrustLevel.MODEL_GENERATED,
                provenance=EventProvenance(
                    message_index=seq_call,
                    sub_index=0,
                    source_role=SourceRole.ASSISTANT,
                    external_tool_call_id=cid,
                    tool_name=tool_name,
                ),
                content="",
                tool_name=tool_name,
                tool_call_id=cid,
                tool_arguments=redact_value(args),
            )
            self.events.append(tool_call_event)

        # 1. Create NormalizedEvent for tool output
        seq = len(self.events)
        ev_id = f"ev_tool_post_{seq}"
        capability = _TOOL_CAPABILITY_MAP.get(tool_name, CanonicalCapability.RUN_COMMAND)
        provenance = EventProvenance(
            message_index=seq,
            sub_index=0,
            source_role=SourceRole.TOOL,
            external_tool_call_id=cid,
            tool_name=tool_name,
        )
        norm_event = NormalizedEvent(
            sequence_no=seq,
            event_id=ev_id,
            kind=EventKind.TOOL_RESULT,
            trust=TrustLevel.UNTRUSTED_EXTERNAL,
            provenance=provenance,
            content=output_str[:4096],
            tool_name=tool_name,
            tool_call_id=cid,
            tool_arguments=redact_value(args),
        )
        self.events.append(norm_event)

        # 2. Reconstruct WorkingState & compute state digest
        dummy_tool = ExternalToolBinding(
            external_name=tool_name,
            description=f"OpenCode tool {tool_name}",
            parameters_schema={},
            canonical_capability=capability,
        )
        working_state = reconstruct(self.events, available_tools=[dummy_tool])
        self.state_digest = working_state.state_digest

        # 3. Extract evidence using Alienese extractors
        new_evidence: list[dict[str, Any]] = []

        failures = self._failure_extractor.extract(self.events, working_state, [dummy_tool])
        for f in failures:
            new_evidence.append(
                {
                    "category": f.category.value,
                    "value": f.value,
                    "provenance": f.source_provenance.model_dump(mode="json"),
                }
            )

        tests = self._test_extractor.extract(self.events, working_state, [dummy_tool])
        for t in tests:
            new_evidence.append(
                {
                    "category": t.category.value,
                    "value": t.value,
                    "provenance": t.source_provenance.model_dump(mode="json"),
                }
            )

        paths = self._path_extractor.extract(self.events, working_state, [dummy_tool])
        for p in paths:
            new_evidence.append(
                {
                    "category": p.category.value,
                    "value": p.value,
                    "provenance": p.source_provenance.model_dump(mode="json"),
                }
            )

        self.evidence_log.extend(new_evidence)

        # 4. Track mutations and verification obligations
        raw_target = (
            args.get("filePath") or args.get("path") or args.get("file") or args.get("target")
        )
        if tool_name in {"write", "edit", "write_file", "edit_file"}:
            if raw_target and str(raw_target) not in self.mutations:
                self.mutations.append(str(raw_target))
            self.verification_obligation = True

        # Check if tests were executed in bash
        if tool_name in {"bash", "run_command", "command"}:
            cmd = str(args.get("command", "")).lower()
            if "pytest" in cmd or "test" in cmd:
                # Determine pass/fail
                has_failed = (
                    "failed" in output_str.lower()
                    or "error" in output_str.lower()
                    or "assertionerror" in output_str.lower()
                )
                has_passed = "passed" in output_str.lower() and not has_failed
                if has_passed:
                    self.last_test_passed = True
                    # If all tests pass and a mutation was made, obligation is satisfied
                    self.verification_obligation = False
                elif has_failed:
                    self.last_test_passed = False
                    self.verification_obligation = True

        # 5. Derive three-model guidance if local model server is available
        three_model_guidance = self._evaluate_three_model_guidance(working_state, output_str)

        # 6. Persist state and append trace record
        self._save_session_state()

        result_info = {
            "session_id": self.session_id,
            "step": self.sequence_no,
            "state_digest": self.state_digest,
            "mutations": list(self.mutations),
            "verification_obligation": self.verification_obligation,
            "last_test_passed": self.last_test_passed,
            "evidence_count": len(self.evidence_log),
            "new_evidence_count": len(new_evidence),
            "three_model_guidance": three_model_guidance,
        }

        self._record_trace(
            "POST_TOOL",
            tool_name,
            args,
            result_info,
            output_summary=output_str[:500],
        )

        return result_info

    def _record_trace(
        self,
        phase: str,
        tool_name: str,
        args: dict[str, Any],
        decision_or_result: dict[str, Any],
        output_summary: str | None = None,
    ) -> None:
        """Write an immutable, secret-redacted trace entry to the session log."""
        try:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            entry: dict[str, Any] = {
                "timestamp": datetime.now(UTC).isoformat(),
                "session_id": self.session_id,
                "sequence_no": self.sequence_no,
                "phase": phase,
                "tool": tool_name,
                "arguments": redact_value(args),
                "state_digest": self.state_digest,
                "verification_obligation": self.verification_obligation,
                "mutations": list(self.mutations),
                "details": decision_or_result,
            }
            if output_summary is not None:
                entry["output_summary"] = redact_value(output_summary)

            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def get_status(self) -> dict[str, Any]:
        """Return the current session status and verification obligations."""
        return {
            "session_id": self.session_id,
            "sequence_no": self.sequence_no,
            "state_digest": self.state_digest,
            "mutations": list(self.mutations),
            "verification_obligation": self.verification_obligation,
            "last_test_passed": self.last_test_passed,
            "evidence_count": len(self.evidence_log),
            "trace_file": str(self.trace_file),
        }

    def _evaluate_three_model_guidance(
        self,
        working_state: WorkingState,
        last_output: str,
    ) -> dict[str, Any] | None:
        """Query local EmbeddingGemma 2 and mini-Jev to derive semantic guidance."""
        import urllib.request

        # Fast health check with 300ms timeout
        try:
            req = urllib.request.Request("http://127.0.0.1:8001/health")
            with urllib.request.urlopen(req, timeout=0.3) as resp:
                if resp.status != 200:
                    return None
        except Exception:
            return None

        # 1. Discover repository files for EmbeddingGemma ranking
        candidate_files: list[tuple[str, str]] = []
        for p in self.workspace.rglob("*.py"):
            parts = p.parts
            if ".alienese" in parts or ".venv" in parts or "__pycache__" in parts:
                continue
            try:
                rel_str = str(p.relative_to(self.workspace)).replace("\\", "/")
                snippet = p.read_text(encoding="utf-8")[:400]
                candidate_files.append((rel_str, snippet))
            except Exception:
                continue

        if not candidate_files:
            return None

        query_text = "fix python bug"
        for ev in reversed(self.evidence_log):
            if ev.get("category") == "failed_test" or "error" in str(ev.get("value", "")).lower():
                query_text = str(ev.get("value", ""))[:200]
                break
        if query_text == "fix python bug" and last_output:
            query_text = last_output[:200]

        # 2. Query EmbeddingGemma 2
        formatted_inputs = [
            f"task: code retrieval | query: {query_text}",
            *[f"title: {f[0]} | text: {f[1]}" for f in candidate_files[:8]],
        ]
        emb_payload = json.dumps(
            {
                "model": "google/embeddinggemma-2",
                "input": formatted_inputs,
                "dimensions": 768,
            }
        ).encode("utf-8")

        ranked_files: list[tuple[str, float]] = []
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8001/embeddings",
                data=emb_payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                emb_data = json.loads(resp.read().decode("utf-8"))
                vectors = [item["embedding"] for item in emb_data["data"]]
                q_vec = vectors[0]
                for idx, (fpath, _) in enumerate(candidate_files[:8]):
                    d_vec = vectors[idx + 1]
                    dot = sum(a * b for a, b in zip(q_vec, d_vec, strict=True))
                    ranked_files.append((fpath, round(dot, 4)))
            ranked_files.sort(key=lambda x: -x[1])
        except Exception:
            return None

        # 3. Query mini-Jev Controller for next action choice
        top_file = ranked_files[0][0] if ranked_files else "auth/session_manager.py"
        test_file = next((f[0] for f in candidate_files if "test" in f[0]), "tests/test_session.py")

        answer_options = [
            {
                "id": "act_read",
                "label": f"read_file: {top_file}",
                "type": "external_tool",
                "description": f"Inspect source code of {top_file}",
            },
            {
                "id": "act_test",
                "label": f"run_command: pytest {test_file}",
                "type": "external_tool",
                "description": f"Run verification tests in {test_file}",
            },
        ]

        pred_payload = json.dumps(
            {
                "model": "samatv256/mini-Jev",
                "revision": "step-010626",
                "state": {
                    "user_goal": query_text,
                    "summary": (
                        f"mutations={len(self.mutations)};"
                        f"verification_needed={self.verification_obligation}"
                    ),
                    "environment": {
                        "available_tool_count": 2,
                        "mutation_attempted": bool(self.mutations),
                        "verification_confirmed": not self.verification_obligation,
                    },
                    "history": [{"role": "tool", "content": last_output[:300]}],
                },
                "question": "Which action should be selected next?",
                "question_type": "choice",
                "answer_options": answer_options,
            }
        ).encode("utf-8")

        decision_data: dict[str, Any] | None = None
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8001/predict",
                data=pred_payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                decision_data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

        retrieval_summary = " > ".join(f"{f} ({s})" for f, s in ranked_files[:3])
        dec_label = decision_data.get("selected_label", "unknown") if decision_data else "unknown"
        opt_probs = (
            ", ".join(
                f"{opt['id']}: {opt['probability']:.2f}" for opt in decision_data.get("options", [])
            )
            if decision_data
            else ""
        )
        decision_summary = f"{dec_label} [{opt_probs}]"

        return {
            "retriever": {
                "model": "google/embeddinggemma-2",
                "ranked_files": ranked_files[:5],
                "summary": retrieval_summary,
            },
            "controller": {
                "model": "samatv256/mini-Jev",
                "revision": "step-010626",
                "selected_id": decision_data.get("selected_id") if decision_data else None,
                "selected_label": dec_label,
                "options": decision_data.get("options", []) if decision_data else [],
                "summary": decision_summary,
            },
            "retrieval_summary": retrieval_summary,
            "decision_summary": decision_summary,
        }
