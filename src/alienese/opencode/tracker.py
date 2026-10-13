"""Alienese OpenCode Session Tracker.

Tracks evidence, enforces workspace containment and safety policies, derives
WorkingState with SHA-256 state digests, and produces redacted session traces
during interactive and headless OpenCode sessions.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
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
    "alienese_plan": CanonicalCapability.CUSTOM_TOOL,
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


def count_tokens(text: str) -> int:
    """Accurately count tokens using tiktoken (cl100k_base).

    Raises RuntimeError if tiktoken is unavailable; never silently substitutes word counts.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except Exception as exc:
        raise RuntimeError(f"Tokenizer failure: tiktoken cl100k_base unavailable: {exc}") from exc


def extract_opencode_token_usage(ws: Path, session_id: str) -> dict[str, int] | None:
    """Extract authentic token usage recorded by OpenCode from its session database."""
    db_file = ws / ".xdg" / "data" / "opencode" / "opencode.db"
    if not db_file.is_file():
        return None
    try:
        import sqlite3

        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT data FROM message WHERE session_id = ?", (session_id,))
        rows = cur.fetchall()
        conn.close()
        total_input = 0
        total_output = 0
        found = False
        for (raw,) in rows:
            try:
                d = json.loads(raw)
                toks = d.get("tokens", {})
                if toks and isinstance(toks, dict):
                    inp = int(toks.get("input", 0))
                    out = int(toks.get("output", 0))
                    total_input += inp
                    total_output += out
                    found = True
            except Exception:
                continue
        if found and (total_input > 0 or total_output > 0):
            return {
                "prompt_tokens": total_input,
                "completion_tokens": total_output,
                "total_tokens": total_input + total_output,
            }
    except Exception:
        pass
    return None


class OpenCodeSessionTracker:
    """Session tracker managing state derivation, safety policies, and evidence tracking."""

    def __init__(
        self,
        workspace: Path,
        session_id: str = "default",
        trace_file: Path | None = None,
        read_only_test_files: bool = False,
        three_model_mode: bool | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.session_id = session_id
        self.trace_file = (
            trace_file.resolve()
            if trace_file is not None
            else (self.workspace / ".alienese" / "opencode_trace.jsonl")
        )
        self.read_only_test_files = read_only_test_files
        self.three_model_mode = (
            three_model_mode
            if three_model_mode is not None
            else (os.environ.get("ALIENESE_THREE_MODEL") == "1")
        )

        self.sequence_no = 0
        self.events: list[NormalizedEvent] = []
        self.mutations: list[str] = []
        self.verification_obligation = False
        self.last_test_passed: bool | None = None
        self.state_digest = "0" * 64
        self.evidence_log: list[dict[str, Any]] = []
        self.plan_executed = False
        self.last_recommended_action: str | None = None
        self.last_adherence: dict[str, Any] | None = None
        self.plan_adherence: dict[str, Any] | None = None
        self.adherence_evaluated = False
        self.token_usage: dict[str, int] = {}

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
            self.plan_executed = bool(data.get("plan_executed", False))
            self.last_recommended_action = data.get("last_recommended_action")
            self.last_adherence = data.get("last_adherence")
            self.plan_adherence = data.get("plan_adherence")
            self.adherence_evaluated = bool(data.get("adherence_evaluated", False))
            self.token_usage = dict(data.get("token_usage", {}))
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
                    "content": redact_value(ev.content[:2048]),
                    "provenance": ev.provenance.model_dump(mode="json"),
                    "tool_name": ev.tool_name,
                    "tool_call_id": ev.tool_call_id,
                    "tool_arguments": redact_value(ev.tool_arguments),
                }
                for ev in self.events[-50:]  # keep recent 50
            ]
            data = {
                "session_id": self.session_id,
                "sequence_no": self.sequence_no,
                "mutations": list(self.mutations),
                "verification_obligation": self.verification_obligation,
                "last_test_passed": self.last_test_passed,
                "state_digest": self.state_digest,
                "plan_executed": self.plan_executed,
                "last_recommended_action": self.last_recommended_action,
                "last_adherence": self.last_adherence,
                "plan_adherence": self.plan_adherence,
                "adherence_evaluated": self.adherence_evaluated,
                "token_usage": self.token_usage,
                "evidence_log": redact_value(self.evidence_log[-50:]),
                "events": serialized_events,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(redact_value(data), f, indent=2)
        except Exception:
            pass

    def _check_test_hash_integrity(self) -> tuple[bool, str | None]:
        """Verify test files match the baseline SHA-256 hashes recorded at session start."""
        hash_file = self.workspace / ".alienese" / "test_baseline_hashes.json"
        if not hash_file.is_file():
            return True, None
        try:
            baseline = json.loads(hash_file.read_text(encoding="utf-8"))
            tests_dir = self.workspace / "tests"
            if not tests_dir.is_dir():
                return True, None
            for rel_path_str, expected_hash in baseline.items():
                test_file = self.workspace / rel_path_str
                if not test_file.is_file():
                    return False, f"Test file '{rel_path_str}' was deleted or moved."
                current_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()
                if current_hash != expected_hash:
                    return False, f"Test file '{rel_path_str}' was modified from baseline hash."
            return True, None
        except Exception:
            return True, None

    def _discover_candidate_files(self) -> list[tuple[str, str]]:
        """Deterministically gather candidate files prioritized by relevance.

        Priority order:
        1. Files from tracebacks and failure evidence.
        2. Files imported by test files via AST inspection.
        3. Alphabetically sorted repository source files.
        4. Alphabetically sorted test files.
        """
        discovered: list[str] = []
        seen: set[str] = set()

        def add_file(rel: str) -> None:
            rel_norm = rel.replace("\\", "/").strip("./")
            p = self.workspace / rel_norm
            if p.is_file() and rel_norm not in seen:
                parts = p.parts
                if not any(x in parts for x in [".alienese", ".venv", "__pycache__", ".git"]):
                    seen.add(rel_norm)
                    discovered.append(rel_norm)

        # 1. Traceback / failure files from evidence log
        for ev in reversed(self.evidence_log):
            val = str(ev.get("value", ""))
            for m in re.finditer(r'File "([^"]+\.py)"', val):
                f_candidate = m.group(1)
                try:
                    rel = Path(f_candidate).resolve().relative_to(self.workspace)
                    add_file(str(rel))
                except Exception:
                    add_file(f_candidate)
            for word in val.split():
                clean_word = word.strip(":,()[]'\"")
                if clean_word.endswith(".py"):
                    add_file(clean_word)

        # 2. AST imports from test files
        tests_dir = self.workspace / "tests"
        if tests_dir.is_dir():
            for t_path in sorted(tests_dir.rglob("*.py")):
                try:
                    tree = ast.parse(t_path.read_text(encoding="utf-8"))
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module:
                            mod_rel = node.module.replace(".", "/") + ".py"
                            add_file(mod_rel)
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                mod_rel = alias.name.replace(".", "/") + ".py"
                                add_file(mod_rel)
                except Exception:
                    pass

        # 3. Sorted repository .py files (source code first, then tests)
        all_py = sorted(self.workspace.rglob("*.py"))
        source_py = [p for p in all_py if "tests" not in p.parts and not p.name.startswith("test_")]
        test_py = [p for p in all_py if p not in source_py]

        for p in source_py + test_py:
            try:
                rel_path = str(p.relative_to(self.workspace))
                add_file(rel_path)
            except Exception:
                pass

        results: list[tuple[str, str]] = []
        for rel_str in discovered:
            p = self.workspace / rel_str
            try:
                snippet = p.read_text(encoding="utf-8")[:500]
                results.append((rel_str, snippet))
            except Exception:
                continue
        return results

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

        # 0. Canonical workspace containment check
        expected_ws = os.environ.get("ALIENESE_EXPECTED_WORKSPACE")
        if expected_ws:
            exp_resolved = Path(expected_ws).resolve()
            if self.workspace != exp_resolved:
                decision.update(
                    {
                        "allowed": False,
                        "decision": "REJECT",
                        "enforced_guard": True,
                        "guard_name": "workspace_containment",
                        "reason": (
                            f"Workspace mismatch: Session running in '{self.workspace}' but "
                            f"demo mode requires '{exp_resolved}'. "
                            "Rejecting uncontained demo operation."
                        ),
                    }
                )
                self._record_trace("PRE_TOOL", tool_name, args, decision)
                return decision

        # 0b. Test file hash integrity baseline check
        intact, tamper_msg = self._check_test_hash_integrity()
        if not intact:
            decision.update(
                {
                    "allowed": False,
                    "decision": "REJECT",
                    "enforced_guard": True,
                    "guard_name": "test_hash_integrity",
                    "reason": f"Test integrity violation detected: {tamper_msg}",
                }
            )
            self._record_trace("PRE_TOOL", tool_name, args, decision)
            return decision

        # 0c. Mandatory planning gate in three-model mode
        three_model_env = self.three_model_mode
        unguided_tools = {
            "glob",
            "grep",
            "search",
            "file_search",
            "find",
            "find_by_name",
            "list_dir",
            "list_directory",
            "ls",
            "read",
            "read_file",
            "view",
            "view_file",
            "open",
            "bash",
            "run_command",
            "command",
            "execute_command",
            "edit",
            "edit_file",
            "write",
            "write_file",
            "write_to_file",
            "replace_file_content",
            "patch",
            "apply_patch",
        }
        if (
            three_model_env
            and not self.plan_executed
            and tool_name != "alienese_plan"
            and tool_name in unguided_tools
        ):
            decision.update(
                {
                    "allowed": False,
                    "decision": "REJECT",
                    "enforced_guard": True,
                    "guard_name": "mandatory_planning_gate",
                    "reason": (
                        "Three-model orchestration is active. You must invoke the "
                        "'alienese_plan' tool first to retrieve and rank repository context "
                        "and select the optimal grounded action before inspecting files, "
                        "running commands, or mutating code."
                    ),
                }
            )
            self._record_trace("PRE_TOOL", tool_name, args, decision)
            return decision

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

            # System file access
            if "/etc/passwd" in cmd or "/etc/shadow" in cmd or "/etc/" in cmd:
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

            # Secret / credential access
            if any(sec in cmd for sec in [".env", "id_rsa", "id_ed25519", "credentials"]):
                decision.update(
                    {
                        "allowed": False,
                        "decision": "REJECT",
                        "enforced_guard": True,
                        "guard_name": "protected_file",
                        "reason": (
                            "Shell execution targeting protected secret files "
                            "(.env, keys, credentials) is strictly blocked."
                        ),
                    }
                )
                self._record_trace("PRE_TOOL", tool_name, args, decision)
                return decision

            # Workspace escaping check
            if re.search(r"\bcd\s+(\.\./\.\.|/|~)", cmd) or "rm -rf /" in cmd:
                decision.update(
                    {
                        "allowed": False,
                        "decision": "REJECT",
                        "enforced_guard": True,
                        "guard_name": "workspace_containment",
                        "reason": (
                            "Shell command attempting to escape repository "
                            "workspace root is blocked."
                        ),
                    }
                )
                self._record_trace("PRE_TOOL", tool_name, args, decision)
                return decision

            # Test file modification / tampering check
            if self.read_only_test_files and re.search(
                r"(?:>|>>)\s*.*tests/|\b(?:rm|mv|cp|truncate|chmod|touch)\s+.*tests/|\bsed\s+.*-i.*tests/",
                cmd,
            ):
                decision.update(
                    {
                        "allowed": False,
                        "decision": "REJECT",
                        "enforced_guard": True,
                        "guard_name": "read_only_test_policy",
                        "reason": (
                            "Shell command attempting to modify or delete test files "
                            "is strictly blocked."
                        ),
                    }
                )
                self._record_trace("PRE_TOOL", tool_name, args, decision)
                return decision

            # General shell execution is observed by Alienese
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

        # 3. Plan adherence tracking: measure first eligible action following the recommendation
        if self.last_recommended_action and tool_name != "alienese_plan":
            actual_action = f"{tool_name}: {raw_target or args.get('command') or ''}".strip()
            rec = self.last_recommended_action
            rec_tool = rec.split(":", 1)[0].strip()
            rec_arg = rec.split(":", 1)[1].strip() if ":" in rec else ""
            matched = (tool_name == rec_tool) and (not rec_arg or rec_arg in actual_action)
            adherence_info = {
                "recommended": self.last_recommended_action,
                "actual": actual_action,
                "matched": matched,
                "evaluated_at_step": self.sequence_no,
            }
            if not self.adherence_evaluated:
                self.plan_adherence = adherence_info
                self.adherence_evaluated = True
            self.last_adherence = adherence_info
            decision["adherence"] = adherence_info

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

        # 4b. Test file hash integrity baseline check on post tool
        intact, tamper_msg = self._check_test_hash_integrity()
        if not intact:
            self.evidence_log.append(
                {
                    "category": "integrity_violation",
                    "value": f"Test hash integrity failed: {tamper_msg}",
                    "provenance": provenance.model_dump(mode="json"),
                }
            )

        # 5. Three-model planning is driven on-demand via alienese_plan
        # Redundant per-tool synchronous model calls removed to preserve speed and avoid loops.
        three_model_guidance = None

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
            "adherence": self.last_adherence,
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
                "adherence": self.last_adherence,
                "details": redact_value(decision_or_result),
            }
            if output_summary is not None:
                entry["output_summary"] = redact_value(output_summary)

            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(redact_value(entry)) + "\n")
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
            with urllib.request.urlopen(req, timeout=5.0) as resp:
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
        if ranked_files:
            top_file = ranked_files[0][0]
        elif candidate_files:
            top_file = candidate_files[0][0]
        elif (self.workspace / "src/alienese/providers/runtime/client.py").is_file():
            top_file = "src/alienese/providers/runtime/client.py"
        else:
            top_file = "auth/session_manager.py"

        test_file = next(
            (f[0] for f in candidate_files if "test" in f[0]),
            "tests/unit/test_provider_runtime.py"
            if (self.workspace / "tests/unit/test_provider_runtime.py").is_file()
            else "tests/test_session.py",
        )

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
            with urllib.request.urlopen(req, timeout=5.0) as resp:
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

    def plan(self, goal: str = "") -> dict[str, Any]:
        """Execute multi-model planning using EmbeddingGemma 2 and mini-Jev."""
        import os
        import urllib.request

        three_model_env = self.three_model_mode

        server_healthy = False
        try:
            req = urllib.request.Request("http://127.0.0.1:8001/health")
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                if resp.status == 200:
                    health_data = json.loads(resp.read().decode("utf-8"))
                    server_healthy = health_data.get("status") == "ok"
        except Exception:
            server_healthy = False

        if not three_model_env or not server_healthy:
            msg = (
                "Alienese three-model services (EmbeddingGemma 2 retriever and mini-Jev "
                "controller) are unavailable. To enable multi-model planning, start the local "
                "model server on port 8001 and run OpenCode with `--three-model`."
            )
            plan_text = (
                "### Alienese Plan (Three-Model Service Unavailable)\n\n"
                f"**Notice:** {msg}\n\n"
                "Please proceed with standard investigation using `read` or `bash`."
            )
            return {
                "status": "unavailable",
                "three_model_active": False,
                "server_healthy": server_healthy,
                "message": msg,
                "plan_text": plan_text,
            }

        expected_ws = os.environ.get("ALIENESE_EXPECTED_WORKSPACE")
        if expected_ws:
            exp_resolved = Path(expected_ws).resolve()
            if self.workspace != exp_resolved:
                msg = (
                    f"Workspace mismatch: Session running in '{self.workspace}' but "
                    f"demo mode requires '{exp_resolved}'. Rejecting uncontained demo operation."
                )
                return {
                    "status": "error",
                    "three_model_active": False,
                    "message": msg,
                    "plan_text": f"### Alienese Plan Error\n\n{msg}",
                }

        # 1. Deterministically discover candidate python files in workspace
        candidate_files = self._discover_candidate_files()

        # Determine query text
        query_text = goal.strip()
        if not query_text:
            for ev in reversed(self.evidence_log):
                val = str(ev.get("value", ""))
                if ev.get("category") == "failed_test" or "error" in val.lower():
                    query_text = val[:200]
                    break
        if not query_text:
            query_text = "Fix test failure and session refresh regression"

        # 2. EmbeddingGemma 2 retrieval ranking
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
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                emb_data = json.loads(resp.read().decode("utf-8"))
                vectors = [item["embedding"] for item in emb_data["data"]]
                q_vec = vectors[0]
                for idx, (fpath, _) in enumerate(candidate_files[:8]):
                    d_vec = vectors[idx + 1]
                    dot = sum(a * b for a, b in zip(q_vec, d_vec, strict=True))
                    ranked_files.append((fpath, round(dot, 4)))
            ranked_files.sort(key=lambda x: -x[1])
        except Exception as e:
            return {
                "status": "error",
                "three_model_active": True,
                "message": f"EmbeddingGemma 2 retrieval error: {e}",
                "plan_text": f"Error during EmbeddingGemma ranking: {e}",
            }

        def is_test_file(fpath: str) -> bool:
            parts = Path(fpath).parts
            return any(p == "tests" for p in parts) or Path(fpath).name.startswith("test_")

        source_ranked = [f for f, s in ranked_files if not is_test_file(f)]
        test_candidates = [f for f, s in ranked_files if is_test_file(f)]

        if source_ranked:
            top_file = source_ranked[0]
        else:
            non_tests = [f[0] for f in candidate_files if not is_test_file(f[0])]
            top_file = non_tests[0] if non_tests else "auth/session_manager.py"

        if len(source_ranked) > 1:
            second_file = source_ranked[1]
        else:
            non_tests_remaining = [
                f[0] for f in candidate_files if not is_test_file(f[0]) and f[0] != top_file
            ]
            second_file = (
                non_tests_remaining[0]
                if non_tests_remaining
                else (
                    "auth/token_store.py"
                    if (self.workspace / "auth/token_store.py").is_file()
                    else top_file
                )
            )

        if test_candidates:
            test_file = test_candidates[0]
        else:
            test_file = next(
                (f[0] for f in candidate_files if is_test_file(f[0]) and "acceptance" not in f[0]),
                next(
                    (f[0] for f in candidate_files if is_test_file(f[0])),
                    "tests/test_session.py",
                ),
            )

        # Measure candidate context reduction using real tokenizer (tiktoken cl100k_base)
        candidate_tokens_map = {
            f: count_tokens(f"File: {f}\n{s}") for f, s in candidate_files
        }
        total_candidate_tokens = sum(candidate_tokens_map.values())
        top_snippet = next((s for f, s in candidate_files if f == top_file), "")
        selected_tokens = candidate_tokens_map.get(
            top_file, count_tokens(f"File: {top_file}\n{top_snippet}")
        )
        reduction_pct = round(
            (1.0 - (selected_tokens / max(1, total_candidate_tokens))) * 100.0, 1
        )
        candidate_reduction_info = {
            "metric": "Candidate Context Selection / Pruning (Representation)",
            "total_candidate_tokens": total_candidate_tokens,
            "selected_tokens": selected_tokens,
            "reduction_pct": reduction_pct,
            "formula": "1 - (selected_candidate_tokens / total_candidate_tokens)",
            "tokenizer": "tiktoken (cl100k_base)",
            "content_type": "Discovered candidate file representations (AST/Traceback/Repo)",
            "note": (
                "Measures candidate representation pruning. Does not represent "
                "full Nemotron input prompt tokens."
            ),
            "status": "measured",
        }

        # Connect real OpenCode/Nebius usage data if present
        live_usage = extract_opencode_token_usage(self.workspace, self.session_id)
        if live_usage:
            self.token_usage = live_usage

        # Metric B: Actual Nemotron input-token savings
        # (only when baseline exists and real usage connected)
        actual_api_savings: dict[str, Any] | None = None
        baseline_metrics_path = self.workspace / ".alienese" / "baseline_metrics.json"
        if baseline_metrics_path.is_file():
            try:
                bdata = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
                b_tokens = int(bdata.get("baseline_prompt_tokens", 0))
                a_tokens = int(self.token_usage.get("prompt_tokens", 0))
                if b_tokens > 0 and a_tokens > 0:
                    sav_pct = round((1.0 - (a_tokens / b_tokens)) * 100.0, 1)
                    actual_api_savings = {
                        "baseline_tokens": b_tokens,
                        "alienese_tokens": a_tokens,
                        "savings_pct": sav_pct,
                        "source": "Nebius API token usage metadata",
                        "status": "measured",
                    }
            except Exception:
                pass

        # 3. Construct multiple grounded next action options (K >= 2, K <= 8)
        answer_options = [
            {
                "id": "act_inspect_top",
                "label": f"read: {top_file}",
                "type": "external_tool",
                "description": f"Read and inspect primary candidate module {top_file}",
            },
            {
                "id": "act_run_pytest",
                "label": f"bash: pytest {test_file}",
                "type": "external_tool",
                "description": f"Run tests in {test_file} to verify current behavior",
            },
            {
                "id": "act_inspect_dep",
                "label": f"read: {second_file}",
                "type": "external_tool",
                "description": f"Inspect secondary dependency {second_file}",
            },
        ]

        # 4. mini-Jev decision controller evaluation
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
                        "available_tool_count": len(answer_options),
                        "mutation_attempted": bool(self.mutations),
                        "verification_confirmed": not self.verification_obligation,
                    },
                    "history": [
                        {"role": "user", "content": query_text},
                        {"role": "assistant", "content": f"Identified top candidate: {top_file}"},
                    ],
                },
                "question": "Which grounded action should be executed next?",
                "question_type": "choice",
                "answer_options": answer_options,
            }
        ).encode("utf-8")

        decision_data: dict[str, Any] = {}
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8001/predict",
                data=pred_payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                decision_data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {
                "status": "error",
                "three_model_active": True,
                "message": f"mini-Jev controller decision error: {e}",
                "plan_text": f"Error during mini-Jev decision evaluation: {e}",
            }

        selected_id = decision_data.get("selected_id", answer_options[0]["id"])
        selected_label = decision_data.get("selected_label", answer_options[0]["label"])
        options = decision_data.get("options", [])
        opt_probs_map = {opt["id"]: opt.get("probability", 0.0) for opt in options}
        selected_prob = opt_probs_map.get(selected_id, 0.0)

        # 5. Format structured markdown plan for Nemotron's context
        ranked_bullets = "\n".join(
            f"  {idx + 1}. `{f}` (cosine similarity: {score:.4f})"
            for idx, (f, score) in enumerate(ranked_files[:4])
        )
        options_bullets = "\n".join(
            f"  - `{opt['id']}`: {opt.get('label', opt['id'])} "
            f"(Softmax option score: {opt.get('probability', 0.0):.4f})"
            + (" [SELECTED]" if opt["id"] == selected_id else "")
            for opt in options
        )

        # Step directives dynamically follow mini-Jev's actual selected choice
        if selected_id == "act_run_pytest":
            step1 = (
                f"Run verification tests: `pytest {test_file}` via `bash` to observe reproduction."
            )
            step2 = f"Use `read` to inspect primary candidate module `{top_file}`."
            step3 = (
                f"Synthesize the fix in `{top_file}` using `edit` (or `write`) "
                f"and verify with `pytest {test_file}`."
            )
        elif selected_id == "act_inspect_dep":
            step1 = f"Use `read` to inspect dependency module `{second_file}`."
            step2 = (
                f"Use `read` to inspect primary module `{top_file}` "
                "and synthesize the fix using `edit`."
            )
            step3 = f"Run `pytest {test_file}` via `bash` to fulfill the verification obligation."
        else:
            step1 = f"Use `read` to inspect primary candidate module `{top_file}`."
            step2 = f"Synthesize the fix in `{top_file}` using `edit` (or `write`)."
            step3 = (
                f"Verify your fix by running `pytest {test_file}` via `bash` "
                "to fulfill the verification obligation."
            )

        api_savings_str = (
            f"{actual_api_savings['savings_pct']}% (measured vs baseline)"
            if actual_api_savings
            else "Not measured (Requires baseline benchmark run)"
        )

        plan_text = (
            "### Alienese Multi-Model Orchestration Plan\n\n"
            f"**Objective:** {query_text}\n\n"
            f"**Candidate Context Pruning (Metric A):** {reduction_pct}% "
            f"({selected_tokens:,} tokens selected from "
            f"{total_candidate_tokens:,} candidate tokens; tokenizer: tiktoken cl100k_base)\n"
            f"**Actual LLM API Savings (Metric B):** {api_savings_str}\n"
            "**Companion Dashboard:** Real-time metrics streaming at http://127.0.0.1:4097/\n\n"
            "**1. Retrieval Context Ranking "
            "(EmbeddingGemma 2 — 768d Dense Matryoshka Embeddings):**\n"
            f"{ranked_bullets}\n\n"
            "**2. Grounded Decision Policy (mini-Jev @ step-010626):**\n"
            f"Evaluated {len(answer_options)} grounded candidate actions with trained set-head:\n"
            f"{options_bullets}\n"
            f"- **Selected Policy Action:** `{selected_label}` "
            f"(Softmax option score: {selected_prob:.1%})\n\n"
            "**3. Concrete Directives for Coding Model:**\n"
            f"- **Step 1 (Selected Action: `{selected_label}`):** {step1}\n"
            f"- **Step 2:** {step2}\n"
            f"- **Step 3:** {step3}"
        )

        # 6. Update tracker state & record immutable trace record
        self.sequence_no += 1
        self.plan_executed = True
        self.last_recommended_action = selected_label
        self.adherence_evaluated = False
        self._save_session_state()

        plan_details = {
            "query": query_text,
            "ranked_files": ranked_files,
            "options": options,
            "selected_id": selected_id,
            "selected_label": selected_label,
            "selected_probability": selected_prob,
            "candidate_context_reduction": candidate_reduction_info,
            "context_reduction": candidate_reduction_info,
            "actual_api_savings": actual_api_savings,
        }
        self._record_trace(
            "ALIENESE_PLAN",
            "alienese_plan",
            {"goal": query_text},
            plan_details,
            output_summary=plan_text,
        )

        return {
            "status": "success",
            "three_model_active": True,
            "query": query_text,
            "top_file": top_file,
            "ranked_files": ranked_files,
            "options": options,
            "selected_id": selected_id,
            "selected_label": selected_label,
            "selected_probability": selected_prob,
            "candidate_context_reduction": candidate_reduction_info,
            "context_reduction": candidate_reduction_info,
            "actual_api_savings": actual_api_savings,
            "plan_text": plan_text,
        }
