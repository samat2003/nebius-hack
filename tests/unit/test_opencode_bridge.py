"""Unit and integration tests for Alienese OpenCode tracker and CLI bridge."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from alienese.opencode.tracker import OpenCodeSessionTracker


def test_opencode_tracker_workspace_containment() -> None:
    """Tracker strictly blocks paths escaping repository boundary."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        tracker = OpenCodeSessionTracker(workspace=ws, session_id="test-containment")

        # Absolute outside path
        res = tracker.evaluate_pre_tool("read", {"filePath": "/etc/passwd"})
        assert not res["allowed"]
        assert res["decision"] == "REJECT"
        assert res["guard_name"] == "workspace_containment"

        # Traversal outside path
        res = tracker.evaluate_pre_tool("read", {"filePath": "../../secret.txt"})
        assert not res["allowed"]
        assert res["decision"] == "REJECT"
        assert res["guard_name"] == "workspace_containment"

        # Normal repo path
        (ws / "main.py").write_text("print('hello')", encoding="utf-8")
        res = tracker.evaluate_pre_tool("read", {"filePath": "main.py"})
        assert res["allowed"]
        assert res["decision"] == "ALLOW"
        assert res["grounded_path"] == "main.py"


def test_opencode_tracker_protected_files() -> None:
    """Tracker blocks access to credentials, .env files, and .git directory."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        tracker = OpenCodeSessionTracker(workspace=ws, session_id="test-protected")

        # .env blocked
        res = tracker.evaluate_pre_tool("read", {"filePath": ".env"})
        assert not res["allowed"]
        assert res["decision"] == "REJECT"
        assert res["guard_name"] == "protected_file"

        # .git/config blocked
        res = tracker.evaluate_pre_tool("read", {"filePath": ".git/config"})
        assert not res["allowed"]
        assert res["decision"] == "REJECT"
        assert res["guard_name"] == "protected_file"

        # id_rsa blocked
        res = tracker.evaluate_pre_tool("read", {"filePath": "id_rsa"})
        assert not res["allowed"]
        assert res["decision"] == "REJECT"
        assert res["guard_name"] == "protected_file"


def test_opencode_tracker_read_only_tests() -> None:
    """Tracker blocks test file modifications when read-only test enforcement is enabled."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        tracker = OpenCodeSessionTracker(
            workspace=ws, session_id="test-readonly", read_only_test_files=True
        )

        # Modifying test file blocked
        res = tracker.evaluate_pre_tool(
            "edit", {"filePath": "tests/test_auth.py", "content": "assert True"}
        )
        assert not res["allowed"]
        assert res["decision"] == "REJECT"
        assert res["guard_name"] == "read_only_test_policy"

        # Modifying source code permitted
        res = tracker.evaluate_pre_tool(
            "edit", {"filePath": "auth/service.py", "content": "return True"}
        )
        assert res["allowed"]
        assert res["decision"] == "ALLOW"


def test_opencode_tracker_verification_obligation_and_evidence() -> None:
    """Tracker updates verification obligations and derives evidence from test results."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        trace_file = ws / ".alienese" / "test_trace.jsonl"
        tracker = OpenCodeSessionTracker(
            workspace=ws, session_id="test-lifecycle", trace_file=trace_file
        )

        assert not tracker.verification_obligation
        assert len(tracker.mutations) == 0

        # Step 1: Pre-tool edit sets up mutation call
        pre_edit = tracker.evaluate_pre_tool("edit", {"filePath": "src/core.py"}, call_id="c1")
        assert pre_edit["allowed"]

        # Step 2: Post-tool edit executes -> sets verification obligation = True
        post_edit = tracker.process_post_tool(
            "edit",
            {"filePath": "src/core.py"},
            output_data="File src/core.py successfully edited.",
            call_id="c1",
        )
        assert post_edit["verification_obligation"] is True
        assert "src/core.py" in post_edit["mutations"]
        assert tracker.verification_obligation is True
        assert len(post_edit["state_digest"]) == 64

        # Step 3: Run pytest that fails -> obligation remains True
        failing_output = """
=================================== FAILURES ===================================
________________________________ test_something ________________________________
    def test_something():
>       assert False
E       AssertionError: assert False
tests/test_core.py:12: AssertionError
=========================== 1 failed in 0.12s ===========================
"""
        pre_test1 = tracker.evaluate_pre_tool(
            "bash", {"command": "pytest tests/test_core.py"}, call_id="c2"
        )
        assert pre_test1["allowed"]

        post_test1 = tracker.process_post_tool(
            "bash",
            {"command": "pytest tests/test_core.py"},
            output_data=failing_output,
            call_id="c2",
        )
        assert post_test1["verification_obligation"] is True
        assert post_test1["last_test_passed"] is False
        assert post_test1["evidence_count"] > 0

        # Step 4: Run pytest that passes -> obligation is resolved (False)
        passing_output = "=========================== 1 passed in 0.05s ==========================="
        pre_test2 = tracker.evaluate_pre_tool(
            "bash", {"command": "pytest tests/test_core.py"}, call_id="c3"
        )
        assert pre_test2["allowed"]

        post_test2 = tracker.process_post_tool(
            "bash",
            {"command": "pytest tests/test_core.py"},
            output_data=passing_output,
            call_id="c3",
        )
        assert post_test2["verification_obligation"] is False
        assert post_test2["last_test_passed"] is True

        # Verify trace file exists and contains entries
        assert trace_file.is_file()
        lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 4
        parsed = [json.loads(line) for line in lines]
        assert any(p["phase"] == "PRE_TOOL" for p in parsed)
        assert any(p["phase"] == "POST_TOOL" for p in parsed)


def test_opencode_tracker_secret_redaction() -> None:
    """Tracker redacts sensitive tokens and API keys in traces."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        trace_file = ws / ".alienese" / "redaction_trace.jsonl"
        tracker = OpenCodeSessionTracker(
            workspace=ws, session_id="test-redaction", trace_file=trace_file
        )

        secret_token = "Bearer nbf_test_secret_key_123456789"
        tracker.evaluate_pre_tool(
            "bash",
            {"command": f"curl -H 'Authorization: {secret_token}' http://example.com"},
            call_id="c_sec",
        )

        trace_text = trace_file.read_text(encoding="utf-8")
        assert any(marker in trace_text for marker in ["[REDACTED]", "***", "[FILTERED]", "Bearer"])


def test_cli_bridge_subprocess() -> None:
    """CLI bridge operates correctly via subprocess invocations."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        (ws / "app.py").write_text("print('test')", encoding="utf-8")

        # 1. pre-tool test via subprocess
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "alienese.opencode.bridge",
                "pre-tool",
                "--workspace",
                str(ws),
                "--session",
                "cli-test",
                "--tool",
                "read",
                "--args",
                json.dumps({"filePath": "app.py"}),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        res = json.loads(proc.stdout.strip())
        assert res["allowed"] is True
        assert res["decision"] == "ALLOW"
        assert res["grounded_path"] == "app.py"

        # 2. post-tool test via subprocess
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "alienese.opencode.bridge",
                "post-tool",
                "--workspace",
                str(ws),
                "--session",
                "cli-test",
                "--tool",
                "read",
                "--args",
                json.dumps({"filePath": "app.py"}),
                "--output",
                "print('test')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        post_res = json.loads(proc.stdout.strip())
        assert post_res["session_id"] == "cli-test"
        assert len(post_res["state_digest"]) == 64

        # 3. status test via subprocess
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "alienese.opencode.bridge",
                "status",
                "--workspace",
                str(ws),
                "--session",
                "cli-test",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        status_res = json.loads(proc.stdout.strip())
        assert status_res["session_id"] == "cli-test"
        assert status_res["sequence_no"] >= 2

        # 4. plan subcommand test via subprocess when three-model is disabled
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "alienese.opencode.bridge",
                "plan",
                "--workspace",
                str(ws),
                "--session",
                "cli-test",
                "--goal",
                "inspect auth service",
            ],
            capture_output=True,
            text=True,
            check=True,
            env={"PATH": sys.exec_prefix + "/bin", "ALIENESE_THREE_MODEL": "0"},
        )
        plan_res = json.loads(proc.stdout.strip())
        assert plan_res["status"] == "unavailable"
        assert plan_res["three_model_active"] is False
        assert "plan_text" in plan_res


def test_opencode_tracker_expected_workspace_containment(monkeypatch) -> None:
    """Tracker strictly rejects operations when workspace does not match expected demo root."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        expected = ws / "expected_repo"
        expected.mkdir()
        monkeypatch.setenv("ALIENESE_EXPECTED_WORKSPACE", str(expected))

        tracker = OpenCodeSessionTracker(workspace=ws, session_id="test-ws-mismatch")
        res = tracker.evaluate_pre_tool("read", {"filePath": "app.py"})
        assert not res["allowed"]
        assert res["decision"] == "REJECT"
        assert res["guard_name"] == "workspace_containment"
        assert "Workspace mismatch" in res["reason"]


def test_opencode_tracker_test_hash_integrity() -> None:
    """Tracker detects tampering with test files based on test_baseline_hashes.json."""
    import hashlib

    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        tests_dir = ws / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_sample.py"
        test_file.write_text("def test_one(): assert True\n", encoding="utf-8")

        # Record baseline
        trace_dir = ws / ".alienese"
        trace_dir.mkdir()
        baseline = {"tests/test_sample.py": hashlib.sha256(test_file.read_bytes()).hexdigest()}
        (trace_dir / "test_baseline_hashes.json").write_text(json.dumps(baseline), encoding="utf-8")

        tracker = OpenCodeSessionTracker(workspace=ws, session_id="test-hashes")

        # Initially intact
        res_ok = tracker.evaluate_pre_tool("read", {"filePath": "tests/test_sample.py"})
        assert res_ok["allowed"]

        # Tamper with test file
        test_file.write_text("def test_one(): pass # modified\n", encoding="utf-8")

        res_tampered = tracker.evaluate_pre_tool("read", {"filePath": "tests/test_sample.py"})
        assert not res_tampered["allowed"]
        assert res_tampered["guard_name"] == "test_hash_integrity"


def test_opencode_tracker_mandatory_planning_gate(monkeypatch) -> None:
    """Unguided tool execution is blocked in three-model mode until alienese_plan executes."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        (ws / "app.py").write_text("print('hello')", encoding="utf-8")
        monkeypatch.setenv("ALIENESE_THREE_MODEL", "1")

        tracker = OpenCodeSessionTracker(workspace=ws, session_id="test-gate")

        # Block unguided read before plan
        res_read = tracker.evaluate_pre_tool("read", {"filePath": "app.py"})
        assert not res_read["allowed"]
        assert res_read["guard_name"] == "mandatory_planning_gate"

        # Block unguided bash before plan
        res_bash = tracker.evaluate_pre_tool("bash", {"command": "pytest"})
        assert not res_bash["allowed"]
        assert res_bash["guard_name"] == "mandatory_planning_gate"


def test_opencode_tracker_shell_isolation() -> None:
    """Bash guard blocks test tampering, credential access, and directory traversal."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        tracker = OpenCodeSessionTracker(
            workspace=ws, session_id="test-shell-iso", read_only_test_files=True
        )

        # Tampering with tests via shell write redirect blocked
        res_write_test = tracker.evaluate_pre_tool(
            "bash", {"command": "echo 'pass' > tests/test_sample.py"}
        )
        assert not res_write_test["allowed"]
        assert res_write_test["guard_name"] == "read_only_test_policy"

        # Accessing .env blocked
        res_env = tracker.evaluate_pre_tool("bash", {"command": "cat .env"})
        assert not res_env["allowed"]
        assert res_env["guard_name"] == "protected_file"

        # Workspace traversal blocked
        res_escape = tracker.evaluate_pre_tool("bash", {"command": "cd ../../ && ls"})
        assert not res_escape["allowed"]
        assert res_escape["guard_name"] == "workspace_containment"


def test_opencode_tracker_plan_adherence() -> None:
    """Tracker accurately compares recommended actions with actual tool invocations."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        tracker = OpenCodeSessionTracker(workspace=ws, session_id="test-adherence")
        tracker.last_recommended_action = "read: auth/session_manager.py"

        # Matching tool and target
        res_match = tracker.evaluate_pre_tool("read", {"filePath": "auth/session_manager.py"})
        assert res_match.get("adherence", {}).get("matched") is True

        # Non-matching tool
        res_mismatch = tracker.evaluate_pre_tool("bash", {"command": "pytest"})
        assert res_mismatch.get("adherence", {}).get("matched") is False


def test_dashboard_server_endpoints() -> None:
    """Visual companion dashboard generates HTML and serves JSON status data."""
    from alienese.dashboard.server import generate_html_dashboard, read_workspace_data

    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        data = read_workspace_data(ws)
        assert "models" in data
        assert "context_reduction" in data
        assert "verification_status" in data

        html_text = generate_html_dashboard(data)
        assert "<!DOCTYPE html>" in html_text
        assert "Alienese Live Orchestration Engine" in html_text
        assert "EmbeddingGemma 2" in html_text
        assert "mini-Jev" in html_text
        assert "Nemotron 3 Super" in html_text
