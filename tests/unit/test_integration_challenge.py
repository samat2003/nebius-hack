"""Unit tests for the Alienese integration debugging challenge and multi-task runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from scripts.run_coding_task import (
    TASKS,
    StandaloneCodingHarness,
    run_failing_to_passing_workflow,
)


def test_task_registry_and_aliases() -> None:
    """Verifies task registry contains token and integration challenge with aliases."""
    assert "token" in TASKS
    assert "integration" in TASKS
    assert "session" in TASKS

    # session is an alias for integration
    assert TASKS["session"] is TASKS["integration"]

    token_task = TASKS["token"]
    assert token_task.name == "token"
    assert token_task.target_module == "auth/token.py"
    assert token_task.target_test == "tests/test_token.py"

    integration_task = TASKS["integration"]
    assert integration_task.name == "integration"
    assert integration_task.target_module == "auth/session_manager.py"
    assert integration_task.target_test == "tests/test_session.py"
    assert integration_task.acceptance_test == "tests/test_acceptance.py"
    assert "auth/session_manager.py" in integration_task.writable_allowlist
    assert "auth/token_store.py" in integration_task.writable_allowlist


def test_integration_fixture_initial_failure_and_traceback() -> None:
    """Verifies integration challenge initially fails with genuine cross-module regression."""
    task = TASKS["integration"]
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str)
        harness = StandaloneCodingHarness(ws, writable_allowlist=task.writable_allowlist)
        harness.setup_task_repo(task)

        # Initial unit test run must fail
        ok, out = harness.execute_tool("run_test", {"target": task.target_test})
        assert not ok, "Initial test must fail"
        assert "FAILED tests/test_session.py::test_session_refresh_lifecycle" in out
        assert "AssertionError" in out
        assert "2 passed" in out  # 2 other tests pass

        # Initial acceptance test must also fail
        ok_acc, _ = harness.execute_tool("run_test", {"target": task.acceptance_test})
        assert not ok_acc, "Initial acceptance test must fail prior to bug fix"


def test_integration_harness_containment_and_safety() -> None:
    """Verifies harness prevents write escaping and test tampering in the integration fixture."""
    task = TASKS["integration"]
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str)
        harness = StandaloneCodingHarness(ws, writable_allowlist=task.writable_allowlist)
        harness.setup_task_repo(task)

        # Attempting to tamper with test files must be rejected
        ok, msg = harness.execute_tool(
            "write_file",
            {"path": "tests/test_session.py", "content": "def test_bypass(): pass"},
        )
        assert not ok
        assert "allowlist" in msg.lower() or "forbidden" in msg.lower()

        ok, msg = harness.execute_tool(
            "write_file",
            {"path": "tests/test_acceptance.py", "content": "def test_bypass(): pass"},
        )
        assert not ok
        assert "allowlist" in msg.lower()

        # Attempting path traversal outside workspace must be rejected
        ok, msg = harness.execute_tool(
            "write_file",
            {"path": "../malicious.py", "content": "print('malicious')"},
        )
        assert not ok
        assert "permission denied" in msg.lower() or "traversal" in msg.lower()

        # Writing to allowlisted module must succeed
        ok, msg = harness.execute_tool(
            "write_file",
            {"path": "auth/session_manager.py", "content": "# safe write\n"},
        )
        assert ok


@pytest.mark.asyncio
async def test_integration_workflow_offline_simulation() -> None:
    """Verifies that the integration workflow runs end-to-end in offline simulation mode."""
    with tempfile.TemporaryDirectory() as tmp_str:
        log_path = Path(tmp_str) / "offline_result.json"
        summary = await run_failing_to_passing_workflow(
            task_name="integration",
            use_live_nebius=False,
            output_log_path=log_path,
        )

        assert summary["status"] == "SUCCESS"
        assert summary["task"] == "integration"
        assert summary["is_live_inference"] is False
        assert "SIMULATED / REPLAYED" in (summary.get("simulation_notice") or "")

        # Verify steps and diff
        steps = summary["steps"]
        step_actions = [s.get("tool") or s.get("action") for s in steps]
        assert "run_test" in step_actions
        assert "read_file" in step_actions
        assert "write_file" in step_actions
        assert "run_acceptance_test" in step_actions

        step_types = [s.get("step_type") for s in steps]
        assert "alienese_decision" in step_types
        assert "harness_inspection" in step_types
        assert "model_generation" in step_types
        assert "harness_verification" in step_types
        assert "harness_acceptance" in step_types

        # Target test and acceptance test passed
        test_step = next(s for s in steps if s["step"] == 4)
        assert test_step["passed"] is True
        acc_step = next(s for s in steps if s["step"] == 5)
        assert acc_step["passed"] is True

        # Diff touched auth/session_manager.py
        diff = summary["applied_diff"]
        assert "auth/session_manager.py" in diff
        assert 'session["active_token_id"] = new_token_id' in diff

        # JSON log was written
        assert log_path.exists()
        assert log_path.stat().st_size > 0


def test_repair_prompt_no_answer_leakage_and_acceptance_separation() -> None:
    """Verifies repair prompt contains NO answer-revealing hints and NO acceptance tests."""
    task = TASKS["integration"]
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str)
        harness = StandaloneCodingHarness(ws, writable_allowlist=task.writable_allowlist)
        harness.setup_task_repo(task)

        # Execute initial test to get real failure output
        ok_init, test_out = harness.execute_tool("run_test", {"target": task.target_test})
        assert not ok_init

        # Ground test file
        ok_read, test_code = harness.execute_tool("read_file", {"path": task.target_test})
        assert ok_read

        # Read source modules
        ok_store, store_code = harness.execute_tool("read_file", {"path": "auth/token_store.py"})
        ok_sess, sess_code = harness.execute_tool("read_file", {"path": "auth/session_manager.py"})
        assert ok_store and ok_sess

        # Build prompt exactly as runner does
        prompt_parts = [
            "You are an expert Python software engineer.",
            f"A test failed in the test suite:\n{test_out}\n",
            f"Here is the content of the failing test file '{task.target_test}':\n{test_code}\n",
            f"Here is the content of the module 'auth/token_store.py':\n{store_code}\n",
            f"Here is the content of the module 'auth/session_manager.py':\n{sess_code}\n",
            f"Fix the bug in '{task.target_module}' so that all tests pass.\n"
            f"Output ONLY the complete, corrected Python code for '{task.target_module}' "
            "inside a ```python``` code block.\n"
            "Do not include any explanation before or after.",
        ]
        prompt = "\n".join(prompt_parts)

        # 1. No leaked answer comments
        assert "# BUG" not in prompt
        assert "was not updated" not in prompt

        # 2. No leaked solution replacement statement in prompt
        assert 'session["active_token_id"] = new_token_id' not in prompt
        assert "session['active_token_id'] = new_token_id" not in prompt

        # 3. Acceptance tests strictly excluded
        assert "test_session_multi_refresh_continuity" not in prompt
        assert "test_unregistered_session_refresh_rejected" not in prompt
        assert "test_revoked_token_session_refresh_rejected_without_mutation" not in prompt
        assert "tests/test_acceptance.py" not in prompt

        # 4. Context contains authentic buggy code and failing test
        assert "def test_session_refresh_lifecycle():" in prompt
        assert "class SessionManager:" in prompt
        assert "class TokenStore:" in prompt


def test_revoked_token_refresh_regression_without_mutation() -> None:
    """Verifies refreshing a session with a revoked token returns None without mutating state."""
    task = TASKS["integration"]
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str)
        harness = StandaloneCodingHarness(ws, writable_allowlist=task.writable_allowlist)
        harness.setup_task_repo(task)

        # Apply the verified offline fix
        ok_write, _ = harness.execute_tool(
            "write_file",
            {
                "path": task.target_module,
                "content": task.offline_simulated_repair.strip("`\n").removeprefix("python\n"),
            },
        )
        assert ok_write

        # Run acceptance test suite which now includes the revoked token rollback check
        ok_acc, acc_out = harness.execute_tool("run_test", {"target": task.acceptance_test})
        assert ok_acc, f"Acceptance test failed: {acc_out}"
        assert "test_revoked_token_session_refresh_rejected_without_mutation PASSED" in acc_out
