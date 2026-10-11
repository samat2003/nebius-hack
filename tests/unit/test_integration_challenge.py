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
        step_actions = [s.get("action") or s.get("tool") for s in steps]
        assert "run_test" in step_actions
        assert "read_file" in step_actions
        assert "write_file" in step_actions
        assert "run_acceptance_test" in step_actions

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
