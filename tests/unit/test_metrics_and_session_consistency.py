"""Unit tests for Alienese metric integrity, session isolation, and enforcement guards.

Verifies:
1. Zero fabricated dashboard metrics (unmeasured values show "Not measured").
2. Context reduction and API savings integrity (Metric A and Metric B).
3. Strict session isolation (no bleed of events, plans, or mutations between sessions).
4. Service health accurately reflects offline model services.
5. Mandatory planning gate blocks unguided tools before alienese_plan.
6. First-action plan adherence tracking.
7. mini-Jev selected option alignment with Step 1 directive and test-file exclusion.
8. Defense-in-depth protection of test files (guard + OS filesystem read-only permissions).
9. Personal OpenCode database is untouched by demo setup.
10. Process ownership validation protects recycled/unrelated PIDs.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts.prepare_opencode_demo import (
    _validate_process_ownership,
    safe_rmtree,
    setup_demo_workspace,
)

from alienese.dashboard.server import (
    check_service_health,
    generate_html_dashboard,
    read_workspace_data,
)
from alienese.opencode.tracker import OpenCodeSessionTracker


def test_candidate_context_reduction_not_measured_when_no_plan() -> None:
    """When no plan has run, candidate context reduction must be None / 'Not measured'."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        data = read_workspace_data(ws)

        assert data["candidate_context_reduction"] is None
        assert data["actual_api_savings"] is None
        assert data["tokens_per_model"] is None

        html = generate_html_dashboard(data)
        assert "Not measured" in html
        assert "Candidate Context Pruning" in html
        assert "API Token Savings" in html


def test_actual_api_savings_not_measured_without_baseline() -> None:
    """Without an authentic baseline measurement file, API token savings is None."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        tracker = OpenCodeSessionTracker(
            workspace=ws, session_id="test-savings", three_model_mode=True
        )

        # Create dummy file to rank
        (ws / "service.py").write_text("def run(): pass\n" * 20, encoding="utf-8")

        plan = tracker.plan("Fix the service bug")
        assert plan["candidate_context_reduction"] is not None
        assert plan["actual_api_savings"] is None

        data = read_workspace_data(ws, target_session_id="test-savings")
        assert data["candidate_context_reduction"] is not None
        assert data["actual_api_savings"] is None

        html = generate_html_dashboard(data)
        assert "Not measured" in html


def test_session_isolation_no_data_bleed() -> None:
    """Events, mutations, and plans from session A must never bleed into session B."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()

        # Session A: plans and mutates auth/session_manager.py
        tracker_a = OpenCodeSessionTracker(
            workspace=ws, session_id="session-A", three_model_mode=True
        )
        (ws / "auth").mkdir(parents=True, exist_ok=True)
        (ws / "auth" / "session_manager.py").write_text(
            "class SessionManager: pass\n", encoding="utf-8"
        )
        tracker_a.plan("Investigate session manager")
        tracker_a.process_post_tool(
            "edit", {"filePath": "auth/session_manager.py"}, "Updated session"
        )

        # Session B: no plan, mutates auth/token_store.py
        tracker_b = OpenCodeSessionTracker(
            workspace=ws, session_id="session-B", three_model_mode=True
        )
        (ws / "auth" / "token_store.py").write_text("class TokenStore: pass\n", encoding="utf-8")
        tracker_b.process_post_tool(
            "edit", {"filePath": "auth/token_store.py"}, "Updated token store"
        )

        # Inspect Session B in dashboard
        data_b = read_workspace_data(ws, target_session_id="session-B")
        assert data_b["session_id"] == "session-B"
        assert data_b["mutations"] == ["auth/token_store.py"]
        assert "auth/session_manager.py" not in data_b["mutations"]
        assert data_b["candidate_context_reduction"] is None
        assert data_b["ranked_files"] == []
        assert all(ev.get("session_id") == "session-B" for ev in data_b["trace_events"])

        # Inspect Session A in dashboard
        data_a = read_workspace_data(ws, target_session_id="session-A")
        assert data_a["session_id"] == "session-A"
        assert "auth/session_manager.py" in data_a["mutations"]
        assert "auth/token_store.py" not in data_a["mutations"]
        assert data_a["candidate_context_reduction"] is not None
        assert all(ev.get("session_id") == "session-A" for ev in data_a["trace_events"])


def test_stopped_services_display_unavailable() -> None:
    """Offline model services must display 'Unavailable (Offline)', never 'Ready'."""
    # Port 19999 has no listener
    health = check_service_health(base_url="http://127.0.0.1:19999")
    assert health["retriever"]["status"] == "Unavailable"
    assert health["controller"]["status"] == "Unavailable"

    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        data = read_workspace_data(ws)
        with patch("alienese.dashboard.server.check_service_health", return_value=health):
            data["models"]["retriever"]["status"] = health["retriever"]["status"]
            data["models"]["controller"]["status"] = health["controller"]["status"]
            html = generate_html_dashboard(data)
            assert "Unavailable" in html


def test_planning_gate_blocks_unguided_tools_before_alienese_plan() -> None:
    """Unguided exploration, search, and mutation tools must be rejected before planning."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        tracker = OpenCodeSessionTracker(
            workspace=ws, session_id="test-gate", three_model_mode=True
        )

        (ws / "auth").mkdir(parents=True, exist_ok=True)
        (ws / "auth" / "session_manager.py").write_text(
            "class SessionManager: pass\n", encoding="utf-8"
        )

        # 1. Block unguided read
        res_read = tracker.evaluate_pre_tool("read", {"filePath": "auth/session_manager.py"})
        assert not res_read["allowed"]
        assert res_read["guard_name"] == "mandatory_planning_gate"

        # 2. Block unguided grep
        res_grep = tracker.evaluate_pre_tool("grep", {"pattern": "def "})
        assert not res_grep["allowed"]
        assert res_grep["guard_name"] == "mandatory_planning_gate"

        # 3. Block unguided glob / search
        res_glob = tracker.evaluate_pre_tool("glob", {"pattern": "*.py"})
        assert not res_glob["allowed"]
        assert res_glob["guard_name"] == "mandatory_planning_gate"

        # 4. Block unguided mutation
        res_edit = tracker.evaluate_pre_tool(
            "edit", {"filePath": "auth/session_manager.py", "content": "# fix"}
        )
        assert not res_edit["allowed"]
        assert res_edit["guard_name"] == "mandatory_planning_gate"

        # 5. Planning tool itself is permitted
        res_plan = tracker.evaluate_pre_tool(
            "alienese_plan", {"task_description": "Fix session bug"}
        )
        assert res_plan["allowed"]

        # Run plan
        tracker.plan("Fix session bug")
        assert tracker.plan_executed

        # 6. Now read is permitted
        res_after = tracker.evaluate_pre_tool("read", {"filePath": "auth/session_manager.py"})
        assert res_after["allowed"]


def test_first_action_plan_adherence_tracking() -> None:
    """Plan adherence must evaluate on the FIRST eligible action following recommendation."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        tracker = OpenCodeSessionTracker(
            workspace=ws, session_id="test-adherence", three_model_mode=True
        )

        (ws / "auth").mkdir(parents=True, exist_ok=True)
        (ws / "auth" / "session_manager.py").write_text(
            "class SessionManager: pass\n", encoding="utf-8"
        )

        plan = tracker.plan("Fix session refresh bug")
        top_file = plan.get("top_file") or "auth/session_manager.py"

        # First action matches recommended target
        tracker.evaluate_pre_tool("read", {"filePath": top_file})
        assert tracker.adherence_evaluated
        assert tracker.plan_adherence is not None
        assert tracker.plan_adherence["matched"] is True

        # Subsequent action on a different file does not overwrite first-action adherence
        tracker.evaluate_pre_tool("read", {"filePath": "auth/other.py"})
        assert tracker.plan_adherence["matched"] is True


def test_minijev_selected_option_matches_step1_and_excludes_test_files() -> None:
    """mini-Jev options must exclude test files for edits, and Step 1 directive must match."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        (ws / "auth").mkdir(parents=True, exist_ok=True)
        (ws / "auth" / "session_manager.py").write_text(
            "class SessionManager: pass\n", encoding="utf-8"
        )
        (ws / "tests").mkdir(parents=True, exist_ok=True)
        (ws / "tests" / "test_session.py").write_text("def test_ok(): pass\n", encoding="utf-8")

        tracker = OpenCodeSessionTracker(
            workspace=ws, session_id="test-directive", three_model_mode=True
        )
        plan = tracker.plan("Fix the session refresh bug in auth/session_manager.py")

        selected_label = plan.get("selected_label", "")
        plan_text = plan.get("plan_text", "")

        assert selected_label != ""
        assert "Step 1" in plan_text
        assert selected_label in plan_text

        # Check candidate options: source edit options must NOT target test files
        options = plan.get("options", [])
        for opt in options:
            label = opt.get("label", "")
            if "edit" in label.lower() or "patch" in label.lower() or "repair" in label.lower():
                assert "tests/" not in label, (
                    f"Source repair candidate proposed a test file: {label}"
                )


def test_defense_in_depth_test_protection() -> None:
    """Test files must be protected at both guard level and OS filesystem level."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        setup_demo_workspace(ws)

        tests_dir = ws / "tests"
        test_file = tests_dir / "test_session.py"
        assert test_file.is_file()

        # 1. Tracker policy level blocks edit
        tracker = OpenCodeSessionTracker(
            workspace=ws, session_id="test-ro", read_only_test_files=True
        )
        tracker.plan("Fix session bug")
        res = tracker.evaluate_pre_tool(
            "edit", {"filePath": "tests/test_session.py", "content": "assert True"}
        )
        assert not res["allowed"]
        assert res["guard_name"] == "read_only_test_policy"

        # 2. OS filesystem level blocks modification
        with pytest.raises(PermissionError):
            test_file.write_text("assert True", encoding="utf-8")

        # 3. OS filesystem level blocks deletion
        with pytest.raises(PermissionError):
            test_file.unlink()

        # 4. OS filesystem level blocks creating new test files
        with pytest.raises(PermissionError):
            (tests_dir / "malicious_test.py").write_text("assert True", encoding="utf-8")

        # 5. Clean teardown with safe_rmtree succeeds despite read-only permissions
        safe_rmtree(ws)
        assert not ws.exists()


def test_personal_opencode_db_untouched() -> None:
    """Demo workspace setup must NEVER touch ~/.local/share/opencode/opencode.db."""
    personal_db = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    personal_db_mtime = personal_db.stat().st_mtime if personal_db.is_file() else None

    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        setup_demo_workspace(ws)

        # Isolated database directory must be created inside demo workspace
        isolated_data = ws / ".xdg" / "data"
        assert isolated_data.is_dir()

        # Personal database must remain unchanged
        if personal_db.is_file():
            assert personal_db.stat().st_mtime == personal_db_mtime, (
                "Personal OpenCode database was modified by demo setup!"
            )


def test_process_ownership_validation() -> None:
    """_validate_process_ownership rejects PIDs outside the workspace or wrong command."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        current_pid = os.getpid()

        # Current test runner process is not in tmp workspace
        is_valid = _validate_process_ownership(current_pid, ws, expected_pattern="opencode")
        assert not is_valid

        # Non-existent PID
        assert not _validate_process_ownership(9999999, ws)


def test_dashboard_http_endpoints_session_filtering() -> None:
    """Dashboard HTTP endpoints filter by session_id."""
    import socket
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from alienese.dashboard.server import DashboardRequestHandler

    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()

        # Setup sessions
        tracker_1 = OpenCodeSessionTracker(
            workspace=ws, session_id="sess-alpha", three_model_mode=True
        )
        (ws / "auth").mkdir(parents=True, exist_ok=True)
        (ws / "auth" / "session_manager.py").write_text(
            "class SessionManager: pass\n", encoding="utf-8"
        )
        tracker_1.plan("Investigate session manager")

        tracker_2 = OpenCodeSessionTracker(
            workspace=ws, session_id="sess-beta", three_model_mode=True
        )
        tracker_2.process_post_tool("edit", {"filePath": "auth/session_manager.py"}, "Fixed bug")

        # Pick ephemeral port
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        DashboardRequestHandler.workspace = ws
        server = ThreadingHTTPServer(("127.0.0.1", port), DashboardRequestHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            # Query /api/metrics for sess-alpha
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/metrics?session_id=sess-alpha"
            ) as resp:
                data = json.loads(resp.read().decode())
                assert data["session_id"] == "sess-alpha"
                assert data["candidate_context_reduction"] is not None

            # Query /api/metrics for sess-beta
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/metrics?session_id=sess-beta"
            ) as resp:
                data = json.loads(resp.read().decode())
                assert data["session_id"] == "sess-beta"
                assert data["candidate_context_reduction"] is None

            # Query /api/events for sess-alpha
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/events?session_id=sess-alpha"
            ) as resp:
                data = json.loads(resp.read().decode())
                assert data["session_id"] == "sess-alpha"
                assert all(ev.get("session_id") == "sess-alpha" for ev in data["events"])

            # Query /api/plan for sess-alpha
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/plan?session_id=sess-alpha"
            ) as resp:
                data = json.loads(resp.read().decode())
                assert data["session_id"] == "sess-alpha"
                assert len(data["ranked_files"]) > 0
        finally:
            server.shutdown()
            server.server_close()


def test_session_id_security_path_traversal_and_allowlist() -> None:
    """Session IDs must be strictly allowlisted and path containment enforced."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        tracker = OpenCodeSessionTracker(
            workspace=ws, session_id="legit-session-1", three_model_mode=True
        )
        (ws / "service.py").write_text("def run(): pass\n", encoding="utf-8")
        tracker.plan("Legit plan")

        # 1. Path traversal target_session_id is rejected and falls back to allowlist
        data_traversal = read_workspace_data(ws, target_session_id="../../etc/passwd")
        assert data_traversal["session_id"] == "legit-session-1"

        # 2. Arbitrary unregistered session ID is rejected and falls back
        data_unregistered = read_workspace_data(ws, target_session_id="unregistered-sess-99")
        assert data_unregistered["session_id"] == "legit-session-1"


def test_dashboard_no_wildcard_cors() -> None:
    """Dashboard API endpoints must NOT include wildcard Access-Control-Allow-Origin."""
    import socket
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from alienese.dashboard.server import DashboardRequestHandler

    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        DashboardRequestHandler.workspace = ws
        server = ThreadingHTTPServer(("127.0.0.1", port), DashboardRequestHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            for endpoint in ("/api/status", "/api/metrics", "/api/events", "/api/plan"):
                req = urllib.request.Request(f"http://127.0.0.1:{port}{endpoint}")
                with urllib.request.urlopen(req) as resp:
                    headers = dict(resp.headers)
                    assert "Access-Control-Allow-Origin" not in headers, (
                        f"Wildcard CORS header found on private localhost API: {endpoint}"
                    )
        finally:
            server.shutdown()
            server.server_close()


def test_count_tokens_raises_runtime_error_on_tokenizer_failure() -> None:
    """count_tokens must raise RuntimeError instead of silently substituting word counts."""
    from alienese.opencode.tracker import count_tokens

    with (
        patch("tiktoken.get_encoding", side_effect=Exception("Simulated tokenizer error")),
        pytest.raises(
            RuntimeError, match="Tokenizer failure: tiktoken cl100k_base unavailable"
        ),
    ):
        count_tokens("def foo(): pass")


def test_nemotron_status_credentials_configured() -> None:
    """Nemotron must report 'Credentials configured' when API key is present."""
    with patch.dict(os.environ, {"NEBIUS_TOKEN_FACTORY_KEY": "test-key-factory"}, clear=False):
        health = check_service_health()
        assert health["generator"]["status"] == "Credentials configured"
        assert "key present" in health["generator"]["detail"].lower()
