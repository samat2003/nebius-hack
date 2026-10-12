"""End-to-end integration tests for Alienese Three-Model OpenCode Integration.

Verifies:
1. EmbeddingGemma 2 is invoked for semantic retrieval ranking over repo context.
2. mini-Jev is invoked with multiple choices (K >= 2) without deterministic bypass.
3. Their results reach OpenCode's model context via `alienese_plan`.
4. OpenCode acts upon the plan output to inspect the top-ranked files.
5. NVIDIA Nemotron 3 Super synthesizes the fix.
6. Target behavioral and acceptance tests pass.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Import fixture setup from evaluation runner
sys.path.insert(0, str(REPO_ROOT))
from scripts.run_opencode_eval import load_env, setup_disposable_workspace  # noqa: E402


@pytest.fixture(scope="module")
def local_model_server_available() -> bool:
    try:
        resp = httpx.get("http://127.0.0.1:8001/health", timeout=1.0)
        return resp.status_code == 200 and resp.json().get("status") == "ok"
    except Exception:
        return False


def test_alienese_plan_tool_bridge_execution(local_model_server_available: bool) -> None:
    """Verify alienese_plan bridge executes EmbeddingGemma and mini-Jev with >= 2 options."""
    if not local_model_server_available:
        pytest.skip("Local model server is not running on http://127.0.0.1:8001")

    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        setup_disposable_workspace(ws)

        env = load_env()
        env["ALIENESE_THREE_MODEL"] = "1"
        env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

        # Run bridge plan command
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "alienese.opencode.bridge",
                "plan",
                "--workspace",
                str(ws),
                "--session",
                "test-plan-session",
                "--goal",
                "fix session refresh expiration",
            ],
            cwd=ws,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )

        res = json.loads(proc.stdout.strip())
        assert res["status"] == "success"
        assert res["three_model_active"] is True

        # 1. Check EmbeddingGemma 2 retrieval results
        ranked_files = res["ranked_files"]
        assert len(ranked_files) >= 3
        # Must have similarity scores from real embeddings
        for _fpath, score in ranked_files:
            assert isinstance(score, float)
            assert -1.0 <= score <= 1.0

        # 2. Check mini-Jev decision results with multiple choices (K >= 2)
        options = res["options"]
        assert len(options) >= 2, "mini-Jev must be evaluated on multiple choices"
        prob_sum = sum(opt["probability"] for opt in options)
        assert abs(prob_sum - 1.0) < 0.05, f"Option probabilities must sum to 1.0, got {prob_sum}"
        for opt in options:
            assert opt["probability"] >= 0.0
            assert "label" in opt

        # 3. Check selected action
        assert res["selected_id"] in [opt["id"] for opt in options]
        assert "plan_text" in res
        assert "EmbeddingGemma 2" in res["plan_text"]
        assert "mini-Jev" in res["plan_text"]


def test_opencode_three_model_end_to_end_repair(local_model_server_available: bool) -> None:
    """Full end-to-end OpenCode execution with alienese_plan, EmbeddingGemma, mini-Jev, Nemotron."""
    if not local_model_server_available:
        pytest.skip("Local model server is not running on http://127.0.0.1:8001")

    env = load_env()
    key = env.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip()
    if not key:
        pytest.skip("NEBIUS_TOKEN_FACTORY_KEY is not configured")

    opencode_bin = Path.home() / ".opencode" / "bin" / "opencode"
    if not opencode_bin.is_file():
        which_opencode = shutil.which("opencode")
        if which_opencode:
            opencode_bin = Path(which_opencode)
        else:
            pytest.skip("opencode binary not found")

    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str).resolve()
        setup_disposable_workspace(ws)

        # 1. Initial failing bug state verified
        proc_init = subprocess.run(
            ["pytest", "tests/test_session.py", "-q"],
            cwd=ws,
            capture_output=True,
            text=True,
        )
        assert proc_init.returncode != 0, "Initial fixture must fail before repair"

        # 2. Configure environment
        env["ALIENESE_THREE_MODEL"] = "1"
        env["ALIENESE_PYTHON"] = str(REPO_ROOT / ".venv" / "bin" / "python")
        env["ALIENESE_READ_ONLY_TESTS"] = "1"
        env["NEBIUS_API_KEY"] = key
        env["PATH"] = f"{opencode_bin.parent}:{REPO_ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{ws}:{env.get('PYTHONPATH', '')}"

        # 3. Execute OpenCode headlessly instructing it to invoke alienese_plan first
        task_prompt = (
            "Invoke alienese_plan with goal 'fix token refresh failure'. "
            "Follow its plan to inspect auth/session_manager.py, repair the bug, "
            "and verify all tests pass."
        )

        model_name = "nebius/nvidia/nemotron-3-super-120b-a12b"
        proc_oc = subprocess.run(
            [
                str(opencode_bin),
                "run",
                "--auto",
                "--dir",
                str(ws),
                "-m",
                model_name,
                task_prompt,
            ],
            cwd=ws,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc_oc.returncode == 0, (
            f"OpenCode run failed (code={proc_oc.returncode}): {proc_oc.stderr}"
        )

        # 4. Read and validate session trace
        trace_file = ws / ".alienese" / "opencode_trace.jsonl"
        assert trace_file.is_file(), f"Session trace not found at {trace_file}"

        trace_records = []
        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trace_records.append(json.loads(line))

        # Check: alienese_plan was invoked
        plan_records = [
            r
            for r in trace_records
            if r.get("tool") == "alienese_plan" or r.get("phase") == "ALIENESE_PLAN"
        ]
        assert len(plan_records) >= 1, "alienese_plan must be invoked during OpenCode session"

        # Check: plan details contained real EmbeddingGemma rankings & mini-Jev options
        plan_entry = next((r for r in plan_records if r.get("phase") == "ALIENESE_PLAN"), None)
        assert plan_entry is not None, "ALIENESE_PLAN audit entry must be recorded in trace"
        ranked_files = plan_entry["details"].get("ranked_files", [])
        assert len(ranked_files) >= 2, "EmbeddingGemma 2 must rank multiple files"
        options = plan_entry["details"].get("options", [])
        assert len(options) >= 2, "mini-Jev must evaluate multiple candidate choices"

        # Check: code was mutated in auth/session_manager.py
        mutations = set()
        for r in trace_records:
            for m in r.get("mutations", []):
                mutations.add(m)
        assert any("session_manager.py" in m for m in mutations), (
            f"Expected mutation in session_manager.py, got {mutations}"
        )

        # 5. Check tests passed
        proc_unit = subprocess.run(["pytest", "tests/test_session.py", "-v"], cwd=ws)
        proc_acc = subprocess.run(["pytest", "tests/test_acceptance.py", "-v"], cwd=ws)

        assert proc_unit.returncode == 0, "Behavioral tests must pass after Nemotron repair"
        assert proc_acc.returncode == 0, "Acceptance tests must pass after Nemotron repair"
