"""Alienese OpenCode Demo Environment Provisioner.

Prepares and verifies the isolated disposable challenge workspace
at /tmp/alienese_opencode_demo for browser demo recordings.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = Path("/tmp/alienese_opencode_demo")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the authentic cross-module integration challenge code fixtures
from scripts.run_opencode_eval import (  # noqa: E402
    SESSION_ACCEPTANCE_TEST_CODE,
    SESSION_MANAGER_BUGGY_CODE,
    SESSION_TEST_CODE,
    TOKEN_STORE_CODE,
)


def load_nebius_key() -> str:
    """Safely extract NEBIUS_TOKEN_FACTORY_KEY from repo .env."""
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if k.strip() == "NEBIUS_TOKEN_FACTORY_KEY":
                return v.strip().strip('"').strip("'")
    return ""


def clean_stale_opencode_sessions() -> None:
    """Prune stale sessions from opencode.db so the browser UI only shows the demo."""
    db_path = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    if not db_path.is_file():
        return
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("DELETE FROM session WHERE directory != ?;", (str(DEMO_DIR),))
        c.execute("DELETE FROM project_directory WHERE directory != ?;", (str(DEMO_DIR),))
        conn.commit()
        conn.close()
    except Exception:
        pass


def setup_demo_workspace(ws: Path, reset: bool = False) -> None:
    """Setup isolated disposable challenge in /tmp/alienese_opencode_demo."""
    if reset and ws.exists():
        shutil.rmtree(ws)

    ws.mkdir(parents=True, exist_ok=True)

    # Source code
    auth_dir = ws / "auth"
    auth_dir.mkdir(parents=True, exist_ok=True)
    (auth_dir / "__init__.py").write_text("", encoding="utf-8")
    (auth_dir / "token_store.py").write_text(TOKEN_STORE_CODE, encoding="utf-8")
    (auth_dir / "session_manager.py").write_text(SESSION_MANAGER_BUGGY_CODE, encoding="utf-8")

    # Tests
    tests_dir = ws / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_session.py").write_text(SESSION_TEST_CODE, encoding="utf-8")
    (tests_dir / "test_acceptance.py").write_text(SESSION_ACCEPTANCE_TEST_CODE, encoding="utf-8")

    # Clear previous trace
    trace_dir = ws / ".alienese"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)

    # Configure OpenCode with Nemotron default model and Alienese plugin
    plugin_js = REPO_ROOT / "plugins" / "opencode" / "alienese.js"
    opencode_cfg = {
        "$schema": "https://opencode.ai/config.json",
        "model": "nebius/nvidia/nemotron-3-super-120b-a12b",
        "plugin": [str(plugin_js.resolve())],
        "provider": {
            "nebius": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Nebius Token Factory",
                "options": {
                    "baseURL": "https://api.tokenfactory.us-central1.nebius.com/v1",
                    "apiKey": "{env:NEBIUS_TOKEN_FACTORY_KEY}",
                },
                "models": {
                    "nvidia/nemotron-3-super-120b-a12b": {
                        "name": "NVIDIA Nemotron 3 Super",
                        "tool_call": True,
                    }
                },
            }
        },
    }
    (ws / "opencode.json").write_text(json.dumps(opencode_cfg, indent=2), encoding="utf-8")

    clean_stale_opencode_sessions()


def check_demo_status(ws: Path) -> dict[str, Any]:
    """Check current state of demo workspace and test suites."""
    if not ws.exists():
        return {"exists": False, "state": "missing"}

    # Run pytest on test_session.py
    proc = subprocess.run(
        ["pytest", "tests/test_session.py", "-q"],
        cwd=ws,
        capture_output=True,
        text=True,
    )
    unit_failed = proc.returncode != 0

    proc_acc = subprocess.run(
        ["pytest", "tests/test_acceptance.py", "-q"],
        cwd=ws,
        capture_output=True,
        text=True,
    )
    acc_passed = proc_acc.returncode == 0

    trace_file = ws / ".alienese" / "opencode_trace.jsonl"
    trace_count = 0
    if trace_file.is_file():
        with open(trace_file, encoding="utf-8") as f:
            trace_count = sum(1 for line in f if line.strip())

    state = "ready_for_repair" if unit_failed else "repaired"
    server_pid = find_server_pid()
    nemotron_available = False
    if server_pid is not None:
        try:
            req = urllib.request.Request("http://127.0.0.1:4096/api/model", headers={"User-Agent": "Alienese"})
            with urllib.request.urlopen(req, timeout=1) as resp:
                data = json.loads(resp.read().decode())
                nemotron_available = any(
                    m.get("providerID") == "nebius" and "nemotron-3-super" in m.get("id", "")
                    for m in data.get("data", [])
                )
        except Exception:
            pass

    return {
        "exists": True,
        "workspace": str(ws),
        "state": state,
        "failing_bug_reproduced": unit_failed,
        "acceptance_tests_passed": acc_passed,
        "trace_events_count": trace_count,
        "server_running": server_pid is not None,
        "server_pid": server_pid,
        "nemotron_available": nemotron_available,
    }


def find_server_pid(port: int = 4096) -> int | None:
    """Find process ID listening on the given port."""
    try:
        proc = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
        pids = [int(p) for p in proc.stdout.strip().split() if p.isdigit()]
        return pids[0] if pids else None
    except Exception:
        return None


def stop_server(port: int = 4096) -> bool:
    """Stop server listening on port."""
    pid = find_server_pid(port)
    if pid is not None:
        with contextlib.suppress(OSError):
            os.kill(pid, 15)
            time.sleep(1)
            with contextlib.suppress(OSError):
                os.kill(pid, 9)
        return True
    return False


def start_server(ws: Path, port: int = 4096, host: str = "127.0.0.1") -> int:
    """Start OpenCode Web as a detached background daemon."""
    existing = find_server_pid(port)
    if existing is not None:
        return existing

    opencode_bin = Path.home() / ".opencode" / "bin" / "opencode"
    if not opencode_bin.is_file():
        which = shutil.which("opencode")
        if which:
            opencode_bin = Path(which)
        else:
            raise FileNotFoundError("opencode binary not found")

    env = os.environ.copy()
    key = load_nebius_key()
    if key:
        env["NEBIUS_TOKEN_FACTORY_KEY"] = key
        env["NEBIUS_API_KEY"] = key
    env["ALIENESE_PYTHON"] = str(REPO_ROOT / ".venv" / "bin" / "python")
    env["ALIENESE_READ_ONLY_TESTS"] = "1"
    env["PATH"] = f"{opencode_bin.parent}:{REPO_ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"

    log_path = ws / "opencode_web.log"
    with open(log_path, "a", encoding="utf-8") as log_file:
        p = subprocess.Popen(
            [str(opencode_bin), "web", "--port", str(port), "--hostname", host],
            cwd=str(ws),
            env=env,
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    (ws / "opencode_web.pid").write_text(str(p.pid), encoding="utf-8")

    # Wait up to 5 seconds for health
    url = f"http://{host}:{port}/"
    for _ in range(10):
        time.sleep(0.5)
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status in (200, 302):
                    break
        except Exception:
            pass

    # Ensure Nebius provider key is connected in OpenCode integration catalog
    if key:
        try:
            connect_url = f"http://{host}:{port}/api/integration/nebius/connect/key"
            connect_req = urllib.request.Request(
                connect_url,
                data=json.dumps({"key": key}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(connect_req, timeout=2):
                pass
        except Exception:
            pass

    return p.pid


def main() -> None:
    parser = argparse.ArgumentParser(description="Alienese OpenCode Demo Provisioner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # setup subcommand
    setup_p = subparsers.add_parser("setup", help="Setup or reset demo workspace")
    setup_p.add_argument("--reset", action="store_true", help="Force reset to buggy state")
    setup_p.add_argument("--workspace", default=str(DEMO_DIR), help="Workspace directory")

    # status subcommand
    stat_p = subparsers.add_parser("status", help="Check demo workspace status")
    stat_p.add_argument("--workspace", default=str(DEMO_DIR), help="Workspace directory")

    # start-server subcommand
    start_p = subparsers.add_parser("start-server", help="Start background OpenCode web server")
    start_p.add_argument("--workspace", default=str(DEMO_DIR), help="Workspace directory")
    start_p.add_argument("--port", type=int, default=4096, help="Port to listen on")
    start_p.add_argument("--host", default="127.0.0.1", help="Host to listen on")

    # stop-server subcommand
    stop_p = subparsers.add_parser("stop-server", help="Stop background OpenCode web server")
    stop_p.add_argument("--port", type=int, default=4096, help="Port to stop")

    # extract-key subcommand
    subparsers.add_parser("extract-key", help="Safely extract Nebius key")

    args = parser.parse_args()
    ws = Path(getattr(args, "workspace", str(DEMO_DIR)))

    if args.command == "setup":
        setup_demo_workspace(ws, reset=args.reset)
        stat = check_demo_status(ws)
        print(json.dumps(stat, indent=2))

    elif args.command == "status":
        stat = check_demo_status(ws)
        print(json.dumps(stat, indent=2))

    elif args.command == "start-server":
        pid = start_server(ws, port=args.port, host=args.host)
        print(
            json.dumps(
                {"server_started": True, "pid": pid, "url": f"http://{args.host}:{args.port}/"}
            )
        )

    elif args.command == "stop-server":
        stopped = stop_server(port=args.port)
        print(json.dumps({"server_stopped": stopped}))

    elif args.command == "extract-key":
        key = load_nebius_key()
        if not key:
            sys.exit(1)
        sys.stdout.write(key)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
