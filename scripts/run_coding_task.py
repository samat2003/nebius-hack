"""Autonomous coding agent runner for reproducible bug-fix workflows.

Demonstrates the failing-test -> inspect -> fix -> passing-test loop
under Alienese deterministic grounding and safety constraints.

Supports:
- Original token-expiry challenge (--task token)
- Integration session-manager challenge (--task integration)
- Offline simulation mode (SIMULATED / REPLAYED)
- Live Nebius Token Factory mode (--live)
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from alienese.api.models import ChatCompletionRequest, ChatMessageInput
from alienese.contracts.context import RequestContext
from alienese.contracts.generation import GenerationJob, GenerationJobType
from alienese.engine.turn import TurnEngine
from alienese.grounding.controller import GroundedRankController
from alienese.grounding.normalization import normalize_file_path
from alienese.providers.fake import FakeGenerator, FakeRetriever
from alienese.providers.generator.nebius_token_factory import NebiusTokenFactoryGenerator
from alienese.providers.runtime.client import ProviderHttpClient

# ---------------------------------------------------------------------------
# Task 1: Token Expiry Challenge (Original Known-Good Baseline)
# ---------------------------------------------------------------------------

TOKEN_BUGGY_MODULE = '''"""Token management module."""

import time

class TokenManager:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self.tokens: dict[str, float] = {}

    def issue_token(self, token_id: str) -> None:
        self.tokens[token_id] = time.time() - (self.ttl_seconds + 100)

    def is_valid(self, token_id: str) -> bool:
        created_at = self.tokens.get(token_id)
        if created_at is None:
            return False
        return (time.time() - created_at) < self.ttl_seconds
'''

TOKEN_TEST_MODULE = '''"""Unit tests for token management."""

from auth.token import TokenManager

def test_fresh_token_is_valid():
    manager = TokenManager(ttl_seconds=3600)
    manager.issue_token("session_abc123")
    assert manager.is_valid("session_abc123"), "Freshly issued token must be valid"
'''

TOKEN_ACCEPTANCE_TEST_MODULE = '''"""Independent acceptance tests for token management."""

import time
from auth.token import TokenManager

def test_token_expiration_behavior():
    manager = TokenManager(ttl_seconds=10)
    manager.issue_token("active_token")
    assert manager.is_valid("active_token"), "Active token must be valid"
    assert not manager.is_valid("non_existent"), "Missing token must be invalid"
'''

OFFLINE_SIMULATED_TOKEN_REPAIR = '''```python
"""Token management module."""

import time

class TokenManager:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self.tokens: dict[str, float] = {}

    def issue_token(self, token_id: str) -> None:
        self.tokens[token_id] = time.time()

    def is_valid(self, token_id: str) -> bool:
        created_at = self.tokens.get(token_id)
        if created_at is None:
            return False
        return (time.time() - created_at) < self.ttl_seconds
```'''


# ---------------------------------------------------------------------------
# Task 2: Cross-Module Session Management & Token Store Integration Challenge
# ---------------------------------------------------------------------------

SESSION_TOKEN_STORE_MODULE = '''"""Token store managing cryptographic tokens and revocation state."""

import time
from typing import Any


class TokenStore:
    def __init__(self, default_ttl: int = 3600) -> None:
        self.default_ttl = default_ttl
        self._tokens: dict[str, dict[str, Any]] = {}
        self._revoked: set[str] = set()

    def store_token(
        self,
        token_id: str,
        user_id: str,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        self._tokens[token_id] = {
            "user_id": user_id,
            "issued_at": time.time(),
            "expires_at": time.time() + ttl,
            "metadata": metadata or {},
        }

    def revoke_token(self, token_id: str) -> None:
        self._revoked.add(token_id)
        self._tokens.pop(token_id, None)

    def is_revoked(self, token_id: str) -> bool:
        return token_id in self._revoked

    def get_token(self, token_id: str) -> dict[str, Any] | None:
        if token_id in self._revoked:
            return None
        data = self._tokens.get(token_id)
        if data is None:
            return None
        if time.time() > data["expires_at"]:
            return None
        return data
'''

SESSION_MANAGER_BUGGY_MODULE = '''"""Session manager coordinating session lifecycles with the token store."""

from typing import Any
from auth.token_store import TokenStore


class SessionManager:
    def __init__(self, token_store: TokenStore, session_ttl: int = 1800) -> None:
        self.token_store = token_store
        self.session_ttl = session_ttl
        self._sessions: dict[str, dict[str, Any]] = {}

    def create_session(self, session_id: str, user_id: str, token_id: str) -> dict[str, Any]:
        self.token_store.store_token(
            token_id=token_id,
            user_id=user_id,
            ttl_seconds=self.session_ttl,
        )
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "active_token_id": token_id,
            "is_active": True,
        }
        self._sessions[session_id] = session_data
        return session_data

    def validate_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if not session or not session.get("is_active"):
            return False
        token_id = session.get("active_token_id")
        if not token_id:
            return False
        token = self.token_store.get_token(token_id)
        return token is not None

    def refresh_session(self, session_id: str, new_token_id: str) -> dict[str, Any] | None:
        """Refresh session with a new token and invalidate old token."""
        session = self._sessions.get(session_id)
        if not session or not session.get("is_active"):
            return None

        old_token_id = session.get("active_token_id")
        if not old_token_id:
            return None

        old_token = self.token_store.get_token(old_token_id)
        if old_token is None:
            return None

        self.token_store.store_token(
            token_id=new_token_id,
            user_id=session["user_id"],
            ttl_seconds=self.session_ttl,
        )
        self.token_store.revoke_token(old_token_id)
        return session

    def terminate_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session["is_active"] = False
            token_id = session.get("active_token_id")
            if token_id:
                self.token_store.revoke_token(token_id)
'''

SESSION_TEST_MODULE = '''"""Unit and regression tests for authentication sessions."""

from auth.token_store import TokenStore
from auth.session_manager import SessionManager


def test_session_create_and_validate():
    store = TokenStore()
    manager = SessionManager(store)
    manager.create_session("sess_101", "user_alice", "tok_init")
    assert manager.validate_session("sess_101"), "New session must be valid"


def test_session_termination():
    store = TokenStore()
    manager = SessionManager(store)
    manager.create_session("sess_102", "user_bob", "tok_init")
    manager.terminate_session("sess_102")
    assert not manager.validate_session("sess_102"), "Terminated session must be invalid"


def test_session_refresh_lifecycle():
    store = TokenStore()
    manager = SessionManager(store)
    manager.create_session("sess_103", "user_charlie", "tok_old")
    assert manager.validate_session("sess_103")

    refreshed = manager.refresh_session("sess_103", "tok_new")
    assert refreshed is not None, "Refresh session must succeed"
    assert manager.validate_session("sess_103"), "Refreshed session must remain valid with new token"
'''

SESSION_ACCEPTANCE_TEST_MODULE = '''"""Independent acceptance tests for session lifecycle and multi-token continuity."""

from auth.token_store import TokenStore
from auth.session_manager import SessionManager


def test_session_multi_refresh_continuity():
    store = TokenStore()
    manager = SessionManager(store)
    manager.create_session("sess_acc_1", "user_acc", "tok_v1")

    # Sequential refreshes
    r1 = manager.refresh_session("sess_acc_1", "tok_v2")
    assert r1 is not None
    assert manager.validate_session("sess_acc_1"), "Session must remain valid after first refresh"

    r2 = manager.refresh_session("sess_acc_1", "tok_v3")
    assert r2 is not None
    assert manager.validate_session("sess_acc_1"), "Session must remain valid after second refresh"

    # Verify old tokens revoked in token store
    assert store.get_token("tok_v1") is None, "Old token v1 must be revoked"
    assert store.get_token("tok_v2") is None, "Old token v2 must be revoked"
    assert store.get_token("tok_v3") is not None, "Current active token v3 must exist"


def test_unregistered_session_refresh_rejected():
    store = TokenStore()
    manager = SessionManager(store)
    res = manager.refresh_session("non_existent_sess", "tok_any")
    assert res is None, "Refreshing nonexistent session must return None"
'''

OFFLINE_SIMULATED_SESSION_REPAIR = '''```python
"""Session manager coordinating session lifecycles with the token store."""

from typing import Any
from auth.token_store import TokenStore


class SessionManager:
    def __init__(self, token_store: TokenStore, session_ttl: int = 1800) -> None:
        self.token_store = token_store
        self.session_ttl = session_ttl
        self._sessions: dict[str, dict[str, Any]] = {}

    def create_session(self, session_id: str, user_id: str, token_id: str) -> dict[str, Any]:
        self.token_store.store_token(
            token_id=token_id,
            user_id=user_id,
            ttl_seconds=self.session_ttl,
        )
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "active_token_id": token_id,
            "is_active": True,
        }
        self._sessions[session_id] = session_data
        return session_data

    def validate_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if not session or not session.get("is_active"):
            return False
        token_id = session.get("active_token_id")
        if not token_id:
            return False
        token = self.token_store.get_token(token_id)
        return token is not None

    def refresh_session(self, session_id: str, new_token_id: str) -> dict[str, Any] | None:
        """Refresh session with a new token and invalidate old token."""
        session = self._sessions.get(session_id)
        if not session or not session.get("is_active"):
            return None

        old_token_id = session.get("active_token_id")
        if not old_token_id:
            return None

        old_token = self.token_store.get_token(old_token_id)
        if old_token is None:
            return None

        self.token_store.store_token(
            token_id=new_token_id,
            user_id=session["user_id"],
            ttl_seconds=self.session_ttl,
        )
        self.token_store.revoke_token(old_token_id)
        session["active_token_id"] = new_token_id
        return session

    def terminate_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session["is_active"] = False
            token_id = session.get("active_token_id")
            if token_id:
                self.token_store.revoke_token(token_id)
```'''


# ---------------------------------------------------------------------------
# Task Definition Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    target_test: str
    acceptance_test: str
    target_module: str
    writable_allowlist: frozenset[str]
    setup_files: dict[str, str]
    offline_simulated_repair: str
    original_target_content: str


TASKS: dict[str, TaskSpec] = {
    "token": TaskSpec(
        name="token",
        description="Single-module token TTL past-offset expiration bug",
        target_test="tests/test_token.py",
        acceptance_test="tests/test_acceptance.py",
        target_module="auth/token.py",
        writable_allowlist=frozenset({"auth/token.py"}),
        setup_files={
            "auth/__init__.py": "",
            "auth/token.py": TOKEN_BUGGY_MODULE,
            "tests/__init__.py": "",
            "tests/test_token.py": TOKEN_TEST_MODULE,
            "tests/test_acceptance.py": TOKEN_ACCEPTANCE_TEST_MODULE,
        },
        offline_simulated_repair=OFFLINE_SIMULATED_TOKEN_REPAIR,
        original_target_content=TOKEN_BUGGY_MODULE,
    ),
    "integration": TaskSpec(
        name="integration",
        description="Cross-module session manager and token store refresh regression",
        target_test="tests/test_session.py",
        acceptance_test="tests/test_acceptance.py",
        target_module="auth/session_manager.py",
        writable_allowlist=frozenset({"auth/session_manager.py", "auth/token_store.py"}),
        setup_files={
            "auth/__init__.py": "",
            "auth/token_store.py": SESSION_TOKEN_STORE_MODULE,
            "auth/session_manager.py": SESSION_MANAGER_BUGGY_MODULE,
            "tests/__init__.py": "",
            "tests/test_session.py": SESSION_TEST_MODULE,
            "tests/test_acceptance.py": SESSION_ACCEPTANCE_TEST_MODULE,
        },
        offline_simulated_repair=OFFLINE_SIMULATED_SESSION_REPAIR,
        original_target_content=SESSION_MANAGER_BUGGY_MODULE,
    ),
}
# Alias 'session' -> 'integration'
TASKS["session"] = TASKS["integration"]


# ---------------------------------------------------------------------------
# Sandboxed Coding Harness
# ---------------------------------------------------------------------------

class StandaloneCodingHarness:
    """Bounded, sandboxed execution harness for coding workflows.

    Enforces:
    - Strict path resolution and containment via Path.is_relative_to().
    - Write access limited to an explicit mutable source allowlist.
    - Tests, test files, and configs are strictly read-only and immutable.
    - Subprocess timeouts and output truncation.
    - Secret redaction in subprocess environments.
    - Rejection of unsupported tool arguments.
    """

    MAX_TOOL_OUTPUT_CHARS = 10_000
    SUBPROCESS_TIMEOUT_SECONDS = 15.0

    def __init__(
        self,
        workspace: Path,
        *,
        writable_allowlist: frozenset[str] = frozenset({"auth/token.py"}),
    ) -> None:
        self.workspace = workspace.resolve()
        self.writable_allowlist = writable_allowlist
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents from repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Repository-relative file path",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write updated code to repository file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Repository-relative file path",
                            },
                            "content": {"type": "string", "description": "New file content"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_test",
                    "description": "Run pytest on a specific test target",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string", "description": "Test file or node ID"}
                        },
                        "required": ["target"],
                    },
                },
            },
        ]

    def setup_task_repo(self, task: TaskSpec) -> None:
        """Create sample codebase with authentic bug, unit test, and acceptance check."""
        for rel_path, content in task.setup_files.items():
            dest = self.workspace / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

    def setup_demo_repo(self) -> None:
        """Backward-compatible setup for original token task demo repo."""
        self.setup_task_repo(TASKS["token"])

    def _resolve_and_contain(self, rel_path: str) -> tuple[bool, Path | str]:
        """Strictly resolve and verify that rel_path is contained in workspace."""
        if not rel_path or not isinstance(rel_path, str):
            return False, "Invalid path argument"
        normalized = normalize_file_path(rel_path)
        if normalized is None:
            return False, f"Permission denied: path '{rel_path}' is invalid or attempts traversal"
        target = (self.workspace / normalized).resolve()
        if not target.is_relative_to(self.workspace):
            return False, f"Permission denied: path '{rel_path}' escapes workspace"
        return True, target

    def execute_tool(self, name: str, args: dict[str, Any]) -> tuple[bool, str]:
        """Execute tool locally in sandbox with strict security controls."""
        if not isinstance(args, dict):
            return False, "Invalid tool arguments: must be a JSON object"

        if name == "read_file":
            allowed_keys = {"path"}
            if set(args.keys()) - allowed_keys:
                return False, f"Unsupported arguments for read_file: {set(args.keys()) - allowed_keys}"
            rel_path = args.get("path", "")
            ok, target_or_err = self._resolve_and_contain(rel_path)
            if not ok or isinstance(target_or_err, str):
                return False, str(target_or_err)
            target = target_or_err
            if not target.exists() or not target.is_file():
                return False, f"File not found: '{rel_path}'"
            content = target.read_text(encoding="utf-8")
            if len(content) > self.MAX_TOOL_OUTPUT_CHARS:
                content = content[: self.MAX_TOOL_OUTPUT_CHARS] + "\n...[truncated]"
            return True, content

        elif name == "write_file":
            allowed_keys = {"path", "content"}
            if set(args.keys()) - allowed_keys:
                return False, f"Unsupported arguments for write_file: {set(args.keys()) - allowed_keys}"
            rel_path = args.get("path", "")
            content = args.get("content", "")
            if not isinstance(content, str):
                return False, "content must be a string"
            if len(content) > 100_000:
                return False, "content exceeds maximum allowable size (100KB)"

            normalized = normalize_file_path(rel_path)
            if normalized not in self.writable_allowlist:
                return (
                    False,
                    f"Permission denied: write to '{rel_path}' is not permitted by mutable allowlist",
                )

            ok, target_or_err = self._resolve_and_contain(rel_path)
            if not ok or isinstance(target_or_err, str):
                return False, str(target_or_err)
            target = target_or_err

            # Strictly prevent writing to test directories or configuration
            if target.is_relative_to(self.workspace / "tests"):
                return False, "Permission denied: modifying tests is forbidden"

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return True, f"Successfully wrote {len(content)} bytes to '{rel_path}'"

        elif name == "run_test":
            allowed_keys = {"target"}
            if set(args.keys()) - allowed_keys:
                return False, f"Unsupported arguments for run_test: {set(args.keys()) - allowed_keys}"
            target_rel = args.get("target", "")
            ok, target_or_err = self._resolve_and_contain(target_rel)
            if not ok or isinstance(target_or_err, str):
                return False, str(target_or_err)
            target_path = target_or_err
            if not target_path.exists():
                return False, f"Test target not found: '{target_rel}'"

            # Clean environment: strip API keys or secrets before spawning subprocess
            clean_env = {
                k: v
                for k, v in os.environ.items()
                if not any(
                    term in k.upper()
                    for term in ("KEY", "SECRET", "TOKEN", "PASSWORD", "AUTH", "CREDENTIAL")
                )
            }
            clean_env["PYTHONPATH"] = str(self.workspace)

            cmd = [sys.executable, "-m", "pytest", str(target_path), "-v"]
            try:
                res = subprocess.run(
                    cmd,
                    cwd=str(self.workspace),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.SUBPROCESS_TIMEOUT_SECONDS,
                    env=clean_env,
                )
                output = res.stdout
                if len(output) > self.MAX_TOOL_OUTPUT_CHARS:
                    output = output[: self.MAX_TOOL_OUTPUT_CHARS] + "\n...[truncated]"
                return (res.returncode == 0), output
            except subprocess.TimeoutExpired:
                return False, f"Test execution timed out after {self.SUBPROCESS_TIMEOUT_SECONDS}s"

        return False, f"Unknown or forbidden tool: {name}"


# ---------------------------------------------------------------------------
# Model Parsing and Unified Diff Utilities
# ---------------------------------------------------------------------------

def parse_model_code_repair(
    raw_content: str,
    target_module: str = "auth/token.py",
) -> tuple[bool, str, str | None]:
    """Extract and validate clean Python source code from model response.

    Returns:
        (is_valid, extracted_code, error_message)
    """
    if not raw_content or not raw_content.strip():
        return False, "", "Empty model response"

    code: str = ""
    # 1. Try markdown code block with python/py tag
    py_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)\n```", raw_content, re.DOTALL)
    if py_blocks:
        code = py_blocks[0].strip()
    else:
        # 2. Try JSON markdown or raw JSON
        json_blocks = re.findall(r"```json\s*\n(.*?)\n```", raw_content, re.DOTALL)
        candidate_json = json_blocks[0] if json_blocks else raw_content
        try:
            parsed = json.loads(candidate_json.strip())
            if isinstance(parsed, dict) and "replacement_content" in parsed:
                code = str(parsed["replacement_content"]).strip()
            elif isinstance(parsed, dict) and "content" in parsed:
                code = str(parsed["content"]).strip()
        except Exception:
            pass

    # 3. If no fence matched and it looks like raw python source
    if not code and ("class " in raw_content or "def " in raw_content):
        clean = raw_content.strip()
        if clean.startswith("```") and clean.endswith("```"):
            clean = "\n".join(clean.splitlines()[1:-1]).strip()
        code = clean

    if not code:
        return False, "", "Could not extract valid Python source code from model response"

    # Validate Python syntax via compile()
    try:
        compile(code, target_module, "exec")
    except SyntaxError as err:
        return False, "", f"Model-generated code failed syntax compilation: {err}"

    # Bounded size check
    if len(code) > 50_000:
        return False, "", "Model-generated code exceeds maximum permissible size (50KB)"

    return True, code, None


def compute_unified_diff(original: str, modified: str, filename: str) -> str:
    """Compute standard unified diff for transparent audit."""
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines,
        mod_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    return "".join(diff)


# ---------------------------------------------------------------------------
# End-to-End Workflow Execution
# ---------------------------------------------------------------------------

async def run_failing_to_passing_workflow(
    *,
    task_name: str = "token",
    use_live_nebius: bool = False,
    output_log_path: Path | None = None,
    keep_workspace: bool = False,
    max_repair_attempts: int = 3,
) -> dict[str, Any]:
    """Execute end-to-end bug fix workflow with genuine model code repair."""
    task = TASKS.get(task_name)
    if task is None:
        raise ValueError(f"Unknown task: '{task_name}'. Available: {list(TASKS.keys())}")

    load_dotenv()
    temp_dir = Path(tempfile.mkdtemp(prefix=f"alienese_{task.name}_"))
    harness = StandaloneCodingHarness(temp_dir, writable_allowlist=task.writable_allowlist)
    harness.setup_task_repo(task)

    steps_log: list[dict[str, Any]] = []
    model_telemetry: dict[str, Any] = {}
    applied_diff: str = ""
    start_time = time.time()

    mode_label = (
        "LIVE NEBIUS TOKEN FACTORY (nvidia/nemotron-3-super-120b-a12b)"
        if use_live_nebius
        else "OFFLINE SIMULATION (REPLAYED - NOT LIVE INFERENCE)"
    )

    print("\n" + "=" * 70)
    print(" ALIENESE AUTONOMOUS CODING WORKFLOW")
    print(f" Task: {task.name.upper()} — {task.description}")
    print(f" Mode: {mode_label}")
    print(f" Workspace: {temp_dir}")
    print("=" * 70)

    # Step 1: Initial failing test execution
    print(f"\n[Step 1] Initial Failing Test Suite (Harness Execution): {task.target_test}...")
    ok, test_out = harness.execute_tool("run_test", {"target": task.target_test})
    print(f"Result: {'PASSED' if ok else 'FAILED (Expected initial bug)'}")
    print("-" * 50)
    for line in test_out.strip().splitlines()[-6:]:
        print(f"  {line}")
    print("-" * 50)
    steps_log.append(
        {
            "step": 1,
            "step_type": "harness_verification",
            "action": "run_test",
            "target": task.target_test,
            "passed": ok,
            "output_summary": test_out[-300:],
        }
    )
    assert not ok, f"Initial test {task.target_test} must fail to demonstrate authentic repair"

    # Step 2: Initialize Alienese Engine with Nebius Generator and GroundedRankController
    gen: Any
    http_client: Any = None
    if use_live_nebius:
        key = os.getenv("NEBIUS_TOKEN_FACTORY_KEY") or ""
        base_url = os.getenv(
            "NEBIUS_TOKEN_FACTORY_BASE_URL",
            "https://api.tokenfactory.us-central1.nebius.com/v1",
        )
        http_client = ProviderHttpClient(
            provider_name="nebius_token_factory",
            base_url=base_url,
            api_key=key,
        )
        gen = NebiusTokenFactoryGenerator(http_client=http_client)
    else:
        gen = FakeGenerator()

    retriever = FakeRetriever()
    controller = GroundedRankController()
    engine = TurnEngine(
        retriever=retriever,
        controller=controller,
        generator=gen,
        include_replay_content=True,
    )

    try:
        # Step 3: Turn 1 - Ground failure traceback deterministically via Alienese
        print(f"\n[Step 2] Turn 1: Deterministic Grounding (Alienese Decision Engine)...")
        ctx1 = RequestContext(request_id="run-turn-1", operation_id="op-turn-1")
        req1 = ChatCompletionRequest(
            model="alienese-default",
            tool_choice="required",
            messages=[
                ChatMessageInput(
                    role="user",
                    content=f"Fix the failing test in repository:\n{test_out}",
                )
            ],
            tools=harness.tools_schema,  # type: ignore[arg-type]
        )
        resp1, artifact1 = await engine.execute_turn(ctx1, req1)
        tool_calls = resp1.choices[0].message.tool_calls or []
        assert tool_calls, "Expected at least one tool call in Turn 1"
        tool_call_1 = tool_calls[0]
        tool_name_1 = tool_call_1.function.name
        tool_args_1 = json.loads(tool_call_1.function.arguments)
        print(f"Alienese selected action: {tool_name_1}({tool_args_1})")
        print(f"Grounded candidates count: {artifact1.semantics.candidate_count}")
        print(f"Replay verified: {artifact1.semantics.replayable}")

        ok1, inspected_test_content = harness.execute_tool(tool_name_1, tool_args_1)
        if not ok1:
            raise RuntimeError(f"Grounded tool call {tool_name_1}({tool_args_1}) failed: {inspected_test_content}")
        print(f"Tool executed: read {len(inspected_test_content)} bytes from {tool_args_1.get('path')}")
        steps_log.append(
            {
                "step": 2,
                "step_type": "alienese_decision",
                "turn": 1,
                "action": tool_name_1,
                "tool": tool_name_1,
                "args": tool_args_1,
                "success": ok1,
                "candidate_count": artifact1.semantics.candidate_count,
                "state_digest": artifact1.semantics.state_digest,
            }
        )

        # Step 4: Discover and read relevant source modules based on test imports
        print(f"\n[Step 2b] Harness Context Inspection: Reading discovered module dependencies...")
        # Inspect imports in test module
        import_modules = re.findall(r"(?:from|import)\s+([a-zA-Z0-9_\.]+)", inspected_test_content)
        candidate_paths = [m.replace(".", "/") + ".py" for m in import_modules]
        # Always include task's mutable target module
        if task.target_module not in candidate_paths:
            candidate_paths.append(task.target_module)

        source_files_content: dict[str, str] = {}
        for c_path in candidate_paths:
            if (harness.workspace / c_path).exists():
                ok_s, c_content = harness.execute_tool("read_file", {"path": c_path})
                if ok_s:
                    source_files_content[c_path] = c_content
                    print(f"  Discovered & read: {c_path} ({len(c_content)} bytes)")
                    steps_log.append(
                        {
                            "step": 2.5,
                            "step_type": "harness_inspection",
                            "action": "read_source",
                            "path": c_path,
                            "success": True,
                            "bytes_read": len(c_content),
                        }
                    )

        target_source = source_files_content.get(task.target_module, "")
        if not target_source:
            raise RuntimeError(f"Target module '{task.target_module}' could not be read from workspace")

        # Step 5: Turn 2 - Genuine Model Code Generation (with bounded repair retry)
        print(f"\n[Step 3] Turn 2: Requesting genuine code repair from Nemotron 3 Super...")

        # Construct prompt passing all discovered context clearly labeled
        prompt_parts = [
            "You are an expert Python software engineer.",
            f"A test failed in the test suite:\n{test_out}\n",
            f"Here is the content of the failing test file '{tool_args_1.get('path')}':\n{inspected_test_content}\n",
        ]
        for src_path, src_code in source_files_content.items():
            prompt_parts.append(f"Here is the content of the module '{src_path}':\n{src_code}\n")

        prompt_parts.append(
            f"Fix the bug in '{task.target_module}' so that all tests pass.\n"
            f"Output ONLY the complete, corrected Python code for '{task.target_module}' inside a ```python``` code block.\n"
            "Do not include any explanation before or after."
        )
        repair_prompt = "\n".join(prompt_parts)

        repaired_code = ""
        attempt = 0
        repair_success = False

        while attempt < max_repair_attempts and not repair_success:
            attempt += 1
            if use_live_nebius:
                print(f"Invoking Nebius Token Factory (Attempt {attempt}/{max_repair_attempts})...")
                ctx_gen = RequestContext(
                    request_id=f"run-turn-gen-{attempt}", operation_id=f"op-turn-gen-{attempt}"
                )
                job_gen = GenerationJob(
                    job_id=f"job_repair_{task.name}_{attempt}",
                    job_type=GenerationJobType.ANSWER,
                    candidate_id="cand_answer",
                    latest_user_request=repair_prompt,
                    max_tokens=2048,
                    temperature=0.0,
                )
                res_gen = await gen.generate(ctx_gen, job_gen)
                raw_repair_output = res_gen.semantics.content
                model_telemetry = {
                    "latency_ms": res_gen.telemetry.latency_ms,
                    "prompt_tokens": res_gen.telemetry.prompt_tokens,
                    "completion_tokens": res_gen.telemetry.completion_tokens,
                    "total_tokens": res_gen.telemetry.total_tokens,
                    "upstream_request_id": res_gen.telemetry.upstream_request_id,
                    "serving_fingerprint": res_gen.telemetry.serving_fingerprint,
                    "attempts": attempt,
                }
                print(f"Nemotron generation latency: {res_gen.telemetry.latency_ms:.1f}ms")
                print(f"Tokens: {res_gen.telemetry.total_tokens} total ({res_gen.telemetry.completion_tokens} completion)")
            else:
                raw_repair_output = task.offline_simulated_repair
                model_telemetry = {
                    "simulated": True,
                    "note": "SIMULATED / REPLAYED - No live model inference was performed.",
                    "attempts": 1,
                }
                print("Using offline simulated model repair (deterministic verification mode)")

            # Parse and validate code
            parse_ok, parsed_code, parse_err = parse_model_code_repair(raw_repair_output, task.target_module)
            if not parse_ok:
                print(f"Attempt {attempt} parse error: {parse_err}")
                if attempt >= max_repair_attempts:
                    raise RuntimeError(f"Model repair rejected after {attempt} attempts: {parse_err}")
                continue

            repaired_code = parsed_code
            applied_diff = compute_unified_diff(task.original_target_content, repaired_code, task.target_module)

            # Apply patch to workspace
            ok_patch, patch_msg = harness.execute_tool(
                "write_file",
                {"path": task.target_module, "content": repaired_code},
            )
            print(f"Patch applied to {task.target_module}: {patch_msg}")
            steps_log.append(
                {
                    "step": 3,
                    "step_type": "model_generation",
                    "turn": 2,
                    "attempt": attempt,
                    "action": "write_file",
                    "path": task.target_module,
                    "success": ok_patch,
                    "model_telemetry": model_telemetry,
                    "diff": applied_diff,
                }
            )

            # Run target verification test
            ok_final, test_final_out = harness.execute_tool(
                "run_test", {"target": task.target_test}
            )
            if ok_final:
                repair_success = True
                break
            else:
                print(f"Attempt {attempt} did not pass target test. Retrying...")

        print("\nModel-Generated Unified Diff:")
        print("-" * 50)
        for line in applied_diff.splitlines():
            print(f"  {line}")
        print("-" * 50)

        # Step 6: Target Test Verification
        print(f"\n[Step 4] Turn 3: Target Test Suite Result ({task.target_test}):")
        print(f"Status: {'PASSED (Fix Verified)' if ok_final else 'FAILED'}")
        print("-" * 50)
        for line in test_final_out.strip().splitlines()[-4:]:
            print(f"  {line}")
        print("-" * 50)
        steps_log.append(
            {
                "step": 4,
                "step_type": "harness_verification",
                "turn": 3,
                "action": "run_test",
                "target": task.target_test,
                "passed": ok_final,
                "output_summary": test_final_out[-250:],
            }
        )

        # Step 7: Independent Acceptance Check (Unseen by model)
        print(f"\n[Step 5] Harness Acceptance Check: Running independent acceptance test ({task.acceptance_test})...")
        ok_acceptance, test_acc_out = harness.execute_tool(
            "run_test", {"target": task.acceptance_test}
        )
        print(f"Acceptance Test Result: {'PASSED' if ok_acceptance else 'FAILED'}")
        print("-" * 50)
        for line in test_acc_out.strip().splitlines()[-4:]:
            print(f"  {line}")
        print("-" * 50)
        steps_log.append(
            {
                "step": 5,
                "step_type": "harness_acceptance",
                "action": "run_acceptance_test",
                "target": task.acceptance_test,
                "passed": ok_acceptance,
                "output_summary": test_acc_out[-250:],
            }
        )

        # Step 8: Post-Mortem Explanation (Derived strictly from verified diff)
        print(f"\n[Step 6] Turn 4: Asking Nebius Token Factory for repair explanation based on diff...")
        explanation = ""
        if use_live_nebius:
            ctx_sum = RequestContext(request_id="run-turn-sum", operation_id="op-turn-sum")
            job_sum = GenerationJob(
                job_id=f"job_explain_{task.name}",
                job_type=GenerationJobType.ANSWER,
                candidate_id="cand_answer",
                trusted_system_instructions=(
                    "You are Alienese coding agent. Summarize the bug and resolution in 2 concise sentences based strictly on the verified diff and failing test.",
                ),
                latest_user_request=(
                    f"A regression in {task.target_module} was fixed in response to failing test {task.target_test}.\n\n"
                    f"Here is the verified unified diff applied to {task.target_module}:\n"
                    f"{applied_diff}\n\n"
                    f"Summarize the exact bug and resolution in 2 concise sentences based strictly on this diff."
                ),
                max_tokens=128,
                temperature=0.0,
            )
            res_sum = await gen.generate(ctx_sum, job_sum)
            explanation = res_sum.semantics.content.strip()
        else:
            if task.name in ("integration", "session"):
                explanation = "Fixed SessionManager.refresh_session() to update session['active_token_id'] with the new token ID after revoking the previous token."
            else:
                explanation = "Fixed TokenManager.issue_token() to assign the current timestamp rather than a past offset."
        print(f'Nebius Nemotron Explanation:\n  "{explanation}"')
        steps_log.append(
            {
                "step": 6,
                "step_type": "model_explanation",
                "turn": 4,
                "action": "explain_repair",
                "explanation": explanation,
            }
        )

    finally:
        if http_client is not None:
            await http_client.aclose()
        if keep_workspace:
            print(f"\n[Notice] Workspace preserved for inspection: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    overall_status = "SUCCESS" if (ok_final and ok_acceptance) else "FAILED"

    summary = {
        "status": overall_status,
        "task": task.name,
        "description": task.description,
        "execution_mode": "LIVE_NEBIUS_TOKEN_FACTORY" if use_live_nebius else "OFFLINE_SIMULATION",
        "is_live_inference": use_live_nebius,
        "workflow_architecture": "Evidence-grounded inspection with harness-constrained repair (Alienese deterministically grounds failure traceback to test file; harness inspects imported modules and constrains model write to designated target module).",
        "simulation_notice": (
            None
            if use_live_nebius
            else "SIMULATED / REPLAYED - No live model inference was performed."
        ),
        "elapsed_ms": elapsed_ms,
        "model_telemetry": model_telemetry,
        "applied_diff": applied_diff,
        "steps": steps_log,
        "explanation": explanation,
    }

    if output_log_path:
        output_log_path.parent.mkdir(parents=True, exist_ok=True)
        output_log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nExecution log written to: {output_log_path}")

    print("\n" + "=" * 70)
    print(f" WORKFLOW {overall_status} IN {elapsed_ms}ms (Task: {task.name.upper()})")
    print("=" * 70 + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Alienese Autonomous Coding Runner")
    parser.add_argument(
        "--task",
        type=str,
        default="integration",
        choices=["integration", "session", "token"],
        help="Task challenge to execute (default: integration)",
    )
    parser.add_argument("--live", action="store_true", help="Use live Nebius Token Factory")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON log path (default: results/live_run_report.json or results/judge_offline.json)",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep disposable temporary workspace on disk for inspection",
    )
    args = parser.parse_args()

    default_out = (
        Path("results/live_run_report.json")
        if args.live
        else Path(f"results/judge_{args.task}_offline.json")
    )
    out_path = args.out or default_out

    asyncio.run(
        run_failing_to_passing_workflow(
            task_name=args.task,
            use_live_nebius=args.live,
            output_log_path=out_path,
            keep_workspace=args.keep_workspace,
        )
    )


if __name__ == "__main__":
    main()
