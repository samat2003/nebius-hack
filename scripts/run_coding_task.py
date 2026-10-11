"""Autonomous coding agent runner for reproducible bug-fix workflows.

Demonstrates the failing-test -> inspect -> fix -> passing-test loop
under Alienese deterministic grounding and safety constraints.
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
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from alienese.api.models import ChatCompletionRequest
from alienese.contracts.context import RequestContext
from alienese.contracts.generation import GenerationJob, GenerationJobType
from alienese.engine.turn import TurnEngine
from alienese.grounding.controller import GroundedRankController
from alienese.grounding.normalization import normalize_file_path
from alienese.providers.fake import FakeGenerator, FakeRetriever
from alienese.providers.generator.nebius_token_factory import NebiusTokenFactoryGenerator
from alienese.providers.runtime.client import ProviderHttpClient

DEMO_BUGGY_MODULE = '''"""Token management module."""

import time

class TokenManager:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self.tokens: dict[str, float] = {}

    def issue_token(self, token_id: str) -> None:
        # BUG: Stored token timestamp with past offset instead of current timestamp
        self.tokens[token_id] = time.time() - (self.ttl_seconds + 100)

    def is_valid(self, token_id: str) -> bool:
        created_at = self.tokens.get(token_id)
        if created_at is None:
            return False
        return (time.time() - created_at) < self.ttl_seconds
'''

DEMO_TEST_MODULE = '''"""Unit tests for token management."""

from auth.token import TokenManager

def test_fresh_token_is_valid():
    manager = TokenManager(ttl_seconds=3600)
    manager.issue_token("session_abc123")
    assert manager.is_valid("session_abc123"), "Freshly issued token must be valid"
'''

DEMO_ACCEPTANCE_TEST_MODULE = '''"""Independent acceptance tests for token management."""

import time
from auth.token import TokenManager

def test_token_expiration_behavior():
    manager = TokenManager(ttl_seconds=10)
    manager.issue_token("active_token")
    assert manager.is_valid("active_token"), "Active token must be valid"
    assert not manager.is_valid("non_existent"), "Missing token must be invalid"
'''

# Offline simulated model response for deterministic offline verification (without live inference)
OFFLINE_SIMULATED_REPAIR_RESPONSE = '''```python
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


class StandaloneCodingHarness:
    """Bounded, sandboxed execution harness for coding workflows.

    Enforces:
    - Strict path resolution and containment via Path.is_relative_to().
    - Write access limited to an explicit mutable source allowlist.
    - Tests, test files, and configs are strictly read-only and immutable.
    - Subprocess timeouts and output truncation.
    - Rejection of unsupported tool arguments.
    """

    MAX_TOOL_OUTPUT_CHARS = 10_000
    SUBPROCESS_TIMEOUT_SECONDS = 15.0

    def __init__(self, workspace: Path, *, writable_allowlist: frozenset[str] | None = None) -> None:
        self.workspace = workspace.resolve()
        self.writable_allowlist = writable_allowlist or frozenset({"auth/token.py"})
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

    def setup_demo_repo(self) -> None:
        """Create sample codebase with authentic bug, unit test, and acceptance check."""
        src_dir = self.workspace / "auth"
        test_dir = self.workspace / "tests"
        src_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "__init__.py").write_text("", encoding="utf-8")
        (src_dir / "token.py").write_text(DEMO_BUGGY_MODULE, encoding="utf-8")
        (test_dir / "__init__.py").write_text("", encoding="utf-8")
        (test_dir / "test_token.py").write_text(DEMO_TEST_MODULE, encoding="utf-8")
        (test_dir / "test_acceptance.py").write_text(DEMO_ACCEPTANCE_TEST_MODULE, encoding="utf-8")

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
            target_rel = args.get("target", "tests/test_token.py")
            ok, target_or_err = self._resolve_and_contain(target_rel)
            if not ok or isinstance(target_or_err, str):
                return False, str(target_or_err)
            target_path = target_or_err
            if not target_path.exists():
                return False, f"Test target not found: '{target_rel}'"

            cmd = [sys.executable, "-m", "pytest", str(target_path), "-v"]
            try:
                res = subprocess.run(
                    cmd,
                    cwd=str(self.workspace),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.SUBPROCESS_TIMEOUT_SECONDS,
                    env={**os.environ, "PYTHONPATH": str(self.workspace)},
                )
                output = res.stdout
                if len(output) > self.MAX_TOOL_OUTPUT_CHARS:
                    output = output[: self.MAX_TOOL_OUTPUT_CHARS] + "\n...[truncated]"
                return (res.returncode == 0), output
            except subprocess.TimeoutExpired:
                return False, f"Test execution timed out after {self.SUBPROCESS_TIMEOUT_SECONDS}s"

        return False, f"Unknown or forbidden tool: {name}"


def parse_model_code_repair(raw_content: str) -> tuple[bool, str, str | None]:
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
    if not code and ("class TokenManager" in raw_content or "def issue_token" in raw_content):
        # Strip extraneous markdown ticks if present
        clean = raw_content.strip()
        if clean.startswith("```") and clean.endswith("```"):
            clean = "\n".join(clean.splitlines()[1:-1]).strip()
        code = clean

    if not code:
        return False, "", "Could not extract valid Python source code from model response"

    # Validate Python syntax via compile()
    try:
        compile(code, "auth/token.py", "exec")
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


async def run_failing_to_passing_workflow(
    *,
    use_live_nebius: bool = False,
    output_log_path: Path | None = None,
) -> dict[str, Any]:
    """Execute end-to-end bug fix workflow with genuine model code repair."""
    load_dotenv()
    temp_dir = Path(tempfile.mkdtemp(prefix="alienese_run_"))
    harness = StandaloneCodingHarness(temp_dir, writable_allowlist=frozenset({"auth/token.py"}))
    harness.setup_demo_repo()

    steps_log: list[dict[str, Any]] = []
    model_telemetry: dict[str, Any] = {}
    applied_diff: str = ""
    start_time = time.time()

    print("\n" + "=" * 70)
    print(" ALIENESE END-TO-END AUTONOMOUS CODING RUN")
    print(f" Workspace: {temp_dir}")
    print(f" Mode: {'LIVE NEBIUS TOKEN FACTORY' if use_live_nebius else 'OFFLINE SIMULATION'}")
    print("=" * 70)

    # Step 1: Initial failing test execution
    print("\n[Step 1] Running initial test suite...")
    ok, test_out = harness.execute_tool("run_test", {"target": "tests/test_token.py"})
    print(f"Result: {'PASSED' if ok else 'FAILED (Expected initial bug)'}")
    print("-" * 50)
    for line in test_out.strip().splitlines()[-6:]:
        print(f"  {line}")
    print("-" * 50)
    steps_log.append(
        {
            "step": 1,
            "action": "run_test",
            "target": "tests/test_token.py",
            "passed": ok,
            "output_summary": test_out[-300:],
        }
    )
    assert not ok, "Initial test must fail to demonstrate authentic repair"

    # Step 2: Initialize Alienese Engine with Nebius Generator and GroundedRankController
    gen: Any
    http_client: Any = None
    if use_live_nebius:
        key = os.getenv("NEBIUS_TOKEN_FACTORY_KEY")
        base_url = os.getenv(
            "NEBIUS_TOKEN_FACTORY_BASE_URL",
            "https://api.tokenfactory.us-central1.nebius.com/v1",
        )
        key = key or ""
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
        # Step 3: Turn 1 - Ground failure traceback deterministically and inspect source
        print("\n[Step 2] Turn 1: Deterministic grounding of failure traceback into action...")
        from alienese.api.models import ChatMessageInput

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

        ok1, file_content = harness.execute_tool(tool_name_1, tool_args_1)
        print(f"Tool executed: read {len(file_content)} bytes")
        steps_log.append(
            {
                "step": 2,
                "turn": 1,
                "tool": tool_name_1,
                "args": tool_args_1,
                "success": ok1,
                "candidate_count": artifact1.semantics.candidate_count,
            }
        )

        # Step 4: Turn 2 - Genuine Model Code Generation
        # Feed the failing test output, grounded source path, and actual buggy source code to Nemotron
        print("\n[Step 3] Turn 2: Requesting genuine code repair from Nemotron 3 Super...")
        repair_prompt = (
            "You are an expert Python software engineer.\n"
            "A test failed in the test suite:\n"
            f"{test_out}\n\n"
            "Here is the current content of 'auth/token.py':\n"
            f"{file_content}\n\n"
            "Fix the bug in 'auth/token.py' so that the token validation works as intended.\n"
            "Output ONLY the complete, corrected Python code for 'auth/token.py' inside a ```python``` code block.\n"
            "Do not include any explanation before or after."
        )

        raw_repair_output: str = ""
        if use_live_nebius:
            ctx_gen = RequestContext(request_id="run-turn-gen", operation_id="op-turn-gen")
            job_gen = GenerationJob(
                job_id="job_repair_token",
                job_type=GenerationJobType.ANSWER,
                candidate_id="cand_answer",
                latest_user_request=repair_prompt,
                max_tokens=1024,
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
            }
            print(f"Nemotron generation latency: {res_gen.telemetry.latency_ms:.1f}ms")
            print(f"Tokens: {res_gen.telemetry.total_tokens} total ({res_gen.telemetry.completion_tokens} completion)")
        else:
            raw_repair_output = OFFLINE_SIMULATED_REPAIR_RESPONSE
            model_telemetry = {"simulated": True, "note": "Offline replay/simulation"}
            print("Using offline simulated model repair (deterministic verification mode)")

        # Parse and validate the model-generated repair
        parse_ok, repaired_code, parse_err = parse_model_code_repair(raw_repair_output)
        if not parse_ok:
            raise RuntimeError(f"Model repair rejected: {parse_err}")

        # Compute diff before applying
        original_source = DEMO_BUGGY_MODULE
        applied_diff = compute_unified_diff(original_source, repaired_code, "auth/token.py")
        print("\nModel-Generated Unified Diff:")
        print("-" * 50)
        for line in applied_diff.splitlines():
            print(f"  {line}")
        print("-" * 50)

        # Step 5: Turn 2 execution - Apply the model-generated code repair
        print("Applying model-generated repair to auth/token.py in disposable sandbox...")
        ok_patch, patch_msg = harness.execute_tool(
            "write_file",
            {"path": "auth/token.py", "content": repaired_code},
        )
        print(f"Patch applied: {patch_msg}")
        steps_log.append(
            {
                "step": 3,
                "turn": 2,
                "action": "write_file",
                "path": "auth/token.py",
                "success": ok_patch,
                "model_telemetry": model_telemetry,
                "diff": applied_diff,
            }
        )

        # Step 6: Turn 3 - Run verification test target
        print("\n[Step 4] Turn 3: Re-running test target to confirm fix...")
        ok_final, test_final_out = harness.execute_tool(
            "run_test", {"target": "tests/test_token.py"}
        )
        print(f"Target Test Result: {'PASSED (Fix Verified)' if ok_final else 'FAILED'}")
        print("-" * 50)
        for line in test_final_out.strip().splitlines()[-4:]:
            print(f"  {line}")
        print("-" * 50)

        # Step 7: Independent Acceptance Check
        print("\n[Step 5] Running independent acceptance test (test_acceptance.py)...")
        ok_acceptance, test_acc_out = harness.execute_tool(
            "run_test", {"target": "tests/test_acceptance.py"}
        )
        print(f"Acceptance Test Result: {'PASSED' if ok_acceptance else 'FAILED'}")
        print("-" * 50)
        for line in test_acc_out.strip().splitlines()[-4:]:
            print(f"  {line}")
        print("-" * 50)
        steps_log.append(
            {
                "step": 4,
                "turn": 3,
                "action": "run_test",
                "target": "tests/test_token.py",
                "passed": ok_final,
                "output_summary": test_final_out[-200:],
            }
        )
        steps_log.append(
            {
                "step": 5,
                "action": "run_acceptance_test",
                "target": "tests/test_acceptance.py",
                "passed": ok_acceptance,
                "output_summary": test_acc_out[-200:],
            }
        )

        # Step 8: Turn 4 - Ask Nebius for final summary explanation
        print("\n[Step 6] Turn 4: Asking Nebius Token Factory for repair explanation...")
        explanation = ""
        if use_live_nebius:
            ctx_sum = RequestContext(request_id="run-turn-sum", operation_id="op-turn-sum")
            job_sum = GenerationJob(
                job_id="job_explain_repair",
                job_type=GenerationJobType.ANSWER,
                candidate_id="cand_answer",
                trusted_system_instructions=("You are Alienese coding agent. Summarize the token expiry fix in 2 sentences.",),
                latest_user_request="We fixed TokenManager.issue_token() by recording time.time() instead of time.time() - ttl. Tests now pass.",
                max_tokens=128,
                temperature=0.0,
            )
            res_sum = await gen.generate(ctx_sum, job_sum)
            explanation = res_sum.semantics.content.strip()
        else:
            explanation = "Fixed TokenManager.issue_token() to assign current timestamp rather than past offset."
        print(f'Nebius Nemotron Explanation:\n  "{explanation}"')
        steps_log.append(
            {
                "step": 6,
                "turn": 4,
                "explanation": explanation,
            }
        )

    finally:
        if http_client is not None:
            await http_client.aclose()
        shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    overall_status = "SUCCESS" if (ok_final and ok_acceptance) else "FAILED"
    summary = {
        "status": overall_status,
        "elapsed_ms": elapsed_ms,
        "use_live_nebius": use_live_nebius,
        "model_telemetry": model_telemetry,
        "applied_diff": applied_diff,
        "steps": steps_log,
    }

    if output_log_path:
        output_log_path.parent.mkdir(parents=True, exist_ok=True)
        output_log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nExecution log written to: {output_log_path}")

    print("\n" + "=" * 70)
    print(f" WORKFLOW COMPLETED IN {elapsed_ms}ms — STATUS: {summary['status']}")
    print("=" * 70 + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Alienese Autonomous Coding Runner")
    parser.add_argument("--live", action="store_true", help="Use live Nebius Token Factory")
    parser.add_argument(
        "--out", type=Path, default=Path("results/live_run_report.json"), help="Output JSON log"
    )
    args = parser.parse_args()

    asyncio.run(
        run_failing_to_passing_workflow(
            use_live_nebius=args.live,
            output_log_path=args.out,
        )
    )


if __name__ == "__main__":
    main()
