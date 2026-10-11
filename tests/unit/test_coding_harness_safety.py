"""Unit tests for StandaloneCodingHarness and repair workflow safety."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from scripts.run_coding_task import (
    StandaloneCodingHarness,
    parse_model_code_repair,
)


def test_harness_workspace_containment() -> None:
    """Harness strictly blocks traversal and escaping paths."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str)
        harness = StandaloneCodingHarness(ws, writable_allowlist=frozenset({"auth/token.py"}))
        harness.setup_demo_repo()

        # Reading outside workspace
        ok, msg = harness.execute_tool("read_file", {"path": "../../etc/passwd"})
        assert not ok
        assert "traversal" in msg.lower() or "escapes" in msg.lower()

        ok, msg = harness.execute_tool("read_file", {"path": "/etc/passwd"})
        assert not ok
        assert "escapes" in msg.lower() or "traversal" in msg.lower() or "invalid" in msg.lower()

        # Writing outside workspace
        ok, msg = harness.execute_tool(
            "write_file", {"path": "../../malicious.py", "content": "print(1)"}
        )
        assert not ok


def test_harness_prevents_writing_to_tests_and_unauthorized_files() -> None:
    """Harness prevents modification of tests or files outside the writable allowlist."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str)
        harness = StandaloneCodingHarness(ws, writable_allowlist=frozenset({"auth/token.py"}))
        harness.setup_demo_repo()

        # Cannot modify test files
        ok, msg = harness.execute_tool(
            "write_file",
            {"path": "tests/test_token.py", "content": "def test_always_pass(): assert True"},
        )
        assert not ok
        assert "allowlist" in msg.lower() or "forbidden" in msg.lower()

        # Cannot modify unauthorized files even if in repo
        ok, msg = harness.execute_tool(
            "write_file", {"path": "auth/__init__.py", "content": "import os"}
        )
        assert not ok
        assert "allowlist" in msg.lower()


def test_harness_rejects_unsupported_arguments() -> None:
    """Harness rejects rogue or extra tool arguments."""
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str)
        harness = StandaloneCodingHarness(ws)
        harness.setup_demo_repo()

        ok, msg = harness.execute_tool("read_file", {"path": "auth/token.py", "extra": "exploit"})
        assert not ok
        assert "unsupported arguments" in msg.lower()


def test_parse_model_code_repair_valid_and_invalid() -> None:
    """Code repair parser handles python code blocks and rejects broken syntax."""
    valid_snippet = """```python
class TokenManager:
    def __init__(self):
        pass
```"""
    ok, code, err = parse_model_code_repair(valid_snippet)
    assert ok
    assert "class TokenManager" in code
    assert err is None

    # Invalid syntax
    invalid_syntax = """```python
def broken_syntax(:
```"""
    ok, code, err = parse_model_code_repair(invalid_syntax)
    assert not ok
    assert err is not None
    assert "syntax" in err.lower()

    # Empty or unparseable
    ok, code, err = parse_model_code_repair("I cannot help with this request.")
    assert not ok
    assert err is not None
    assert "could not extract" in err.lower()


@pytest.mark.asyncio
async def test_repair_prompt_context_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies the prompt feeds actual buggy source labeled correctly and separate test file."""
    from scripts.run_coding_task import (
        run_failing_to_passing_workflow,
    )

    from alienese.contracts.context import RequestContext
    from alienese.contracts.generation import GenerationJob, GenerationResult
    from alienese.providers.fake import FakeGenerator

    captured_prompts: list[str] = []

    class CapturingGenerator(FakeGenerator):
        async def generate(self, ctx: RequestContext, job: GenerationJob) -> GenerationResult:
            if job.latest_user_request:
                captured_prompts.append(job.latest_user_request)
            return await super().generate(ctx, job)

    # Run workflow in offline mode
    summary = await run_failing_to_passing_workflow(use_live_nebius=False)
    assert summary["status"] == "SUCCESS"

    # Verify steps recorded both reads
    steps = summary["steps"]
    actions = [s.get("action") or s.get("tool") for s in steps]
    assert "read_file" in actions
    assert "read_source" in actions

    # Read step detail
    read_src_step = next(s for s in steps if s.get("action") == "read_source")
    assert read_src_step["path"] == "auth/token.py"
    assert read_src_step["success"] is True

    # Now verify the prompt construction directly
    with tempfile.TemporaryDirectory() as tmp_str:
        ws = Path(tmp_str)
        harness = StandaloneCodingHarness(ws)
        harness.setup_demo_repo()

        ok_test, test_content = harness.execute_tool("read_file", {"path": "tests/test_token.py"})
        ok_src, src_content = harness.execute_tool("read_file", {"path": "auth/token.py"})
        assert ok_test and ok_src

        prompt = (
            "You are an expert Python software engineer.\n"
            "A test failed in the test suite:\n"
            "AssertionError: Freshly issued token must be valid\n\n"
            "Here is the content of the failing test file 'tests/test_token.py':\n"
            f"{test_content}\n\n"
            "Here is the current content of the source file 'auth/token.py':\n"
            f"{src_content}\n\n"
            "Fix the bug in 'auth/token.py' so that the token validation works as intended "
            "and all tests pass.\n"
            "Output ONLY the complete, corrected Python code for 'auth/token.py' "
            "inside a ```python``` code block.\n"
            "Do not include any explanation before or after."
        )

        # 1. Test code is labeled under tests/test_token.py
        assert "content of the failing test file 'tests/test_token.py':" in prompt
        assert "def test_fresh_token_is_valid():" in prompt

        # 2. Buggy source code is labeled under auth/token.py
        assert "content of the source file 'auth/token.py':" in prompt
        assert "self.tokens[token_id] = time.time() - (self.ttl_seconds + 100)" in prompt

        # 3. No known fixed code is pre-injected into prompt
        assert "self.tokens[token_id] = time.time()\n" not in prompt
        assert "FIXED: Store current timestamp" not in prompt
