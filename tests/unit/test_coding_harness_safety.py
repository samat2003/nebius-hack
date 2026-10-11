"""Unit tests for StandaloneCodingHarness and repair workflow safety."""

from __future__ import annotations

import tempfile
from pathlib import Path

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
