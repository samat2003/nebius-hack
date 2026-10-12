"""Alienese CLI Bridge for OpenCode Plugin.

Provides a deterministic subprocess bridge between OpenCode hooks and
Alienese's Python runtime, grounding engine, and safety policies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from alienese.opencode.tracker import OpenCodeSessionTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alienese OpenCode Bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # pre-tool subcommand
    pre = subparsers.add_parser("pre-tool", help="Evaluate pre-tool safety guards")
    pre.add_argument("--workspace", required=True, help="Workspace directory")
    pre.add_argument("--session", default="default", help="OpenCode session ID")
    pre.add_argument("--call-id", default=None, help="Tool call ID")
    pre.add_argument("--tool", required=True, help="Tool name")
    pre.add_argument("--args", required=True, help="Tool arguments JSON string")
    pre.add_argument("--trace-file", default=None, help="Custom trace file path")
    pre.add_argument(
        "--read-only-tests",
        action="store_true",
        help="Disallow mutations to test files",
    )

    # post-tool subcommand
    post = subparsers.add_parser("post-tool", help="Process post-tool output and evidence")
    post.add_argument("--workspace", required=True, help="Workspace directory")
    post.add_argument("--session", default="default", help="OpenCode session ID")
    post.add_argument("--call-id", default=None, help="Tool call ID")
    post.add_argument("--tool", required=True, help="Tool name")
    post.add_argument("--args", required=True, help="Tool arguments JSON string")
    post.add_argument("--output", required=True, help="Tool output string or JSON")
    post.add_argument("--trace-file", default=None, help="Custom trace file path")

    # status subcommand
    stat = subparsers.add_parser("status", help="Get session status")
    stat.add_argument("--workspace", required=True, help="Workspace directory")
    stat.add_argument("--session", default="default", help="OpenCode session ID")
    stat.add_argument("--trace-file", default=None, help="Custom trace file path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)
    trace_path = Path(args.trace_file) if getattr(args, "trace_file", None) else None

    if args.command == "pre-tool":
        try:
            tool_args = json.loads(args.args)
        except Exception:
            tool_args = {}

        tracker = OpenCodeSessionTracker(
            workspace=workspace,
            session_id=args.session,
            trace_file=trace_path,
            read_only_test_files=args.read_only_tests,
        )
        decision = tracker.evaluate_pre_tool(args.tool, tool_args, call_id=args.call_id)
        sys.stdout.write(json.dumps(decision) + "\n")
        sys.stdout.flush()

    elif args.command == "post-tool":
        try:
            tool_args = json.loads(args.args)
        except Exception:
            tool_args = {}

        output_val: Any
        try:
            parsed = json.loads(args.output)
            output_val = parsed if isinstance(parsed, (str, dict)) else str(args.output)
        except Exception:
            output_val = args.output

        tracker = OpenCodeSessionTracker(
            workspace=workspace,
            session_id=args.session,
            trace_file=trace_path,
        )
        result = tracker.process_post_tool(args.tool, tool_args, output_val, call_id=args.call_id)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()

    elif args.command == "status":
        tracker = OpenCodeSessionTracker(
            workspace=workspace,
            session_id=args.session,
            trace_file=trace_path,
        )
        status = tracker.get_status()
        sys.stdout.write(json.dumps(status) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
