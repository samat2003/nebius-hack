"""Evidence normalization, deduplication, and sanity sanitization."""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence

from alienese.contracts.events import TrustLevel
from alienese.grounding.evidence import EvidenceCategory, EvidenceStatus, GroundingEvidence

MAX_PATH_LENGTH = 512
MAX_COMMAND_LENGTH = 1024
MAX_SYMBOL_LENGTH = 256
MAX_SNIPPET_LENGTH = 2048

_UNSAFE_PATH_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_PATH_STRIP_CHARS = "'\"`<>[],;:() \t\n\r"

_URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_WINDOWS_DRIVE_RE = re.compile(r"^[a-zA-Z]:")
_TEST_NODE_PART_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\[[^\]\x00-\x1f\x7f]+\])?$")

_SAFE_EXECUTABLES = frozenset({"pytest", "python", "python3", "ruff", "mypy"})
_FORBIDDEN_SHELL_CHARS = re.compile(r"[;&|><$`\n\r\\]")


def normalize_file_path(raw_path: str) -> str | None:
    """Clean and normalize a raw repository-relative file path.

    Enforces repository containment:
    Rejects:
    - Unix absolute paths (e.g. /etc/passwd)
    - Windows drive-absolute (C:\\...) and drive-relative (C:foo) paths
    - UNC paths (\\\\server\\share or //server/share)
    - URI-style path forms (file://, http://)
    - Traversal escaping the workspace (../)
    - Invalid control characters
    """
    if not raw_path or not isinstance(raw_path, str):
        return None

    cleaned = raw_path.strip(_PATH_STRIP_CHARS)
    if not cleaned or len(cleaned) > MAX_PATH_LENGTH:
        return None

    if _UNSAFE_PATH_CHARS.search(cleaned):
        return None

    # Reject URI schemes
    if _URI_SCHEME_RE.match(cleaned):
        return None

    # Reject UNC paths
    if cleaned.startswith(("//", "\\\\")):
        return None

    # Reject Unix absolute paths and Windows root-relative paths
    if cleaned.startswith(("/", "\\")):
        return None

    # Reject Windows drive-absolute and drive-relative paths
    if _WINDOWS_DRIVE_RE.match(cleaned):
        return None

    # Normalize slashes to forward slash
    normalized = cleaned.replace("\\", "/")

    # Check for line number suffix (e.g., file.py:42)
    # Must only be stripped if suffix is numeric and preceding path has no other colons
    if ":" in normalized:
        prefix, suffix = normalized.rsplit(":", 1)
        if suffix.isdigit():
            normalized = prefix
        else:
            return None

    if ":" in normalized:
        return None

    # Reject empty or leading slash after normalization
    if not normalized or normalized.startswith("/"):
        return None

    # Resolve .. and . segments strictly within repository workspace
    parts = normalized.split("/")
    resolved_parts: list[str] = []
    for part in parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not resolved_parts:
                # Navigates above repository workspace
                return None
            resolved_parts.pop()
        else:
            resolved_parts.append(part)

    if not resolved_parts:
        return None

    result = "/".join(resolved_parts)
    # Reject shell wildcards or illegal file characters
    if any(c in result for c in '*?<>|"'):
        return None

    return result


_TEST_TARGET_STRIP_CHARS = "'\"`<> \t\n\r"


def normalize_test_target(raw_target: str) -> str | None:
    """Normalize a pytest node ID or test filename, rejecting shell chars and malformed IDs."""
    if not raw_target or not isinstance(raw_target, str):
        return None

    cleaned = raw_target.strip(_TEST_TARGET_STRIP_CHARS)
    if not cleaned or len(cleaned) > MAX_PATH_LENGTH:
        return None

    if _UNSAFE_PATH_CHARS.search(cleaned):
        return None

    # Reject shell metacharacters
    if any(c in cleaned for c in ";|&$`'\"<>\n\r\t"):
        return None

    # Pytest node ID: path/to/test.py::TestClass::test_method
    if "::" in cleaned:
        path_part, *test_parts = cleaned.split("::")
        if not test_parts or any(not p.strip() for p in test_parts):
            return None

        norm_path = normalize_file_path(path_part)
        if not norm_path or not norm_path.endswith(".py"):
            return None

        clean_parts: list[str] = []
        for part in test_parts:
            p = part.strip()
            if not p or not _TEST_NODE_PART_RE.match(p):
                return None
            clean_parts.append(p)

        return f"{norm_path}::{'::'.join(clean_parts)}"

    return normalize_file_path(cleaned)


def normalize_symbol_name(raw_symbol: str) -> str | None:
    """Normalize a function, class, or module symbol identifier."""
    if not raw_symbol or not isinstance(raw_symbol, str):
        return None

    cleaned = raw_symbol.strip(" \t\n\r()[]`'\"")
    if not cleaned or len(cleaned) > MAX_SYMBOL_LENGTH:
        return None

    # Valid Python identifier or qualified identifier (e.g. Class.method)
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$", cleaned):
        return None

    return cleaned


def is_safe_verification_command(command: str) -> tuple[bool, str | None]:
    """Validate whether command is a safe, narrowly recognized verification command.

    Rejects arbitrary shell pipelines, chaining, redirection, substitution,
    destructive commands, and unsupported executables.
    """
    if not command or not isinstance(command, str):
        return False, "empty_command"

    cleaned = command.strip()
    if len(cleaned) > MAX_COMMAND_LENGTH:
        return False, "command_too_long"

    # Reject shell metacharacters: chaining, pipelines, redirection, backticks, substitutions
    if _FORBIDDEN_SHELL_CHARS.search(cleaned):
        return False, "shell_metacharacters_detected"

    if "${" in cleaned or "$(" in cleaned:
        return False, "command_substitution_detected"

    try:
        tokens = shlex.split(cleaned)
    except ValueError:
        return False, "invalid_shell_quoting"

    if not tokens:
        return False, "empty_tokens"

    exe = tokens[0]
    if exe not in _SAFE_EXECUTABLES:
        return False, f"unsupported_executable:{exe}"

    # Verify python invocations
    if exe in ("python", "python3"):
        if len(tokens) < 3:
            return False, "incomplete_python_command"
        if tokens[1] != "-m":
            return False, "python_without_module_flag"
        module = tokens[2]
        if module not in ("pytest", "unittest", "py_compile"):
            return False, f"unsupported_python_module:{module}"
        sub_tokens = tokens[3:]
    elif exe == "pytest":
        sub_tokens = tokens[1:]
    elif exe == "ruff":
        if len(tokens) < 2 or tokens[1] not in ("check", "format"):
            return False, "unsupported_ruff_subcommand"
        if tokens[1] == "format" and (len(tokens) < 3 or tokens[2] != "--check"):
            return False, "ruff_format_without_check_flag"
        sub_tokens = tokens[2:] if tokens[1] == "check" else tokens[3:]
    elif exe == "mypy":
        sub_tokens = tokens[1:]
    else:
        return False, "unrecognized_executable"

    # Validate remaining arguments (must be safe flags, paths, or test targets)
    for tok in sub_tokens:
        if tok.startswith("-"):
            if not re.match(r"^--?[a-zA-Z0-9_\-=]+$", tok):
                return False, f"invalid_flag:{tok}"
        else:
            norm = normalize_test_target(tok) or normalize_file_path(tok)
            if not norm:
                return False, f"unrecognized_command_argument:{tok}"

    return True, None


def deduplicate_evidence(
    evidence_items: Sequence[GroundingEvidence],
) -> tuple[GroundingEvidence, ...]:
    """Deduplicate evidence deterministically while preserving highest trust and status."""
    seen: dict[tuple[EvidenceCategory, str], GroundingEvidence] = {}

    for item in evidence_items:
        key = (item.category, item.value)
        existing = seen.get(key)
        if existing is None:
            seen[key] = item
            continue

        # Comparison logic for keeping the best evidence record:
        # 1. Higher trust wins (SYSTEM_TRUSTED > USER > MODEL_GENERATED > UNTRUSTED_EXTERNAL)
        trust_order = {
            TrustLevel.SYSTEM_TRUSTED: 4,
            TrustLevel.USER: 3,
            TrustLevel.MODEL_GENERATED: 2,
            TrustLevel.UNTRUSTED_EXTERNAL: 1,
        }
        item_trust_score = trust_order.get(item.trust, 0)
        existing_trust_score = trust_order.get(existing.trust, 0)

        # 2. Confirmed > Observed > Inferred > Failed
        status_order = {
            EvidenceStatus.CONFIRMED: 4,
            EvidenceStatus.OBSERVED: 3,
            EvidenceStatus.INFERRED: 2,
            EvidenceStatus.FAILED: 1,
        }
        item_status_score = status_order.get(item.status, 0)
        existing_status_score = status_order.get(existing.status, 0)

        should_replace = False
        if item_trust_score > existing_trust_score or (
            item_trust_score == existing_trust_score
            and (
                item_status_score > existing_status_score
                or (
                    item_status_score == existing_status_score
                    and item.sequence_no > existing.sequence_no
                )
            )
        ):
            should_replace = True

        if should_replace:
            seen[key] = item

    # Sort deterministically by (category, sequence_no, value)
    sorted_items = sorted(
        seen.values(),
        key=lambda e: (e.category.value, e.sequence_no, e.value),
    )
    return tuple(sorted_items)
