"""Evidence normalization, deduplication, and sanity sanitization."""

from __future__ import annotations

import re
from collections.abc import Sequence

from alienese.contracts.events import TrustLevel
from alienese.grounding.evidence import EvidenceCategory, EvidenceStatus, GroundingEvidence

MAX_PATH_LENGTH = 512
MAX_COMMAND_LENGTH = 1024
MAX_SYMBOL_LENGTH = 256
MAX_SNIPPET_LENGTH = 2048

_UNSAFE_PATH_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_PATH_STRIP_CHARS = "'\"`<>[],;:() \t\n\r"


def normalize_file_path(raw_path: str) -> str | None:
    """Clean and normalize a raw file path.

    Returns None if the path is invalid, empty, contains control characters,
    exceeds length bounds, or exhibits unsafe path traversal.
    """
    if not raw_path or not isinstance(raw_path, str):
        return None

    cleaned = raw_path.strip(_PATH_STRIP_CHARS)
    if not cleaned or len(cleaned) > MAX_PATH_LENGTH:
        return None

    if _UNSAFE_PATH_CHARS.search(cleaned):
        return None

    # Normalize slashes to forward slash for consistent internal representation
    normalized = cleaned.replace("\\", "/")

    # Reject dangerous path traversal sequences if escaping base workspace
    parts = normalized.split("/")
    if ".." in parts:
        # Check if attempts ../ traversal
        resolved_parts: list[str] = []
        for part in parts:
            if part in ("", "."):
                continue
            if part == "..":
                if not resolved_parts or resolved_parts[-1] == "..":
                    # Navigates above root/base
                    return None
                resolved_parts.pop()
            else:
                resolved_parts.append(part)
        normalized = "/".join(resolved_parts)
        if not normalized:
            return None

    # Strip leading ./
    if normalized.startswith("./"):
        normalized = normalized[2:]

    # Remove any line number suffix accidentally attached (e.g., file.py:42)
    if ":" in normalized:
        prefix, suffix = normalized.rsplit(":", 1)
        if suffix.isdigit():
            normalized = prefix

    return normalized if normalized else None


_TEST_TARGET_STRIP_CHARS = "'\"`<> \t\n\r"


def normalize_test_target(raw_target: str) -> str | None:
    """Normalize a pytest node ID or test filename."""
    if not raw_target or not isinstance(raw_target, str):
        return None

    cleaned = raw_target.strip(_TEST_TARGET_STRIP_CHARS)
    if not cleaned or len(cleaned) > MAX_PATH_LENGTH:
        return None

    if _UNSAFE_PATH_CHARS.search(cleaned):
        return None

    # If it's a pytest node ID: path/to/test.py::test_func
    if "::" in cleaned:
        path_part, *test_parts = cleaned.split("::")
        norm_path = normalize_file_path(path_part)
        if not norm_path:
            return None
        return f"{norm_path}::{'::'.join(test_parts)}"

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
