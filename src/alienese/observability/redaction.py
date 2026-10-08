"""Central secret redaction boundary for logs, settings, and persisted replay artifacts.

This module enforces Invariant 14 from AGENTS.md:
Secrets must never enter logs, traces, or model context accidentally.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence, Set
from typing import Any

from pydantic import SecretStr

REDACTED_PLACEHOLDER = "[REDACTED]"

_SENSITIVE_KEY_NAMES: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "api_key",
        "apikey",
        "x_api_key",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "client_secret",
        "secret",
        "secret_key",
        "private_key",
        "password",
        "passwd",
        "credential",
        "credentials",
        "retriever_api_key",
        "controller_api_key",
        "generator_api_key",
    }
)

_SENSITIVE_KEY_SUFFIXES: tuple[str, ...] = (
    "_api_key",
    "_apikey",
    "_token",
    "_secret",
    "_password",
    "_private_key",
    "_credential",
    "_credentials",
)

_BEARER_PATTERN = re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]{6,}")
_BASIC_AUTH_PATTERN = re.compile(r"(?i)\b(Basic\s+)[A-Za-z0-9+/=]{6,}")
_PEM_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
)
_KNOWN_TOKEN_PREFIX_PATTERN = re.compile(
    r"\b(?:sk|rk|pk|nvapi)-[A-Za-z0-9_-]{12,}\b"
    r"|\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\b"
    r"|\bgithub_pat_[A-Za-z0-9_]{16,}\b"
    r"|\bhf_[A-Za-z0-9]{16,}\b"
    r"|\bxox[baprs]-[A-Za-z0-9-]{12,}\b"
    r"|\bAIza[0-9A-Za-z_-]{20,}\b"
)
_JSON_KEY_VALUE_PATTERN = re.compile(
    r'(?i)("(?:[a-z0-9_]*_)?(?:api[_-]?key|apikey|secret|token|password|private_key|authorization)"\s*:\s*")'
    r'(?!\[REDACTED\])([^"]{4,})(")'
)
_INLINE_KEY_VALUE_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|apikey|secret|token|password|authorization)\s*([:=])\s*"
    r"(?!Bearer\b|Basic\b|\[REDACTED\])([^\s,;\x22\x27&]{6,})"
)


def is_sensitive_key(key: str) -> bool:
    """Return True if a mapping key name designates a secret or credential."""
    normalized = key.strip().lower().replace("-", "_")
    if normalized in _SENSITIVE_KEY_NAMES:
        return True
    return normalized.endswith(_SENSITIVE_KEY_SUFFIXES)


def redact_string(value: str) -> str:
    """Redact obvious secret patterns inside a string value."""
    if not value:
        return value
    result = _PEM_PRIVATE_KEY_PATTERN.sub(REDACTED_PLACEHOLDER, value)
    result = _BEARER_PATTERN.sub(rf"\1{REDACTED_PLACEHOLDER}", result)
    result = _BASIC_AUTH_PATTERN.sub(rf"\1{REDACTED_PLACEHOLDER}", result)
    result = _KNOWN_TOKEN_PREFIX_PATTERN.sub(REDACTED_PLACEHOLDER, result)
    result = _JSON_KEY_VALUE_PATTERN.sub(rf"\1{REDACTED_PLACEHOLDER}\3", result)
    result = _INLINE_KEY_VALUE_PATTERN.sub(rf"\1\2{REDACTED_PLACEHOLDER}", result)
    return result


def redact_value(value: Any) -> Any:
    """Recursively redact sensitive keys and string patterns in arbitrary data structures."""
    if isinstance(value, SecretStr):
        return REDACTED_PLACEHOLDER
    if isinstance(value, str):
        return redact_string(value)
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, frozenset):
        return frozenset(redact_value(item) for item in value)
    if isinstance(value, Set):
        return {redact_value(item) for item in value}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [redact_value(item) for item in value]
    return value


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of a mapping with all sensitive keys and string values redacted."""
    redacted: dict[str, Any] = {}
    for raw_key, raw_val in data.items():
        key_str = str(raw_key)
        if is_sensitive_key(key_str):
            redacted[key_str] = REDACTED_PLACEHOLDER if raw_val is not None else None
        else:
            redacted[key_str] = redact_value(raw_val)
    return redacted
