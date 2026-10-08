"""Unit tests for central secret redaction and span attribute sanitization."""

from __future__ import annotations

from pydantic import SecretStr

from alienese.observability.redaction import (
    REDACTED_PLACEHOLDER,
    is_sensitive_key,
    redact_mapping,
    redact_string,
    redact_value,
)
from alienese.observability.tracing import sanitize_span_attributes


def test_sensitive_key_detection() -> None:
    assert is_sensitive_key("Authorization")
    assert is_sensitive_key("api_key")
    assert is_sensitive_key("X-API-Key")
    assert is_sensitive_key("retriever_api_key")
    assert is_sensitive_key("access_token")
    assert is_sensitive_key("client_secret")
    assert is_sensitive_key("private_key")
    assert is_sensitive_key("custom_service_token")
    assert not is_sensitive_key("request_id")
    assert not is_sensitive_key("completion_tokens")
    assert not is_sensitive_key("prompt_tokens")


def test_redact_string_patterns() -> None:
    # Bearer tokens
    text = "Header Authorization: Bearer " + "abcdef1234567890.xyz_token"
    redacted = redact_string(text)
    assert "abcdef1234567890" not in redacted
    assert f"Bearer {REDACTED_PLACEHOLDER}" in redacted

    # API key prefixes
    sk_token = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    nv_token = "nvapi-" + "9876543210abcdef"
    sk_text = f"Using key {sk_token} and {nv_token}"
    redacted_sk = redact_string(sk_text)
    assert "abcdef" not in redacted_sk
    assert "98765" not in redacted_sk

    # PEM private keys
    pem_begin = "".join(["-----BEGIN ", "RSA PRIVATE KEY-----"])
    pem_end = "".join(["-----END ", "RSA PRIVATE KEY-----"])
    pem = f"Prefix\n{pem_begin}\nMIIEpAIBAAKCAQEA0123456789secretkeymaterial\n{pem_end}\nSuffix"
    redacted_pem = redact_string(pem)
    assert "MIIEpAIBAAKCAQEA0123456789secretkeymaterial" not in redacted_pem
    assert REDACTED_PLACEHOLDER in redacted_pem


def test_redact_mapping_and_nested_values() -> None:
    payload = {
        "request_id": "req_123",
        "Authorization": "Bearer secret_jwt_token_value",
        "nested": {
            "api_key": "top-secret-key",
            "secret_obj": SecretStr("pydantic-secret-val"),
            "items": [
                {"user_token": "tok_99999999"},
                "inline api_key=my_super_secret_12345",
            ],
        },
    }
    cleaned = redact_mapping(payload)
    assert cleaned["request_id"] == "req_123"
    assert cleaned["Authorization"] == REDACTED_PLACEHOLDER
    assert cleaned["nested"]["api_key"] == REDACTED_PLACEHOLDER
    assert cleaned["nested"]["secret_obj"] == REDACTED_PLACEHOLDER
    assert cleaned["nested"]["items"][0]["user_token"] == REDACTED_PLACEHOLDER
    assert "my_super_secret_12345" not in cleaned["nested"]["items"][1]

    # Tuple preservation
    tup = redact_value(("Bearer my_secret_token_12345", "safe_value"))
    assert isinstance(tup, tuple)
    assert "my_secret_token_12345" not in tup[0]
    assert tup[1] == "safe_value"


def test_sanitize_span_attributes_strips_raw_content_and_secrets() -> None:
    attrs: dict[str, str | int | float | bool] = {
        "event_count": 4,
        "state_digest": "a" * 64,
        "content": "raw source code that must not enter spans",
        "prompt": "raw user prompt",
        "tool_output": "raw repository file content",
        "api_key": "secret-key",
        "note": "Bearer hidden_jwt_token_987654",
    }
    clean = sanitize_span_attributes(attrs)
    assert clean["event_count"] == 4
    assert clean["state_digest"] == "a" * 64
    assert "content" not in clean
    assert "prompt" not in clean
    assert "tool_output" not in clean
    assert "api_key" not in clean
    assert "hidden_jwt_token_987654" not in str(clean["note"])
