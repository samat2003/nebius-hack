"""Unit tests for deterministic OpenAI message and tool normalization."""

from __future__ import annotations

import pytest

from alienese.api.errors import CompatibilityError, ProtocolError
from alienese.api.models import ChatCompletionRequest
from alienese.contracts.events import EventKind, SourceRole, TrustLevel
from alienese.contracts.state import CanonicalCapability
from alienese.engine.normalize import normalize_request


def test_normalize_simple_user_conversation() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "Explain how the turn engine works."}],
        }
    )
    events, tools = normalize_request(req)
    assert len(events) == 1
    assert tools == ()
    assert events[0].sequence_no == 0
    assert events[0].kind == EventKind.USER_MESSAGE
    assert events[0].trust == TrustLevel.USER
    assert events[0].provenance.source_role == SourceRole.USER
    assert events[0].content == "Explain how the turn engine works."


def test_normalize_system_developer_user_and_assistant() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [
                {"role": "system", "content": "System policy A"},
                {"role": "developer", "content": "Developer policy B"},
                {"role": "user", "content": "User prompt"},
                {"role": "assistant", "content": "Assistant reply"},
            ],
        }
    )
    events, _ = normalize_request(req)
    assert [e.kind for e in events] == [
        EventKind.SYSTEM_MESSAGE,
        EventKind.SYSTEM_MESSAGE,
        EventKind.USER_MESSAGE,
        EventKind.ASSISTANT_MESSAGE,
    ]
    assert [e.trust for e in events] == [
        TrustLevel.SYSTEM_TRUSTED,
        TrustLevel.SYSTEM_TRUSTED,
        TrustLevel.USER,
        TrustLevel.MODEL_GENERATED,
    ]
    assert events[0].provenance.source_role == SourceRole.SYSTEM
    assert events[1].provenance.source_role == SourceRole.DEVELOPER


def test_normalize_multiple_tool_calls_and_results() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [
                {"role": "user", "content": "Inspect files"},
                {
                    "role": "assistant",
                    "content": "Checking both files.",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "b.py"}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "print('a')"},
                {"role": "tool", "tool_call_id": "call_b", "content": "print('b')"},
            ],
        }
    )
    events, _ = normalize_request(req)
    assert [e.kind for e in events] == [
        EventKind.USER_MESSAGE,
        EventKind.ASSISTANT_MESSAGE,
        EventKind.TOOL_CALL,
        EventKind.TOOL_CALL,
        EventKind.TOOL_RESULT,
        EventKind.TOOL_RESULT,
    ]
    assert events[2].tool_call_id == "call_a"
    assert events[2].tool_arguments == {"path": "a.py"}
    assert events[2].trust == TrustLevel.MODEL_GENERATED
    assert events[4].tool_call_id == "call_a"
    assert events[4].trust == TrustLevel.UNTRUSTED_EXTERNAL
    assert events[5].tool_call_id == "call_b"
    assert events[5].trust == TrustLevel.UNTRUSTED_EXTERNAL


def test_normalize_is_deterministic() -> None:
    payload = {
        "model": "alienese-default",
        "messages": [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "Hello 世界 🚀"},
        ],
    }
    req1 = ChatCompletionRequest.model_validate(payload)
    req2 = ChatCompletionRequest.model_validate(payload)
    assert normalize_request(req1) == normalize_request(req2)


def test_normalize_unicode_and_empty_content() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [
                {"role": "user", "content": "Привет мир! こんにちは 🌍"},
                {"role": "assistant", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Part 1 — "},
                        {"type": "text", "text": "Part 2 αβγ"},
                    ],
                },
            ],
        }
    )
    events, _ = normalize_request(req)
    assert events[0].content == "Привет мир! こんにちは 🌍"
    assert events[1].content == ""
    assert events[2].content == "Part 1 — Part 2 αβγ"


def test_normalize_preserves_unknown_tools_as_custom_tool() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "Run custom linter"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "acme_proprietary_analyzer",
                        "description": "Run custom static analysis",
                        "parameters": {
                            "type": "object",
                            "properties": {"strict": {"type": "boolean", "default": True}},
                        },
                    },
                },
            ],
        }
    )
    _, tools = normalize_request(req)
    assert len(tools) == 2
    assert tools[0].external_name == "read_file"
    assert tools[0].canonical_capability == CanonicalCapability.READ_FILE
    assert tools[1].external_name == "acme_proprietary_analyzer"
    assert tools[1].canonical_capability == CanonicalCapability.CUSTOM_TOOL
    assert tools[1].description == "Run custom static analysis"
    assert tools[1].parameters_schema == {
        "type": "object",
        "properties": {"strict": {"type": "boolean", "default": True}},
    }


def test_normalize_rejects_malformed_ordering_and_arguments() -> None:
    # Orphan tool result
    req_orphan = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "tool", "tool_call_id": "call_missing", "content": "output"},
            ],
        }
    )
    with pytest.raises(ProtocolError, match="Orphan tool result"):
        normalize_request(req_orphan)

    # Duplicate tool result for same call ID
    req_dup_result = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "first"},
                {"role": "tool", "tool_call_id": "call_1", "content": "second"},
            ],
        }
    )
    with pytest.raises(ProtocolError, match="Duplicate tool result"):
        normalize_request(req_dup_result)

    # Invalid JSON in tool arguments
    req_bad_json = ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{not-json"},
                        }
                    ],
                },
            ],
        }
    )
    with pytest.raises(ProtocolError, match="Malformed JSON"):
        normalize_request(req_bad_json)


def test_unsupported_request_parameters_fail_with_compatibility_error() -> None:
    with pytest.raises(CompatibilityError, match="Streaming"):
        ChatCompletionRequest.model_validate(
            {
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            }
        )

    with pytest.raises(CompatibilityError, match="n=1"):
        ChatCompletionRequest.model_validate(
            {
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Hi"}],
                "n": 2,
            }
        )

    with pytest.raises(CompatibilityError, match="parallel_tool_calls"):
        ChatCompletionRequest.model_validate(
            {
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Hi"}],
                "parallel_tool_calls": True,
            }
        )

    with pytest.raises(CompatibilityError, match="Conflicting"):
        ChatCompletionRequest.model_validate(
            {
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
                "max_completion_tokens": 200,
            }
        )
