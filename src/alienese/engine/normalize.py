"""Deterministic protocol normalization into immutable typed events.

Enforces:
- 4-tier trust boundary (SYSTEM_TRUSTED, USER, MODEL_GENERATED, UNTRUSTED_EXTERNAL)
- Strict tool-call / tool-result ordering and ID uniqueness
- Explicit deterministic capability mapping that preserves unknown tools as CUSTOM_TOOL
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from alienese.api.errors import ProtocolError
from alienese.api.models import ChatCompletionRequest, ChatMessageInput, ToolDefinitionInput
from alienese.contracts.events import (
    EventKind,
    EventProvenance,
    NormalizedEvent,
    SourceRole,
    TrustLevel,
)
from alienese.contracts.state import CanonicalCapability, ExternalToolBinding

# Small explicit deterministic registry for known canonical tool names.
# Unknown external tools are preserved as CUSTOM_TOOL rather than guessed as RUN_COMMAND.
KNOWN_TOOL_CAPABILITIES: dict[str, CanonicalCapability] = {
    "read_file": CanonicalCapability.READ_FILE,
    "search_text": CanonicalCapability.SEARCH_TEXT,
    "list_files": CanonicalCapability.LIST_FILES,
    "run_command": CanonicalCapability.RUN_COMMAND,
    "run_test": CanonicalCapability.RUN_TEST,
    "apply_patch": CanonicalCapability.APPLY_PATCH,
    "write_file": CanonicalCapability.WRITE_FILE,
}


def resolve_tool_capability(external_name: str) -> CanonicalCapability:
    """Resolve an external tool name using the explicit deterministic registry."""
    normalized = external_name.strip().lower()
    return KNOWN_TOOL_CAPABILITIES.get(normalized, CanonicalCapability.CUSTOM_TOOL)


def normalize_tools(
    tools: Sequence[ToolDefinitionInput] | None,
) -> tuple[ExternalToolBinding, ...]:
    """Normalize external tool definitions while preserving schema and unknown tool names."""
    if not tools:
        return ()

    seen_names: set[str] = set()
    bindings: list[ExternalToolBinding] = []

    for idx, tool_def in enumerate(tools):
        fn = tool_def.function
        name = fn.name.strip()
        if not name:
            raise ProtocolError(
                f"Tool definition at index {idx} has an empty function name.",
                param=f"tools[{idx}].function.name",
            )
        if name in seen_names:
            raise ProtocolError(
                f"Duplicate tool definition name '{name}'.",
                param=f"tools[{idx}].function.name",
            )
        seen_names.add(name)
        schema = dict(fn.parameters) if fn.parameters else {}
        if schema and not isinstance(schema.get("type", "object"), str):
            raise ProtocolError(
                f"Tool '{name}' parameters schema has invalid 'type'.",
                param=f"tools[{idx}].function.parameters",
            )
        bindings.append(
            ExternalToolBinding(
                external_name=name,
                description=fn.description,
                parameters_schema=schema,
                canonical_capability=resolve_tool_capability(name),
            )
        )

    return tuple(bindings)


def _compute_event_id(
    *,
    sequence_no: int,
    kind: EventKind,
    provenance: EventProvenance,
    content: str,
    tool_arguments: dict[str, Any] | None,
) -> str:
    payload = {
        "seq": sequence_no,
        "kind": kind.value,
        "msg_idx": provenance.message_index,
        "sub_idx": provenance.sub_index,
        "role": provenance.source_role.value,
        "tool_call_id": provenance.external_tool_call_id,
        "tool_name": provenance.tool_name,
        "content": content,
        "args": tool_arguments,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"evt_{sequence_no:04d}_{digest}"


def _parse_tool_arguments(
    raw_arguments: str, *, message_index: int, sub_index: int
) -> dict[str, Any]:
    if not raw_arguments.strip():
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            f"Malformed JSON in tool_calls[{sub_index}] at messages[{message_index}]: {exc.msg}",
            param=f"messages[{message_index}].tool_calls[{sub_index}].function.arguments",
        ) from exc
    if not isinstance(parsed, dict):
        raise ProtocolError(
            f"Tool arguments at messages[{message_index}].tool_calls[{sub_index}] "
            "must decode to a JSON object.",
            param=f"messages[{message_index}].tool_calls[{sub_index}].function.arguments",
        )
    return parsed


def normalize_messages(
    messages: Sequence[ChatMessageInput],
) -> tuple[NormalizedEvent, ...]:
    """Deterministically normalize OpenAI protocol messages into immutable events."""
    if not messages:
        raise ProtocolError("Request 'messages' cannot be empty.", param="messages")

    events: list[NormalizedEvent] = []
    seen_tool_call_ids: set[str] = set()
    pending_tool_calls: dict[str, str] = {}
    sequence_no = 0

    for msg_idx, msg in enumerate(messages):
        role = SourceRole(msg.role)
        text_content = msg.normalized_text_content()

        if role in (SourceRole.SYSTEM, SourceRole.DEVELOPER):
            if msg.tool_calls:
                raise ProtocolError(
                    f"Message role '{role.value}' at index {msg_idx} cannot contain tool_calls.",
                    param=f"messages[{msg_idx}].tool_calls",
                )
            if msg.tool_call_id is not None:
                raise ProtocolError(
                    f"Message role '{role.value}' at index {msg_idx} cannot contain tool_call_id.",
                    param=f"messages[{msg_idx}].tool_call_id",
                )
            provenance = EventProvenance(
                message_index=msg_idx,
                sub_index=0,
                source_role=role,
            )
            events.append(
                NormalizedEvent(
                    sequence_no=sequence_no,
                    event_id=_compute_event_id(
                        sequence_no=sequence_no,
                        kind=EventKind.SYSTEM_MESSAGE,
                        provenance=provenance,
                        content=text_content,
                        tool_arguments=None,
                    ),
                    kind=EventKind.SYSTEM_MESSAGE,
                    trust=TrustLevel.SYSTEM_TRUSTED,
                    provenance=provenance,
                    content=text_content,
                )
            )
            sequence_no += 1

        elif role == SourceRole.USER:
            if msg.tool_calls:
                raise ProtocolError(
                    f"User message at index {msg_idx} cannot contain tool_calls.",
                    param=f"messages[{msg_idx}].tool_calls",
                )
            if msg.tool_call_id is not None:
                raise ProtocolError(
                    f"User message at index {msg_idx} cannot contain tool_call_id.",
                    param=f"messages[{msg_idx}].tool_call_id",
                )
            provenance = EventProvenance(
                message_index=msg_idx,
                sub_index=0,
                source_role=SourceRole.USER,
            )
            events.append(
                NormalizedEvent(
                    sequence_no=sequence_no,
                    event_id=_compute_event_id(
                        sequence_no=sequence_no,
                        kind=EventKind.USER_MESSAGE,
                        provenance=provenance,
                        content=text_content,
                        tool_arguments=None,
                    ),
                    kind=EventKind.USER_MESSAGE,
                    trust=TrustLevel.USER,
                    provenance=provenance,
                    content=text_content,
                )
            )
            sequence_no += 1

        elif role == SourceRole.ASSISTANT:
            if msg.tool_call_id is not None:
                raise ProtocolError(
                    f"Assistant message at index {msg_idx} cannot contain tool_call_id.",
                    param=f"messages[{msg_idx}].tool_call_id",
                )
            sub_idx = 0
            has_tool_calls = bool(msg.tool_calls)
            if text_content or not has_tool_calls:
                provenance = EventProvenance(
                    message_index=msg_idx,
                    sub_index=sub_idx,
                    source_role=SourceRole.ASSISTANT,
                )
                events.append(
                    NormalizedEvent(
                        sequence_no=sequence_no,
                        event_id=_compute_event_id(
                            sequence_no=sequence_no,
                            kind=EventKind.ASSISTANT_MESSAGE,
                            provenance=provenance,
                            content=text_content,
                            tool_arguments=None,
                        ),
                        kind=EventKind.ASSISTANT_MESSAGE,
                        trust=TrustLevel.MODEL_GENERATED,
                        provenance=provenance,
                        content=text_content,
                    )
                )
                sequence_no += 1
                sub_idx += 1

            if msg.tool_calls:
                for call_idx, tool_call in enumerate(msg.tool_calls):
                    tc_id = tool_call.id.strip()
                    tc_name = tool_call.function.name.strip()
                    if not tc_id:
                        raise ProtocolError(
                            f"Empty tool_call id at messages[{msg_idx}].tool_calls[{call_idx}].",
                            param=f"messages[{msg_idx}].tool_calls[{call_idx}].id",
                        )
                    if tc_id in seen_tool_call_ids:
                        raise ProtocolError(
                            f"Duplicate tool_call_id '{tc_id}' at messages[{msg_idx}].",
                            param=f"messages[{msg_idx}].tool_calls[{call_idx}].id",
                        )
                    seen_tool_call_ids.add(tc_id)
                    pending_tool_calls[tc_id] = tc_name
                    parsed_args = _parse_tool_arguments(
                        tool_call.function.arguments,
                        message_index=msg_idx,
                        sub_index=call_idx,
                    )
                    provenance = EventProvenance(
                        message_index=msg_idx,
                        sub_index=sub_idx,
                        source_role=SourceRole.ASSISTANT,
                        external_tool_call_id=tc_id,
                        tool_name=tc_name,
                    )
                    events.append(
                        NormalizedEvent(
                            sequence_no=sequence_no,
                            event_id=_compute_event_id(
                                sequence_no=sequence_no,
                                kind=EventKind.TOOL_CALL,
                                provenance=provenance,
                                content="",
                                tool_arguments=parsed_args,
                            ),
                            kind=EventKind.TOOL_CALL,
                            trust=TrustLevel.MODEL_GENERATED,
                            provenance=provenance,
                            content="",
                            tool_name=tc_name,
                            tool_call_id=tc_id,
                            tool_arguments=parsed_args,
                        )
                    )
                    sequence_no += 1
                    sub_idx += 1

        elif role == SourceRole.TOOL:
            if msg.tool_calls:
                raise ProtocolError(
                    f"Tool message at index {msg_idx} cannot contain tool_calls.",
                    param=f"messages[{msg_idx}].tool_calls",
                )
            if not msg.tool_call_id or not msg.tool_call_id.strip():
                raise ProtocolError(
                    f"Tool message at index {msg_idx} is missing required 'tool_call_id'.",
                    param=f"messages[{msg_idx}].tool_call_id",
                )
            tc_id = msg.tool_call_id.strip()
            if tc_id not in pending_tool_calls:
                if tc_id in seen_tool_call_ids:
                    raise ProtocolError(
                        f"Duplicate tool result for already-resolved tool_call_id '{tc_id}' "
                        f"at messages[{msg_idx}].",
                        param=f"messages[{msg_idx}].tool_call_id",
                    )
                raise ProtocolError(
                    f"Orphan tool result referencing unknown tool_call_id '{tc_id}' "
                    f"at messages[{msg_idx}].",
                    param=f"messages[{msg_idx}].tool_call_id",
                )
            origin_tool_name = pending_tool_calls.pop(tc_id)
            effective_tool_name = (
                msg.name.strip() if msg.name and msg.name.strip() else origin_tool_name
            )
            provenance = EventProvenance(
                message_index=msg_idx,
                sub_index=0,
                source_role=SourceRole.TOOL,
                external_tool_call_id=tc_id,
                tool_name=effective_tool_name,
            )
            events.append(
                NormalizedEvent(
                    sequence_no=sequence_no,
                    event_id=_compute_event_id(
                        sequence_no=sequence_no,
                        kind=EventKind.TOOL_RESULT,
                        provenance=provenance,
                        content=text_content,
                        tool_arguments=None,
                    ),
                    kind=EventKind.TOOL_RESULT,
                    trust=TrustLevel.UNTRUSTED_EXTERNAL,
                    provenance=provenance,
                    content=text_content,
                    tool_name=effective_tool_name,
                    tool_call_id=tc_id,
                )
            )
            sequence_no += 1

    return tuple(events)


def normalize_request(
    request: ChatCompletionRequest,
) -> tuple[tuple[NormalizedEvent, ...], tuple[ExternalToolBinding, ...]]:
    """Normalize both messages and tools from a ChatCompletionRequest."""
    events = normalize_messages(request.messages)
    tools = normalize_tools(request.tools)
    return events, tools
