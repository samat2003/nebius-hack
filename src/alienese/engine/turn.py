"""TurnEngine: deterministic single-turn runtime orchestration.

Orchestrates:
1. Protocol normalization (`normalize_request`)
2. Deterministic `WorkingState` reconstruction (`reconstruct`)
3. Finite deterministic `CandidateAction` construction (`build_deterministic_candidates`)
4. Reserved Phase 3 retrieval hook (`_maybe_rank_candidates`, bypassed in Phase 1)
5. `Controller.decide()` policy selection + executable candidate invariant validation
6. Optional `Generator.generate()` execution for `GENERATION_JOB` candidates
7. Single external action serialization (`_serialize_response`) without mutating
   or redacting outgoing protocol tool arguments or assistant content
8. Trace emission in `TraceMode.METADATA_ONLY` by default, or
   `TraceMode.FULL_FIDELITY_REPLAY` when explicitly enabled for local replay.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from alienese.api.errors import (
    CompatibilityError,
    InvalidProviderResponse,
    InvariantViolation,
)
from alienese.api.models import (
    AssistantMessageOutput,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    FunctionCallOutput,
    NamedToolChoice,
    ToolCallOutput,
    UsageInfo,
)
from alienese.contracts.candidates import (
    CandidateAction,
    CandidateDisposition,
    CostClass,
    RiskClass,
)
from alienese.contracts.context import RequestContext
from alienese.contracts.decisions import DecisionResult
from alienese.contracts.events import NormalizedEvent
from alienese.contracts.generation import (
    GenerationJob,
    GenerationJobType,
    GenerationResult,
)
from alienese.contracts.providers import Controller, Generator, Retriever
from alienese.contracts.state import (
    CanonicalCapability,
    ExternalToolBinding,
    WorkingState,
)
from alienese.contracts.traces import (
    CandidateMetadataDigest,
    ComponentVersions,
    EventMetadataDigest,
    ReplayArtifact,
    ReplaySemantics,
    ReplayTelemetry,
    TraceMode,
)
from alienese.engine.normalize import normalize_request
from alienese.engine.reconstruct import reconstruct
from alienese.observability.logging import get_request_logger
from alienese.observability.tracing import RuntimeTracer
from alienese.storage.traces import TraceStore, sanitize_replay_artifact

MAX_SCHEMA_DEPTH = 16

_VALID_JSON_SCHEMA_TYPES: frozenset[str] = frozenset(
    {"string", "boolean", "integer", "number", "array", "object", "null"}
)

# Explicit allowlist of supported JSON Schema keywords in Phase 1.
# Any keyword outside this set (e.g., `pattern`, `format`, `uniqueItems`,
# `exclusiveMinimum`, `$ref`, `oneOf`, `anyOf`, `allOf`, `not`, etc.)
# causes schema validation to fail closed recursively.
_ALLOWED_SCHEMA_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "default",
        "description",
        "title",
        "examples",
    }
)


def classify_tool_risk(capability: CanonicalCapability) -> RiskClass:
    """Classify the risk level of a canonical tool capability conservatively.

    Unknown (`UNKNOWN`) and unclassified (`CUSTOM_TOOL`) tools are classified
    as `RiskClass.HIGH` so they are never treated as low-risk read operations.
    """
    if capability in (
        CanonicalCapability.READ_FILE,
        CanonicalCapability.SEARCH_TEXT,
        CanonicalCapability.LIST_FILES,
        CanonicalCapability.RESPOND,
        CanonicalCapability.EXPLAIN_FAILURE,
        CanonicalCapability.SYNTHESIZE_SEARCH,
        CanonicalCapability.FINISH,
    ):
        return RiskClass.LOW
    if capability in (
        CanonicalCapability.RUN_TEST,
        CanonicalCapability.WRITE_TEST,
    ):
        return RiskClass.MEDIUM
    return RiskClass.HIGH


def _matches_json_schema_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if expected_type == "array":
        return isinstance(value, (list, tuple))
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "null":
        return value is None
    return False


def _json_values_equal(left: Any, right: Any) -> bool:
    """Type-aware JSON equality (`True` != `1`, `False` != `0`)."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _json_values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_json_values_equal(left[k], right[k]) for k in left)
    return type(left) is type(right) and bool(left == right)


def _is_non_negative_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and val >= 0


def _is_finite_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _validate_schema_structure(
    schema: Any,
    *,
    path: str = "$",
    depth: int = 0,
) -> tuple[bool, str | None]:
    """Recursively validate that a JSON Schema uses only the supported keyword subset
    and is structurally well-formed at every node (even for optional/unpopulated fields).
    """
    if depth > MAX_SCHEMA_DEPTH:
        return (
            False,
            f"Schema validation exceeded maximum recursion depth ({MAX_SCHEMA_DEPTH}) at '{path}'.",
        )

    if not isinstance(schema, Mapping):
        return False, f"Schema node at '{path}' must be a JSON object."

    for raw_key in schema:
        if not isinstance(raw_key, str):
            return False, f"Schema keyword at '{path}' must be a string."
        if raw_key not in _ALLOWED_SCHEMA_KEYWORDS:
            return (
                False,
                f"Unsupported JSON Schema keyword '{raw_key}' at '{path}'.",
            )

    if "description" in schema and not isinstance(schema["description"], str):
        return False, f"Malformed 'description' at '{path}': must be a string."
    if "title" in schema and not isinstance(schema["title"], str):
        return False, f"Malformed 'title' at '{path}': must be a string."
    if "examples" in schema and not isinstance(schema["examples"], list):
        return False, f"Malformed 'examples' at '{path}': must be a list."

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if expected_type not in _VALID_JSON_SCHEMA_TYPES:
            return False, f"Unsupported schema 'type' value '{expected_type}' at '{path}'."
    elif isinstance(expected_type, list):
        if not expected_type or not all(
            isinstance(t, str) and t in _VALID_JSON_SCHEMA_TYPES for t in expected_type
        ):
            return False, f"Malformed union 'type' list at '{path}'."
    elif expected_type is not None:
        return False, f"Malformed 'type' specification at '{path}'."

    if "enum" in schema:
        enum_vals = schema["enum"]
        if not isinstance(enum_vals, list) or len(enum_vals) == 0:
            return False, f"Invalid or empty 'enum' definition at '{path}'."

    if "minLength" in schema and not _is_non_negative_int(schema["minLength"]):
        return False, f"Malformed 'minLength' at '{path}': must be a non-negative integer."
    if "maxLength" in schema and not _is_non_negative_int(schema["maxLength"]):
        return False, f"Malformed 'maxLength' at '{path}': must be a non-negative integer."
    if (
        "minLength" in schema
        and "maxLength" in schema
        and schema["minLength"] > schema["maxLength"]
    ):
        return False, f"Contradictory string bounds at '{path}': minLength > maxLength."

    if "minimum" in schema and not _is_finite_number(schema["minimum"]):
        return False, f"Malformed 'minimum' at '{path}': must be a finite number."
    if "maximum" in schema and not _is_finite_number(schema["maximum"]):
        return False, f"Malformed 'maximum' at '{path}': must be a finite number."
    if (
        "minimum" in schema
        and "maximum" in schema
        and float(schema["minimum"]) > float(schema["maximum"])
    ):
        return False, f"Contradictory numeric bounds at '{path}': minimum > maximum."

    if "minItems" in schema and not _is_non_negative_int(schema["minItems"]):
        return False, f"Malformed 'minItems' at '{path}': must be a non-negative integer."
    if "maxItems" in schema and not _is_non_negative_int(schema["maxItems"]):
        return False, f"Malformed 'maxItems' at '{path}': must be a non-negative integer."
    if "minItems" in schema and "maxItems" in schema and schema["minItems"] > schema["maxItems"]:
        return False, f"Contradictory array bounds at '{path}': minItems > maxItems."

    if "items" in schema:
        items_schema = schema["items"]
        if not isinstance(items_schema, Mapping):
            return False, f"Malformed 'items' schema at '{path}': must be a schema object."
        ok, reason = _validate_schema_structure(
            items_schema,
            path=f"{path}.items",
            depth=depth + 1,
        )
        if not ok:
            return False, reason

    if "required" in schema:
        req = schema["required"]
        if (
            not isinstance(req, list)
            or not all(isinstance(k, str) and bool(k) for k in req)
            or len(set(req)) != len(req)
        ):
            return (
                False,
                f"Malformed 'required' at '{path}': must be a list of unique non-empty strings.",
            )

    if "properties" in schema:
        props = schema["properties"]
        if not isinstance(props, Mapping):
            return False, f"Malformed 'properties' at '{path}': must be an object."
        for prop_name, prop_schema in props.items():
            if not isinstance(prop_name, str) or not prop_name:
                return False, f"Malformed property name in 'properties' at '{path}'."
            child_path = f"{path}.{prop_name}" if path != "$" else prop_name
            ok, reason = _validate_schema_structure(
                prop_schema,
                path=child_path,
                depth=depth + 1,
            )
            if not ok:
                return False, reason

    if "additionalProperties" in schema:
        add_props = schema["additionalProperties"]
        if isinstance(add_props, Mapping):
            ok, reason = _validate_schema_structure(
                add_props,
                path=f"{path}.additionalProperties",
                depth=depth + 1,
            )
            if not ok:
                return False, reason
        elif not isinstance(add_props, bool):
            return (
                False,
                f"Malformed 'additionalProperties' at '{path}': must be bool or schema object.",
            )

    return True, None


def _validate_schema_node(
    schema: Mapping[str, Any],
    value: Any,
    *,
    path: str,
    depth: int = 0,
) -> tuple[bool, str | None]:
    if depth > MAX_SCHEMA_DEPTH:
        return (
            False,
            f"Schema validation exceeded maximum recursion depth ({MAX_SCHEMA_DEPTH}) at '{path}'.",
        )

    if "const" in schema and not _json_values_equal(value, schema["const"]):
        return False, f"Argument '{path}' did not match required const value."

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if not _matches_json_schema_type(value, expected_type):
            return False, f"Argument '{path}' failed type check: expected {expected_type}."
    elif isinstance(expected_type, list) and not any(
        isinstance(t, str) and _matches_json_schema_type(value, t) for t in expected_type
    ):
        return (
            False,
            f"Argument '{path}' failed union type check: expected one of {expected_type}.",
        )

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and not any(
        _json_values_equal(value, candidate) for candidate in enum_values
    ):
        return False, f"Argument '{path}' value is not in allowed enum {enum_values}."

    if isinstance(value, str):
        min_len = schema.get("minLength")
        if isinstance(min_len, int) and len(value) < min_len:
            return False, f"Argument '{path}' shorter than minLength={min_len}."
        max_len = schema.get("maxLength")
        if isinstance(max_len, int) and len(value) > max_len:
            return False, f"Argument '{path}' exceeds maxLength={max_len}."

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return False, f"Argument '{path}' must be a finite number."
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and float(value) < float(minimum):
            return False, f"Argument '{path}' is less than minimum={minimum}."
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and float(value) > float(maximum):
            return False, f"Argument '{path}' exceeds maximum={maximum}."

    if isinstance(value, (list, tuple)):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            return False, f"Argument '{path}' has fewer than minItems={min_items}."
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            return False, f"Argument '{path}' exceeds maxItems={max_items}."
        items_schema = schema.get("items")
        if isinstance(items_schema, Mapping):
            for idx, elem in enumerate(value):
                ok, reason = _validate_schema_node(
                    items_schema,
                    elem,
                    path=f"{path}[{idx}]",
                    depth=depth + 1,
                )
                if not ok:
                    return False, reason

    if isinstance(value, Mapping):
        required_fields = schema.get("required", [])
        for req_key in required_fields:
            if req_key not in value:
                return False, f"Missing required tool argument '{req_key}'."

        properties = schema.get("properties", {})
        additional_props = schema.get("additionalProperties", True)
        if additional_props is False:
            for arg_key in value:
                if arg_key not in properties:
                    return (
                        False,
                        f"Unexpected argument '{arg_key}' (additionalProperties=false).",
                    )
        elif isinstance(additional_props, Mapping):
            for arg_key, arg_val in value.items():
                if arg_key not in properties:
                    child_path = str(arg_key) if path == "$" else f"{path}.{arg_key}"
                    ok, reason = _validate_schema_node(
                        additional_props,
                        arg_val,
                        path=child_path,
                        depth=depth + 1,
                    )
                    if not ok:
                        return False, reason

        for arg_key, arg_val in value.items():
            prop_schema = properties.get(arg_key)
            if isinstance(prop_schema, Mapping):
                child_path = str(arg_key) if path == "$" else f"{path}.{arg_key}"
                ok, reason = _validate_schema_node(
                    prop_schema,
                    arg_val,
                    path=child_path,
                    depth=depth + 1,
                )
                if not ok:
                    return False, reason

    return True, None


def validate_tool_arguments_against_schema(
    parameters_schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Validate candidate tool arguments against the explicit supported JSON Schema subset.

    Fails closed (`False`) if `parameters_schema` is malformed, exceeds
    `MAX_SCHEMA_DEPTH`, or contains any unsupported JSON Schema keyword at any
    nesting level.
    """
    if not isinstance(parameters_schema, Mapping) or not isinstance(arguments, Mapping):
        return False, "Tool parameters_schema and arguments must be JSON objects."

    struct_ok, struct_reason = _validate_schema_structure(
        parameters_schema,
        path="$",
        depth=0,
    )
    if not struct_ok:
        return False, struct_reason

    schema_type = parameters_schema.get("type", "object")
    if schema_type != "object":
        return False, f"Unsupported top-level parameters schema type '{schema_type}'."

    return _validate_schema_node(parameters_schema, arguments, path="$", depth=0)


def extract_deterministic_tool_arguments(
    binding: ExternalToolBinding,
) -> tuple[dict[str, Any], bool]:
    """Extract only safe schema-defined defaults without fabricating required values.

    Fails closed (`{}, False`) if the tool's `parameters_schema` is malformed,
    uses unsupported JSON Schema keywords anywhere in its tree, or defines an
    invalid default value.
    """
    schema = binding.parameters_schema
    if not isinstance(schema, Mapping):
        return {}, False

    struct_ok, _ = _validate_schema_structure(schema, path="$", depth=0)
    if not struct_ok:
        return {}, False

    if schema.get("type", "object") != "object":
        return {}, False

    properties = schema.get("properties", {})
    extracted: dict[str, Any] = {}
    if isinstance(properties, Mapping):
        for prop_name, prop_def in properties.items():
            if isinstance(prop_def, Mapping) and "default" in prop_def:
                default_val = copy.deepcopy(prop_def["default"])
                ok, _ = _validate_schema_node(
                    prop_def,
                    default_val,
                    path=str(prop_name),
                    depth=1,
                )
                if not ok:
                    return {}, False
                extracted[str(prop_name)] = default_val

    valid, _reason = validate_tool_arguments_against_schema(schema, extracted)
    return extracted, valid


def build_deterministic_candidates(
    state: WorkingState,
    request: ChatCompletionRequest,
) -> tuple[CandidateAction, ...]:
    """Construct a finite deterministic set of CandidateActions for the current turn."""
    tool_choice = request.tool_choice
    candidates: list[CandidateAction] = []

    answer_candidate = CandidateAction(
        candidate_id="cand_answer",
        canonical_intent=CanonicalCapability.RESPOND,
        disposition=CandidateDisposition.GENERATION_JOB,
        requires_generation=True,
        generation_job_type=GenerationJobType.ANSWER,
        risk_class=RiskClass.LOW,
        cost_class=CostClass.LOW,
        rationale="Synthesize a direct assistant response to the latest user request.",
    )

    if tool_choice == "none" or not state.available_tools:
        return (answer_candidate,)

    if isinstance(tool_choice, NamedToolChoice):
        target_name = tool_choice.function.name
        matched = [t for t in state.available_tools if t.external_name == target_name]
        if not matched:
            raise CompatibilityError(
                f"Requested tool '{target_name}' is not present in available tools.",
                param="tool_choice",
            )
        binding = matched[0]
        args, is_complete = extract_deterministic_tool_arguments(binding)
        if not is_complete:
            raise CompatibilityError(
                f"Tool '{target_name}' requires arguments that cannot be deterministically "
                "grounded in Phase 1 without fabricating values.",
                param="tool_choice",
                code="ungrounded_required_tool_arguments",
            )
        tool_risk = classify_tool_risk(binding.canonical_capability)
        return (
            CandidateAction(
                candidate_id=f"cand_tool_{binding.external_name}",
                canonical_intent=binding.canonical_capability,
                disposition=CandidateDisposition.EXTERNAL_TOOL,
                external_tool_name=binding.external_name,
                arguments=args,
                arguments_complete=True,
                requires_generation=False,
                risk_class=tool_risk,
                cost_class=CostClass.LOW,
                rationale=f"Explicitly requested tool '{binding.external_name}'.",
            ),
        )

    for binding in state.available_tools:
        args, is_complete = extract_deterministic_tool_arguments(binding)
        tool_risk = classify_tool_risk(binding.canonical_capability)
        candidates.append(
            CandidateAction(
                candidate_id=f"cand_tool_{binding.external_name}",
                canonical_intent=binding.canonical_capability,
                disposition=CandidateDisposition.EXTERNAL_TOOL,
                external_tool_name=binding.external_name,
                arguments=args,
                arguments_complete=is_complete,
                requires_generation=False,
                risk_class=tool_risk,
                cost_class=CostClass.LOW,
                rationale=f"Candidate for external tool '{binding.external_name}'.",
            )
        )

    if tool_choice == "required":
        # Conservative fake-mode policy: generic `tool_choice="required"` only
        # auto-selects low-risk tools with complete deterministic arguments.
        safe_executable_tools = [
            c for c in candidates if c.arguments_complete and c.risk_class == RiskClass.LOW
        ]
        if not safe_executable_tools:
            raise CompatibilityError(
                "tool_choice='required' was specified, but no low-risk tool has "
                "deterministically complete required arguments in Phase 1.",
                param="tool_choice",
                code="ungrounded_required_tool_arguments",
            )
        return tuple(safe_executable_tools)

    candidates.append(answer_candidate)
    return tuple(candidates)


def enforce_executable_candidate_invariant(
    selected: CandidateAction,
    state: WorkingState,
) -> ExternalToolBinding | None:
    """Enforce pre-serialization invariants on the controller-selected candidate."""
    if selected.disposition == CandidateDisposition.INTERNAL_TRANSITION:
        raise InvariantViolation(
            f"Internal transition candidate '{selected.candidate_id}' "
            f"({selected.canonical_intent}) cannot reach external serialization."
        )

    if selected.disposition == CandidateDisposition.EXTERNAL_TOOL:
        if not selected.external_tool_name:
            raise InvariantViolation(
                f"External tool candidate '{selected.candidate_id}' has no external_tool_name."
            )
        tools_by_name = {t.external_name: t for t in state.available_tools}
        binding = tools_by_name.get(selected.external_tool_name)
        if binding is None:
            raise InvariantViolation(
                f"Selected tool '{selected.external_tool_name}' does not exist in WorkingState."
            )
        if not selected.arguments_complete:
            raise InvariantViolation(
                f"Selected tool candidate '{selected.candidate_id}' has "
                "arguments_complete=False and cannot be serialized as a tool call."
            )
        is_valid, reason = validate_tool_arguments_against_schema(
            binding.parameters_schema,
            selected.arguments,
        )
        if not is_valid:
            raise InvariantViolation(
                f"Selected tool '{selected.external_tool_name}' arguments failed schema "
                f"validation: {reason}"
            )
        return binding

    return None


def _deterministic_tool_call_id(state_digest: str, candidate_id: str) -> str:
    raw = f"{state_digest}:{candidate_id}".encode()
    short_hash = hashlib.sha256(raw).hexdigest()[:16]
    return f"call_{short_hash}"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_event_digests(events: Sequence[NormalizedEvent]) -> tuple[EventMetadataDigest, ...]:
    digests: list[EventMetadataDigest] = []
    for ev in events:
        content_hash = hashlib.sha256(ev.content.encode("utf-8")).hexdigest()
        args_hash = _canonical_sha256(ev.tool_arguments) if ev.tool_arguments is not None else None
        digests.append(
            EventMetadataDigest(
                sequence_no=ev.sequence_no,
                event_id=ev.event_id,
                kind=ev.kind,
                trust=ev.trust,
                source_role=ev.provenance.source_role,
                content_sha256=content_hash,
                content_length=len(ev.content),
                tool_name=ev.tool_name,
                tool_call_id=ev.tool_call_id,
                arguments_sha256=args_hash,
            )
        )
    return tuple(digests)


def _build_candidate_digests(
    candidates: Sequence[CandidateAction],
) -> tuple[CandidateMetadataDigest, ...]:
    return tuple(
        CandidateMetadataDigest(
            candidate_id=cand.candidate_id,
            canonical_intent=cand.canonical_intent,
            disposition=cand.disposition,
            external_tool_name=cand.external_tool_name,
            arguments_complete=cand.arguments_complete,
            requires_generation=cand.requires_generation,
            generation_job_type=cand.generation_job_type,
            risk_class=cand.risk_class,
        )
        for cand in candidates
    )


def _build_generation_job(
    state: WorkingState,
    selected: CandidateAction,
    request: ChatCompletionRequest,
) -> GenerationJob:
    job_type = selected.generation_job_type or GenerationJobType.ANSWER
    job_seed = f"{state.state_digest}:{selected.candidate_id}:{job_type.value}"
    job_hash = hashlib.sha256(job_seed.encode("utf-8")).hexdigest()[:16]

    evidence: list[str] = []
    if state.latest_tool_observation is not None:
        evidence.append(state.latest_tool_observation.content)

    return GenerationJob(
        job_id=f"job_{job_hash}",
        job_type=job_type,
        candidate_id=selected.candidate_id,
        initial_user_request=state.initial_user_request,
        latest_user_request=state.latest_user_request,
        trusted_system_instructions=state.trusted_system_instructions,
        selected_evidence=tuple(evidence),
        target_tool_name=selected.external_tool_name,
        max_tokens=request.effective_max_tokens,
        temperature=request.temperature,
    )


class TurnEngine:
    """Explicit single-turn orchestration engine."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        controller: Controller,
        generator: Generator,
        trace_store: TraceStore | None = None,
        tracer: RuntimeTracer | None = None,
        include_replay_content: bool = False,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._retriever = retriever
        self._controller = controller
        self._generator = generator
        self._trace_store = trace_store
        self._tracer = tracer or RuntimeTracer()
        self._include_replay_content = include_replay_content
        self._clock: Callable[[], int] = clock or (lambda: int(time.time()))

    async def _maybe_rank_candidates(
        self,
        ctx: RequestContext,
        state: WorkingState,
        candidates: Sequence[CandidateAction],
    ) -> None:
        """Explicit Phase 3 integration point for retrieval ranking.

        In Phase 1, optional retrieval is bypassed rather than executing a
        provider call whose ranking output is not consumed by policy selection.
        """
        _ = (ctx, state, candidates, self._retriever)
        return None

    async def execute_turn(
        self,
        ctx: RequestContext,
        request: ChatCompletionRequest,
    ) -> tuple[ChatCompletionResponse, ReplayArtifact]:
        """Run a single turn from request normalization to response serialization."""
        logger = get_request_logger(ctx, "engine.turn")

        with self._tracer.start_turn_span(ctx, attributes={"model": request.model}):
            # 1. Normalize protocol messages and tool definitions
            with self._tracer.span("protocol.normalize") as norm_span:
                events, tools = normalize_request(request)
                self._tracer.set_attributes_safe(
                    norm_span,
                    {"event_count": len(events), "tool_count": len(tools)},
                )
                logger.info(
                    "protocol_normalized",
                    event_count=len(events),
                    tool_count=len(tools),
                )

            # 2. Reconstruct WorkingState deterministically
            with self._tracer.span("state.reconstruct") as state_span:
                state = reconstruct(events, available_tools=tools)
                self._tracer.set_attributes_safe(
                    state_span,
                    {
                        "state_digest": state.state_digest,
                        "event_count": state.event_count,
                    },
                )
                logger.info(
                    "state_reconstructed",
                    state_digest=state.state_digest,
                    event_count=state.event_count,
                )

            # 3. Construct finite deterministic candidate set
            with self._tracer.span("candidates.construct") as cand_span:
                candidates = build_deterministic_candidates(state, request)
                if not candidates:
                    raise InvariantViolation("Candidate construction produced an empty set.")
                self._tracer.set_attributes_safe(
                    cand_span,
                    {"candidate_count": len(candidates)},
                )
                logger.info("candidates_constructed", candidate_count=len(candidates))

            # Phase 3 retrieval hook (bypassed in Phase 1)
            await self._maybe_rank_candidates(ctx, state, candidates)

            # 4. Controller policy decision
            with self._tracer.span("policy.decide") as decide_span:
                decision: DecisionResult = await self._controller.decide(ctx, state, candidates)
                candidates_by_id = {c.candidate_id: c for c in candidates}
                selected = candidates_by_id.get(decision.selected_candidate_id)
                if selected is None:
                    raise InvalidProviderResponse(
                        f"Controller selected unknown candidate_id "
                        f"'{decision.selected_candidate_id}'."
                    )
                enforce_executable_candidate_invariant(selected, state)
                self._tracer.set_attributes_safe(
                    decide_span,
                    {
                        "selected_candidate_id": selected.candidate_id,
                        "selected_disposition": selected.disposition.value,
                    },
                )
                logger.info(
                    "controller_decided",
                    selected_candidate_id=selected.candidate_id,
                    disposition=selected.disposition.value,
                )

            # 5. Optional narrow GenerationJob execution
            gen_job: GenerationJob | None = None
            gen_result: GenerationResult | None = None
            if selected.disposition == CandidateDisposition.GENERATION_JOB:
                with self._tracer.span("generation") as gen_span:
                    gen_job = _build_generation_job(state, selected, request)
                    gen_result = await self._generator.generate(ctx, gen_job)
                    if not gen_result.content:
                        raise InvalidProviderResponse(
                            "Generator returned an empty content string for GenerationJob."
                        )
                    self._tracer.set_attributes_safe(
                        gen_span,
                        {
                            "job_id": gen_job.job_id,
                            "job_type": gen_job.job_type.value,
                        },
                    )
                    logger.info(
                        "generation_completed",
                        job_id=gen_job.job_id,
                        job_type=gen_job.job_type.value,
                    )

            # 6. Serialize exactly one external action (without redacting outgoing protocol fields)
            created_ts = self._clock()
            with self._tracer.span("protocol.serialize") as ser_span:
                response, logical_response = self._serialize_response(
                    ctx=ctx,
                    request=request,
                    state=state,
                    selected=selected,
                    gen_result=gen_result,
                    created_timestamp=created_ts,
                )
                self._tracer.set_attributes_safe(
                    ser_span,
                    {"finish_reason": response.choices[0].finish_reason},
                )
                logger.info(
                    "response_serialized",
                    finish_reason=response.choices[0].finish_reason,
                )

            finish_reason = response.choices[0].finish_reason
            event_digests = _build_event_digests(events)
            candidate_digests = _build_candidate_digests(candidates)
            response_sha256 = _canonical_sha256(logical_response)

            if self._include_replay_content:
                raw_semantics = ReplaySemantics(
                    trace_mode=TraceMode.FULL_FIDELITY_REPLAY,
                    replayable=True,
                    versions=ComponentVersions(),
                    state_digest=state.state_digest,
                    event_count=len(events),
                    tool_count=len(tools),
                    candidate_count=len(candidates),
                    event_digests=event_digests,
                    candidate_digests=candidate_digests,
                    decision_semantics=decision.semantics,
                    finish_reason=finish_reason,
                    response_sha256=response_sha256,
                    normalized_events=events,
                    available_tools=tools,
                    candidates=candidates,
                    generation_job=gen_job,
                    generation_semantics=gen_result.semantics if gen_result else None,
                    logical_response=logical_response,
                )
            else:
                raw_semantics = ReplaySemantics(
                    trace_mode=TraceMode.METADATA_ONLY,
                    replayable=False,
                    versions=ComponentVersions(),
                    state_digest=state.state_digest,
                    event_count=len(events),
                    tool_count=len(tools),
                    candidate_count=len(candidates),
                    event_digests=event_digests,
                    candidate_digests=candidate_digests,
                    decision_semantics=decision.semantics,
                    finish_reason=finish_reason,
                    response_sha256=response_sha256,
                )

            artifact = sanitize_replay_artifact(
                ReplayArtifact(
                    semantics=raw_semantics,
                    telemetry=ReplayTelemetry(
                        request_id=ctx.request_id,
                        operation_id=ctx.operation_id,
                        correlation_trace_id=ctx.correlation_trace_id,
                        created_timestamp=created_ts,
                        decision_telemetry=decision.telemetry,
                        generation_telemetry=gen_result.telemetry if gen_result else None,
                    ),
                ),
                allow_full_fidelity=self._include_replay_content,
            )

            if self._trace_store is not None:
                try:
                    artifact = await self._trace_store.save(artifact)
                except Exception:
                    logger.warning("trace_store_save_failed")

            return response, artifact

    @staticmethod
    def _serialize_response(
        *,
        ctx: RequestContext,
        request: ChatCompletionRequest,
        state: WorkingState,
        selected: CandidateAction,
        gen_result: GenerationResult | None,
        created_timestamp: int,
    ) -> tuple[ChatCompletionResponse, dict[str, Any]]:
        if selected.disposition == CandidateDisposition.EXTERNAL_TOOL:
            assert selected.external_tool_name is not None
            call_id = _deterministic_tool_call_id(state.state_digest, selected.candidate_id)
            serialized_args = json.dumps(
                selected.arguments,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            assistant_msg = AssistantMessageOutput(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCallOutput(
                        id=call_id,
                        type="function",
                        function=FunctionCallOutput(
                            name=selected.external_tool_name,
                            arguments=serialized_args,
                        ),
                    )
                ],
            )
            choice = ChatCompletionChoice(
                index=0,
                message=assistant_msg,
                finish_reason="tool_calls",
            )
            usage = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        elif selected.disposition in (
            CandidateDisposition.GENERATION_JOB,
            CandidateDisposition.ASSISTANT_RESPONSE,
        ):
            content = gen_result.content if gen_result is not None else selected.rationale
            if not content:
                raise InvariantViolation("Assistant response candidate produced empty content.")
            assistant_msg = AssistantMessageOutput(
                role="assistant",
                content=content,
                tool_calls=None,
            )
            choice = ChatCompletionChoice(
                index=0,
                message=assistant_msg,
                finish_reason="stop",
            )
            p_tok = gen_result.telemetry.prompt_tokens if gen_result else 0
            c_tok = gen_result.telemetry.completion_tokens if gen_result else 0
            usage = UsageInfo(
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=p_tok + c_tok,
            )
        else:
            raise InvariantViolation(
                f"Unsupported disposition '{selected.disposition}' reached response serialization."
            )

        response = ChatCompletionResponse(
            id=f"chatcmpl-{ctx.operation_id}",
            created=created_timestamp,
            model=request.model,
            choices=[choice],
            usage=usage,
            system_fingerprint=f"fp_{state.state_digest[:12]}",
        )
        logical_response = {
            "model": response.model,
            "choices": [c.model_dump(mode="json") for c in response.choices],
            "system_fingerprint": response.system_fingerprint,
        }
        return response, logical_response
