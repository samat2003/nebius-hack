"""Single-turn deterministic orchestration engine for Alienese.

Enforces core runtime invariants:
1. One external action per turn (either 1 assistant response OR 1 tool call).
2. No model-to-model orchestration or recursive loops.
3. Never fabricates tool arguments; validates `arguments_complete` and JSON Schema
   before serializing any external tool call.
4. Internal transitions (`CandidateDisposition.INTERNAL_TRANSITION`) can never
   reach external serialization.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
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
from alienese.contracts.generation import (
    GenerationJob,
    GenerationJobType,
    GenerationResult,
)
from alienese.contracts.providers import (
    Controller,
    Generator,
    RetrievalCandidateItem,
    RetrievalRequest,
    Retriever,
)
from alienese.contracts.state import (
    CanonicalCapability,
    ExternalToolBinding,
    WorkingState,
)
from alienese.contracts.traces import (
    ComponentVersions,
    ReplayArtifact,
    ReplaySemantics,
    ReplayTelemetry,
)
from alienese.engine.normalize import normalize_request
from alienese.engine.reconstruct import reconstruct
from alienese.observability.logging import get_request_logger
from alienese.observability.tracing import RuntimeTracer
from alienese.storage.traces import TraceStore, sanitize_replay_artifact

_FIXED_LOGICAL_CREATED_EPOCH = 1728345600


def _matches_json_schema_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, (list, tuple))
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "null":
        return value is None
    return False


def validate_tool_arguments_against_schema(
    parameters_schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Validate candidate tool arguments against the supported JSON Schema subset."""
    if not parameters_schema:
        return True, None

    schema_type = parameters_schema.get("type", "object")
    if schema_type != "object":
        return False, f"Unsupported top-level parameters schema type '{schema_type}'."

    required_fields = parameters_schema.get("required", [])
    if not isinstance(required_fields, list):
        return False, "Schema 'required' field must be a list."

    for req_key in required_fields:
        if req_key not in arguments:
            return False, f"Missing required tool argument '{req_key}'."

    properties = parameters_schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return False, "Schema 'properties' field must be an object."

    additional_props = parameters_schema.get("additionalProperties", True)
    if additional_props is False:
        for arg_key in arguments:
            if arg_key not in properties:
                return False, f"Unexpected argument '{arg_key}' (additionalProperties=false)."

    for arg_key, arg_val in arguments.items():
        prop_schema = properties.get(arg_key)
        if not isinstance(prop_schema, Mapping):
            continue
        expected_type = prop_schema.get("type")
        if isinstance(expected_type, str) and not _matches_json_schema_type(arg_val, expected_type):
            return (
                False,
                f"Argument '{arg_key}' failed type check: expected {expected_type}.",
            )
        if isinstance(expected_type, list) and not any(
            isinstance(t, str) and _matches_json_schema_type(arg_val, t) for t in expected_type
        ):
            return (
                False,
                f"Argument '{arg_key}' failed union type check: expected one of {expected_type}.",
            )
        enum_values = prop_schema.get("enum")
        if isinstance(enum_values, list) and arg_val not in enum_values:
            return (
                False,
                f"Argument '{arg_key}' value is not in allowed enum {enum_values}.",
            )

    return True, None


def extract_deterministic_tool_arguments(
    binding: ExternalToolBinding,
) -> tuple[dict[str, Any], bool]:
    """Extract only safe schema-defined defaults without fabricating required values."""
    schema = binding.parameters_schema
    if not schema:
        return {}, True

    properties = schema.get("properties", {})
    extracted: dict[str, Any] = {}
    if isinstance(properties, Mapping):
        for prop_name, prop_def in properties.items():
            if isinstance(prop_def, Mapping) and "default" in prop_def:
                default_val = prop_def["default"]
                expected_type = prop_def.get("type")
                if not isinstance(expected_type, str) or _matches_json_schema_type(
                    default_val, expected_type
                ):
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
        return (
            CandidateAction(
                candidate_id=f"cand_tool_{binding.external_name}",
                canonical_intent=binding.canonical_capability,
                disposition=CandidateDisposition.EXTERNAL_TOOL,
                external_tool_name=binding.external_name,
                arguments=args,
                arguments_complete=True,
                requires_generation=False,
                risk_class=RiskClass.LOW,
                cost_class=CostClass.LOW,
                rationale=f"Explicitly requested tool '{binding.external_name}'.",
            ),
        )

    for binding in state.available_tools:
        args, is_complete = extract_deterministic_tool_arguments(binding)
        candidates.append(
            CandidateAction(
                candidate_id=f"cand_tool_{binding.external_name}",
                canonical_intent=binding.canonical_capability,
                disposition=CandidateDisposition.EXTERNAL_TOOL,
                external_tool_name=binding.external_name,
                arguments=args,
                arguments_complete=is_complete,
                requires_generation=False,
                risk_class=RiskClass.LOW,
                cost_class=CostClass.LOW,
                rationale=f"Candidate for external tool '{binding.external_name}'.",
            )
        )

    if tool_choice == "required":
        executable_tools = [c for c in candidates if c.arguments_complete]
        if not executable_tools:
            raise CompatibilityError(
                "tool_choice='required' was specified, but no provided tool has "
                "deterministically complete required arguments in Phase 1.",
                param="tool_choice",
                code="ungrounded_required_tool_arguments",
            )
        return tuple(executable_tools)

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
    ) -> None:
        self._retriever = retriever
        self._controller = controller
        self._generator = generator
        self._trace_store = trace_store
        self._tracer = tracer or RuntimeTracer()

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

            # Optional retrieval ranking over candidates when > 1 candidate
            if len(candidates) > 1:
                retrieval_items = tuple(
                    RetrievalCandidateItem(
                        item_id=c.candidate_id,
                        content=f"{c.canonical_intent.value}:{c.external_tool_name or ''}",
                        mandatory=c.disposition == CandidateDisposition.GENERATION_JOB,
                    )
                    for c in candidates
                )
                await self._retriever.rank(
                    ctx,
                    RetrievalRequest(
                        query=state.latest_user_request or "",
                        items=retrieval_items,
                        max_results=len(candidates),
                    ),
                )

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

            # 6. Serialize exactly one external action
            with self._tracer.span("protocol.serialize") as ser_span:
                response, logical_response = self._serialize_response(
                    ctx=ctx,
                    request=request,
                    state=state,
                    selected=selected,
                    gen_result=gen_result,
                )
                self._tracer.set_attributes_safe(
                    ser_span,
                    {"finish_reason": response.choices[0].finish_reason},
                )
                logger.info(
                    "response_serialized",
                    finish_reason=response.choices[0].finish_reason,
                )

            artifact = sanitize_replay_artifact(
                ReplayArtifact(
                    semantics=ReplaySemantics(
                        versions=ComponentVersions(),
                        normalized_events=events,
                        available_tools=tools,
                        state_digest=state.state_digest,
                        candidates=candidates,
                        decision_semantics=decision.semantics,
                        generation_job=gen_job,
                        generation_semantics=gen_result.semantics if gen_result else None,
                        logical_response=logical_response,
                    ),
                    telemetry=ReplayTelemetry(
                        request_id=ctx.request_id,
                        operation_id=ctx.operation_id,
                        correlation_trace_id=ctx.correlation_trace_id,
                        decision_telemetry=decision.telemetry,
                        generation_telemetry=gen_result.telemetry if gen_result else None,
                    ),
                )
            )

            if self._trace_store is not None:
                try:
                    await self._trace_store.save(artifact)
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
    ) -> tuple[ChatCompletionResponse, dict[str, Any]]:
        if selected.disposition == CandidateDisposition.EXTERNAL_TOOL:
            assert selected.external_tool_name is not None
            call_id = _deterministic_tool_call_id(state.state_digest, selected.candidate_id)
            serialized_args = json.dumps(
                selected.arguments,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
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
            created=_FIXED_LOGICAL_CREATED_EPOCH,
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
