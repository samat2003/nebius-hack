"""OpenAI-compatible Chat Completions subset models with strict compatibility checking.

Supported request fields:
- `model` (implemented: validated non-empty string, echoed in response)
- `messages` (implemented: normalized into typed event sequence)
- `tools` (implemented: bound to ExternalToolBinding with schema preservation)
- `tool_choice` (implemented: "auto", "none", "required", or named function choice)
- `temperature` (implemented: validated [0.0, 2.0] and passed to GenerationJob)
- `max_tokens` / `max_completion_tokens` (implemented: passed to GenerationJob)
- `metadata`, `user` (documented pass-through metadata: validated, no policy effect in Phase 1)

Unsupported features fail explicitly with `CompatibilityError` (HTTP 400):
- `stream=True`
- `n != 1`
- `parallel_tool_calls=True` (Alienese enforces at most one external action per turn)
- `functions` / `function_call` (deprecated legacy OpenAI format)
- `audio`, `modalities`, `logprobs`, `top_logprobs`, `response_format`, `stop`,
  `presence_penalty`, `frequency_penalty`, `logit_bias`, `seed`, `service_tier`
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alienese.api.errors import CompatibilityError, ProtocolError

PUBLIC_MODEL_ID = "alienese-default"
SUPPORTED_PUBLIC_MODELS: frozenset[str] = frozenset({PUBLIC_MODEL_ID})


class FunctionCallInput(BaseModel):
    """Function name and JSON-encoded arguments inside an assistant tool_call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    arguments: str


class ToolCallInput(BaseModel):
    """Tool call emitted by an assistant in conversation history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: FunctionCallInput


class ContentPartText(BaseModel):
    """Text content block inside a multimodal/structured content array."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["text"]
    text: str


class ChatMessageInput(BaseModel):
    """Single message in an OpenAI-compatible Chat Completions request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ContentPartText] | None = None
    name: str | None = None
    tool_calls: list[ToolCallInput] | None = None
    tool_call_id: str | None = None

    def normalized_text_content(self) -> str:
        """Extract deterministic string content or raise ProtocolError if invalid."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content)


class FunctionDefinitionInput(BaseModel):
    """External tool function schema definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None


class ToolDefinitionInput(BaseModel):
    """External tool definition in an OpenAI Chat Completions request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["function"] = "function"
    function: FunctionDefinitionInput


class NamedFunctionChoice(BaseModel):
    """Target function name in a named tool_choice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)


class NamedToolChoice(BaseModel):
    """Explicitly requested function tool choice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["function"] = "function"
    function: NamedFunctionChoice


ToolChoiceInput = Literal["none", "auto", "required"] | NamedToolChoice


class ChatCompletionRequest(BaseModel):
    """Supported OpenAI-compatible Chat Completions request contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessageInput]
    tools: list[ToolDefinitionInput] | None = None
    tool_choice: ToolChoiceInput | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)

    # Accepted pass-through metadata (no policy effect in Phase 1, documented explicitly)
    user: str | None = None
    metadata: dict[str, str] | None = None

    # Explicitly checked compatibility fields
    stream: bool | None = None
    n: int | None = None
    parallel_tool_calls: bool | None = None
    functions: Any | None = None
    function_call: Any | None = None
    audio: Any | None = None
    modalities: Any | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    response_format: Any | None = None
    stop: Any | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: Any | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def _enforce_compatibility_and_protocol(self) -> ChatCompletionRequest:
        if self.model not in SUPPORTED_PUBLIC_MODELS:
            raise CompatibilityError(
                f"Model '{self.model}' is not supported. Supported public model: "
                f"'{PUBLIC_MODEL_ID}'.",
                param="model",
                code="model_not_supported",
            )
        if not self.messages:
            raise ProtocolError(
                "Request 'messages' must contain at least one message.",
                param="messages",
            )
        if self.stream is True:
            raise CompatibilityError(
                "Streaming ('stream=true') is not supported in Phase 1.",
                param="stream",
                code="streaming_not_supported",
            )
        if self.n is not None and self.n != 1:
            raise CompatibilityError(
                "Only 'n=1' is supported; Alienese returns at most one action per turn.",
                param="n",
                code="multiple_choices_not_supported",
            )
        if self.parallel_tool_calls is True:
            raise CompatibilityError(
                "'parallel_tool_calls=true' is not supported; Alienese enforces at most "
                "one external action per turn.",
                param="parallel_tool_calls",
                code="parallel_tool_calls_not_supported",
            )
        if self.functions is not None or self.function_call is not None:
            raise CompatibilityError(
                "Legacy 'functions' and 'function_call' parameters are not supported; "
                "use 'tools' and 'tool_choice'.",
                param="functions" if self.functions is not None else "function_call",
                code="legacy_functions_not_supported",
            )
        unsupported_optional_fields = (
            "audio",
            "modalities",
            "logprobs",
            "top_logprobs",
            "response_format",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "seed",
        )
        for field_name in unsupported_optional_fields:
            if getattr(self, field_name) is not None:
                raise CompatibilityError(
                    f"Parameter '{field_name}' is not supported in the Alienese Phase 1 subset.",
                    param=field_name,
                    code="unsupported_parameter",
                )

        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise CompatibilityError(
                "Conflicting 'max_tokens' and 'max_completion_tokens' values provided.",
                param="max_completion_tokens",
                code="conflicting_token_limits",
            )

        if self.tool_choice is not None and self.tool_choice != "none":
            has_tools = bool(self.tools)
            if self.tool_choice == "required" and not has_tools:
                raise CompatibilityError(
                    "tool_choice='required' was specified, but no 'tools' were provided.",
                    param="tool_choice",
                    code="invalid_tool_choice",
                )
            if isinstance(self.tool_choice, NamedToolChoice):
                if not self.tools:
                    raise CompatibilityError(
                        f"Named tool_choice '{self.tool_choice.function.name}' was specified, "
                        "but no 'tools' were provided.",
                        param="tool_choice",
                        code="invalid_tool_choice",
                    )
                known_names = {t.function.name for t in self.tools}
                if self.tool_choice.function.name not in known_names:
                    raise CompatibilityError(
                        f"Named tool_choice '{self.tool_choice.function.name}' does not match "
                        "any provided tool definition.",
                        param="tool_choice",
                        code="unknown_tool_choice",
                    )
        return self

    @property
    def effective_max_tokens(self) -> int | None:
        """Return the effective token limit across max_completion_tokens and max_tokens."""
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        return self.max_tokens


class FunctionCallOutput(BaseModel):
    """Serialized function call name and JSON string arguments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    arguments: str


class ToolCallOutput(BaseModel):
    """Single serialized tool call in an assistant response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    type: Literal["function"] = "function"
    function: FunctionCallOutput


class AssistantMessageOutput(BaseModel):
    """Assistant message returned in a ChatCompletionChoice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCallOutput] | None = None

    @model_validator(mode="after")
    def _enforce_single_external_action(self) -> AssistantMessageOutput:
        has_tool_calls = bool(self.tool_calls)
        if has_tool_calls:
            if self.tool_calls is not None and len(self.tool_calls) != 1:
                raise ProtocolError(
                    "Invariant violation: at most one tool_call may be returned per turn."
                )
            if self.content is not None:
                raise ProtocolError(
                    "Invariant violation: response cannot return both content and tool_calls."
                )
        elif self.content is None:
            raise ProtocolError(
                "Invariant violation: assistant response must contain either content "
                "or 1 tool_call."
            )
        return self


class ChatCompletionChoice(BaseModel):
    """Single completion choice (index 0)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = 0
    message: AssistantMessageOutput
    finish_reason: Literal["stop", "tool_calls"]


class UsageInfo(BaseModel):
    """Token usage statistics for the completion response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible Chat Completion response object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo | None = None
    system_fingerprint: str | None = None


class ModelInfo(BaseModel):
    """Single model entry returned by GET /v1/models."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    object: Literal["model"] = "model"
    created: int = 1728345600
    owned_by: str = "alienese"


class ModelListResponse(BaseModel):
    """OpenAI-compatible response for GET /v1/models."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    object: Literal["list"] = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Response contract for GET /health."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok"] = "ok"
    version: str
    provider_mode: str
