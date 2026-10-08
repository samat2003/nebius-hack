"""NVIDIA API Catalog (`build.nvidia.com`) Generator adapter for Nemotron 3 Super.

Enforces:
- Strict NVIDIA origin verification (`integrate.api.nvidia.com` / `*.api.nvidia.com`).
- Trust boundary separation (`trusted_system_instructions` in `system`, untrusted
  `selected_evidence` fenced inside `user`).
- Single consistent reasoning configuration (`chat_template_kwargs={"enable_thinking": ...}`).
- Strict response validation (single choice, `finish_reason == "stop"`, verbatim
  `content` preservation, rejection of reasoning-only or empty outputs, and
  model-mismatch detection).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from alienese.api.errors import CompatibilityError, InvalidProviderResponse
from alienese.contracts.context import RequestContext
from alienese.contracts.generation import (
    GenerationJob,
    GenerationJobType,
    GenerationResult,
    GenerationSemantics,
)
from alienese.providers.runtime.client import ProviderHttpClient

NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_NEMOTRON_SUPER_MODEL = "nvidia/nemotron-3-super-120b-a12b"
MAX_NVIDIA_GENERATION_TOKENS = 16_384


def _is_valid_nvidia_host(hostname: str, *, allow_test_hosts: bool) -> bool:
    host = hostname.lower()
    if host == "integrate.api.nvidia.com" or host.endswith(".api.nvidia.com"):
        return True
    return bool(allow_test_hosts and host in {"127.0.0.1", "localhost", "::1", "testserver"})


def build_generation_messages(job: GenerationJob) -> list[dict[str, str]]:
    """Construct chat completion messages preserving Alienese trust boundaries.

    - `trusted_system_instructions` are placed in the `system` role message.
    - `selected_evidence` (`UNTRUSTED_EXTERNAL`) is fenced inside the `user`
      message and never promoted to `system` instructions.
    """
    messages: list[dict[str, str]] = []

    if job.trusted_system_instructions:
        joined_system = "\n\n".join(
            instr.strip() for instr in job.trusted_system_instructions if instr.strip()
        )
        if joined_system:
            messages.append({"role": "system", "content": joined_system})

    user_prompt = (job.latest_user_request or job.initial_user_request or "").strip()
    if not user_prompt:
        raise CompatibilityError(
            "GenerationJob requires a non-empty user request.",
            param="messages",
            code="empty_generation_prompt",
        )

    if job.selected_evidence:
        evidence_blocks = "\n---\n".join(ev.strip() for ev in job.selected_evidence if ev.strip())
        if evidence_blocks:
            user_content = (
                f"{user_prompt}\n\n"
                "<untrusted_external_evidence>\n"
                f"{evidence_blocks}\n"
                "</untrusted_external_evidence>"
            )
        else:
            user_content = user_prompt
    else:
        user_content = user_prompt

    messages.append({"role": "user", "content": user_content})
    return messages


def parse_and_validate_chat_completion(
    *,
    provider_name: str,
    expected_model_id: str,
    data: Mapping[str, Any],
) -> tuple[str, str]:
    """Validate an OpenAI-compatible chat completion response and return `(content, revision)`.

    Raises `InvalidProviderResponse` on:
    - Model mismatch
    - Missing or multiple `choices`
    - Missing `message` or non-assistant `role`
    - Non-`stop` `finish_reason` (including `"length"` truncation)
    - Empty or reasoning-only output (`content` null/whitespace while `reasoning_content` present)
    """
    reported_model = data.get("model")
    if (
        isinstance(reported_model, str)
        and reported_model.strip()
        and reported_model.strip() != expected_model_id
    ):
        raise InvalidProviderResponse(
            f"Provider '{provider_name}' returned model '{reported_model.strip()}', "
            f"expected '{expected_model_id}'.",
            code="provider_model_mismatch",
        )

    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        count = len(choices) if isinstance(choices, list) else "non-list"
        raise InvalidProviderResponse(
            f"Provider '{provider_name}' must return exactly 1 choice (got {count}).",
            code="invalid_choices_count",
        )

    choice0 = choices[0]
    if not isinstance(choice0, Mapping):
        raise InvalidProviderResponse(
            f"Provider '{provider_name}' choice[0] must be a JSON object.",
            code="invalid_provider_response",
        )

    finish_reason = choice0.get("finish_reason")
    if finish_reason == "length":
        raise InvalidProviderResponse(
            f"Provider '{provider_name}' truncated generation at token limit "
            "(finish_reason='length').",
            code="truncated_provider_response",
        )
    if finish_reason != "stop":
        raise InvalidProviderResponse(
            f"Provider '{provider_name}' returned unexpected finish_reason '{finish_reason}'.",
            code="unexpected_finish_reason",
        )

    message = choice0.get("message")
    if not isinstance(message, Mapping):
        raise InvalidProviderResponse(
            f"Provider '{provider_name}' choice[0].message is missing or invalid.",
            code="missing_assistant_message",
        )

    role = message.get("role")
    if role != "assistant":
        raise InvalidProviderResponse(
            f"Provider '{provider_name}' choice[0].message.role must be 'assistant' "
            f"(got '{role}').",
            code="invalid_assistant_role",
        )

    raw_content = message.get("content")
    has_reasoning = bool(
        (isinstance(message.get("reasoning_content"), str) and message["reasoning_content"].strip())
        or (isinstance(message.get("reasoning"), str) and message["reasoning"].strip())
    )

    if not isinstance(raw_content, str) or not raw_content.strip():
        if has_reasoning:
            raise InvalidProviderResponse(
                f"Provider '{provider_name}' returned reasoning-only output with empty "
                "final assistant content.",
                code="reasoning_only_response",
            )
        raise InvalidProviderResponse(
            f"Provider '{provider_name}' returned an empty assistant content string.",
            code="empty_generator_content",
        )

    raw_fp = data.get("system_fingerprint")
    revision = raw_fp.strip() if isinstance(raw_fp, str) and raw_fp.strip() else "unknown"
    return raw_content, revision


class NvidiaBuildGenerator:
    """Operational Generator adapter backed by NVIDIA API Catalog (`build.nvidia.com`)."""

    def __init__(
        self,
        *,
        http_client: ProviderHttpClient,
        model_id: str = NVIDIA_NEMOTRON_SUPER_MODEL,
        default_max_tokens: int = 1024,
        default_temperature: float = 1.0,
        default_top_p: float = 0.95,
        enable_thinking: bool = False,
        allow_test_hosts: bool = False,
    ) -> None:
        self._provider_name = "nvidia_build"
        cleaned_model = model_id.strip()
        if not cleaned_model or cleaned_model == "nvidia/nemotron":
            raise CompatibilityError(
                "NvidiaBuildGenerator requires an explicit NVIDIA API Catalog model identifier "
                f"(e.g. '{NVIDIA_NEMOTRON_SUPER_MODEL}').",
                param="generator_model",
                code="invalid_generator_model",
            )
        if not (1 <= default_max_tokens <= MAX_NVIDIA_GENERATION_TOKENS):
            raise CompatibilityError(
                f"default_max_tokens must be between 1 and {MAX_NVIDIA_GENERATION_TOKENS}.",
                param="generator_max_tokens",
                code="unsupported_max_tokens",
            )
        if not (0.0 <= default_temperature <= 2.0):
            raise CompatibilityError(
                "default_temperature must be between 0.0 and 2.0.",
                param="temperature",
                code="unsupported_temperature",
            )
        if enable_thinking and default_temperature <= 0.0:
            raise CompatibilityError(
                "NVIDIA Nemotron thinking mode requires temperature > 0.0.",
                param="temperature",
                code="unsupported_temperature_for_thinking",
            )

        parsed_host = (urlparse(http_client.base_url).hostname or "").lower()
        if not _is_valid_nvidia_host(parsed_host, allow_test_hosts=allow_test_hosts):
            raise CompatibilityError(
                f"NvidiaBuildGenerator requires an NVIDIA API Catalog origin "
                f"('integrate.api.nvidia.com'), got host '{parsed_host}'.",
                param="generator_base_url",
                code="provider_origin_mismatch",
            )

        self._http = http_client
        self._model_id = cleaned_model
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self._default_top_p = default_top_p
        self._enable_thinking = enable_thinking

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_id(self) -> str:
        return self._model_id

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def generate(
        self,
        ctx: RequestContext,
        job: GenerationJob,
    ) -> GenerationResult:
        """Execute a narrow `ANSWER` synthesis job via NVIDIA API Catalog."""
        if job.job_type != GenerationJobType.ANSWER:
            raise CompatibilityError(
                f"NvidiaBuildGenerator in Phase 2 supports only GenerationJobType.ANSWER "
                f"(got '{job.job_type.value}').",
                param="job_type",
                code="unsupported_generation_job_type",
            )

        temperature = job.temperature if job.temperature is not None else self._default_temperature
        if not (0.0 <= temperature <= 2.0):
            raise CompatibilityError(
                f"Requested temperature {temperature} is outside supported range [0.0, 2.0].",
                param="temperature",
                code="unsupported_temperature",
            )
        if self._enable_thinking and temperature <= 0.0:
            raise CompatibilityError(
                "NVIDIA Nemotron thinking mode does not support greedy temperature=0.0.",
                param="temperature",
                code="unsupported_temperature_for_thinking",
            )

        max_tokens = job.max_tokens if job.max_tokens is not None else self._default_max_tokens
        if not (1 <= max_tokens <= MAX_NVIDIA_GENERATION_TOKENS):
            raise CompatibilityError(
                f"Requested max_tokens {max_tokens} exceeds NVIDIA adapter bounds "
                f"[1, {MAX_NVIDIA_GENERATION_TOKENS}].",
                param="max_tokens",
                code="unsupported_max_tokens",
            )

        messages = build_generation_messages(job)
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": self._default_top_p,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": self._enable_thinking,
            },
        }

        http_resp = await self._http.post_json(
            ctx,
            "/chat/completions",
            payload,
        )
        content, revision = parse_and_validate_chat_completion(
            provider_name=self._provider_name,
            expected_model_id=self._model_id,
            data=http_resp.data,
        )

        return GenerationResult(
            semantics=GenerationSemantics(
                job_id=job.job_id,
                job_type=job.job_type,
                content=content,
                structured_arguments=None,
                provider_name=self._provider_name,
                model_id=self._model_id,
                model_revision=revision,
            ),
            telemetry=http_resp.telemetry,
        )
