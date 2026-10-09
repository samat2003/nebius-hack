"""Nebius Token Factory Generator adapter for Nemotron 3 Super.

Implements the same `Generator` protocol as `NvidiaBuildGenerator` so switching
between NVIDIA API Catalog and Nebius Token Factory requires no changes to the
Alienese public API or `TurnEngine` orchestration.

Live verification status: `LIVE_VERIFIED` against
`https://api.tokenfactory.us-central1.nebius.com/v1` with model
`nvidia/nemotron-3-super-120b-a12b`.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from alienese.api.errors import CompatibilityError
from alienese.contracts.context import RequestContext
from alienese.contracts.generation import (
    GenerationJob,
    GenerationJobType,
    GenerationResult,
    GenerationSemantics,
)
from alienese.providers.generator.nvidia_build import (
    NVIDIA_NEMOTRON_SUPER_MODEL,
    build_generation_messages,
    parse_and_validate_chat_completion,
)
from alienese.providers.runtime.client import ProviderHttpClient

NEBIUS_DEFAULT_BASE_URL = "https://api.tokenfactory.us-central1.nebius.com/v1"
MAX_NEBIUS_GENERATION_TOKENS = 16_384
ALLOWED_NEBIUS_HOSTS: frozenset[str] = frozenset(
    {
        "api.tokenfactory.us-central1.nebius.com",
        "api.tokenfactory.nebius.com",
    }
)


def _is_valid_nebius_host(hostname: str, *, allow_test_hosts: bool) -> bool:
    host = hostname.lower()
    if host in ALLOWED_NEBIUS_HOSTS:
        return True
    return bool(allow_test_hosts and host in {"127.0.0.1", "localhost", "::1", "testserver"})


class NebiusTokenFactoryGenerator:
    """Operational Generator adapter backed by Nebius Token Factory."""

    def __init__(
        self,
        *,
        http_client: ProviderHttpClient,
        model_id: str = NVIDIA_NEMOTRON_SUPER_MODEL,
        default_max_tokens: int = 1024,
        default_temperature: float = 0.7,
        default_top_p: float = 0.95,
        enable_thinking: bool = False,
        allow_test_hosts: bool = False,
    ) -> None:
        self._provider_name = "nebius_token_factory"
        cleaned_model = model_id.strip()
        if not cleaned_model or cleaned_model == "nvidia/nemotron":
            raise CompatibilityError(
                "NebiusTokenFactoryGenerator requires an explicit Nebius model identifier.",
                param="generator_model",
                code="invalid_generator_model",
            )
        if not (1 <= default_max_tokens <= MAX_NEBIUS_GENERATION_TOKENS):
            raise CompatibilityError(
                f"default_max_tokens must be between 1 and {MAX_NEBIUS_GENERATION_TOKENS}.",
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
                "Nebius Nemotron thinking mode requires temperature > 0.0.",
                param="temperature",
                code="unsupported_temperature_for_thinking",
            )

        parsed_host = (urlparse(http_client.base_url).hostname or "").lower()
        if not _is_valid_nebius_host(parsed_host, allow_test_hosts=allow_test_hosts):
            raise CompatibilityError(
                "NebiusTokenFactoryGenerator requires a Nebius Token Factory origin "
                f"('api.tokenfactory.us-central1.nebius.com' or 'api.tokenfactory.nebius.com'), "
                f"got host '{parsed_host}'.",
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
        """Execute a narrow `ANSWER` synthesis job via Nebius Token Factory."""
        if job.job_type != GenerationJobType.ANSWER:
            raise CompatibilityError(
                f"NebiusTokenFactoryGenerator in Phase 2 supports only "
                f"GenerationJobType.ANSWER (got '{job.job_type.value}').",
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
                "Nebius Nemotron thinking mode requires temperature > 0.0.",
                param="temperature",
                code="unsupported_temperature_for_thinking",
            )

        max_tokens = job.max_tokens if job.max_tokens is not None else self._default_max_tokens
        if not (1 <= max_tokens <= MAX_NEBIUS_GENERATION_TOKENS):
            raise CompatibilityError(
                f"Requested max_tokens {max_tokens} exceeds Nebius adapter bounds "
                f"[1, {MAX_NEBIUS_GENERATION_TOKENS}].",
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
            "chat_template_kwargs": {"enable_thinking": self._enable_thinking},
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
