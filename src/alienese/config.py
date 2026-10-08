"""Environment-driven typed configuration for the Alienese runtime.

Provider API keys use SecretStr so that accidental serialization or logging
never exposes credentials. Fake providers and loopback-only binding are used
by default in development and test environments.
"""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from alienese.api.errors import CompatibilityError
from alienese.observability.redaction import REDACTED_PLACEHOLDER, redact_mapping
from alienese.providers.generator.nebius_token_factory import NEBIUS_DEFAULT_BASE_URL
from alienese.providers.generator.nvidia_build import (
    NVIDIA_DEFAULT_BASE_URL,
    NVIDIA_NEMOTRON_SUPER_MODEL,
)

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
_SUPPORTED_PROVIDER_MODES: frozenset[str] = frozenset({"fake", "hybrid"})


class Settings(BaseSettings):
    """Typed runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    # Serving environment (defaults strictly to loopback-only)
    alienese_env: Literal["development", "test", "production"] = Field(
        default="development",
        validation_alias="ALIENESE_ENV",
    )
    alienese_host: str = Field(
        default="127.0.0.1",
        validation_alias="ALIENESE_HOST",
    )
    alienese_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        validation_alias="ALIENESE_PORT",
    )
    alienese_allow_non_loopback: bool = Field(
        default=False,
        validation_alias="ALIENESE_ALLOW_NON_LOOPBACK",
    )
    alienese_log_level: str = Field(
        default="INFO",
        validation_alias="ALIENESE_LOG_LEVEL",
    )
    alienese_trace_content: bool = Field(
        default=False,
        validation_alias="ALIENESE_TRACE_CONTENT",
    )
    alienese_provider_mode: str = Field(
        default="fake",
        validation_alias="ALIENESE_PROVIDER_MODE",
    )

    # Resource bounds
    max_request_body_bytes: int = Field(
        default=1_048_576,
        ge=1024,
        le=10_485_760,
        validation_alias="ALIENESE_MAX_REQUEST_BODY_BYTES",
    )
    idempotency_max_entries: int = Field(
        default=1024,
        ge=1,
        validation_alias="ALIENESE_IDEMPOTENCY_MAX_ENTRIES",
    )
    idempotency_ttl_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        validation_alias="ALIENESE_IDEMPOTENCY_TTL_SECONDS",
    )
    trace_store_max_entries: int = Field(
        default=1024,
        ge=1,
        validation_alias="ALIENESE_TRACE_STORE_MAX_ENTRIES",
    )
    trace_store_ttl_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        validation_alias="ALIENESE_TRACE_STORE_TTL_SECONDS",
    )

    # Shared remote-provider operational runtime settings
    provider_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        le=300.0,
        validation_alias="ALIENESE_PROVIDER_TIMEOUT_SECONDS",
    )
    provider_max_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        validation_alias="ALIENESE_PROVIDER_MAX_ATTEMPTS",
    )
    provider_retry_ambiguous: bool = Field(
        default=False,
        validation_alias="ALIENESE_PROVIDER_RETRY_AMBIGUOUS",
    )
    provider_max_concurrency: int = Field(
        default=8,
        ge=1,
        le=128,
        validation_alias="ALIENESE_PROVIDER_MAX_CONCURRENCY",
    )
    provider_max_queue_waiters: int = Field(
        default=16,
        ge=0,
        le=512,
        validation_alias="ALIENESE_PROVIDER_MAX_QUEUE_WAITERS",
    )
    provider_cb_failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        validation_alias="ALIENESE_PROVIDER_CB_FAILURE_THRESHOLD",
    )
    provider_cb_recovery_seconds: float = Field(
        default=30.0,
        gt=0.0,
        le=600.0,
        validation_alias="ALIENESE_PROVIDER_CB_RECOVERY_SECONDS",
    )

    # Retriever provider (EmbeddingGemma 2 — Fake in Phase 2)
    retriever_provider: str = Field(
        default="fake",
        validation_alias="RETRIEVER_PROVIDER",
    )
    retriever_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="RETRIEVER_API_KEY",
    )
    retriever_base_url: str | None = Field(
        default=None,
        validation_alias="RETRIEVER_BASE_URL",
    )
    retriever_model: str = Field(
        default="google/embeddinggemma-2",
        validation_alias="RETRIEVER_MODEL",
    )

    # Controller provider (mini-Jev — Fake in Phase 2)
    controller_provider: str = Field(
        default="fake",
        validation_alias="CONTROLLER_PROVIDER",
    )
    controller_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="CONTROLLER_API_KEY",
    )
    controller_base_url: str | None = Field(
        default=None,
        validation_alias="CONTROLLER_BASE_URL",
    )
    controller_model: str = Field(
        default="samatv256/mini-Jev",
        validation_alias="CONTROLLER_MODEL",
    )

    # Generator provider (Fake by default; NVIDIA API Catalog or Nebius in hybrid mode)
    generator_provider: str = Field(
        default="fake",
        validation_alias="GENERATOR_PROVIDER",
    )
    generator_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="GENERATOR_API_KEY",
    )
    generator_base_url: str | None = Field(
        default=None,
        validation_alias="GENERATOR_BASE_URL",
    )
    generator_model: str | None = Field(
        default=None,
        validation_alias="GENERATOR_MODEL",
    )
    generator_max_tokens: int = Field(
        default=1024,
        ge=1,
        le=16384,
        validation_alias="GENERATOR_MAX_TOKENS",
    )
    generator_enable_thinking: bool = Field(
        default=False,
        validation_alias="GENERATOR_ENABLE_THINKING",
    )

    # Optional OpenTelemetry exporter endpoint
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )

    @property
    def effective_generator_model(self) -> str:
        """Return the resolved generator model identifier for the configured provider."""
        if self.alienese_provider_mode.strip().lower() == "fake":
            return "nvidia/nemotron"
        if self.generator_model and self.generator_model.strip():
            return self.generator_model.strip()
        if self.generator_provider.strip().lower() == "nvidia_build":
            return NVIDIA_NEMOTRON_SUPER_MODEL
        return "nvidia/nemotron"

    @property
    def effective_generator_base_url(self) -> str:
        """Return the resolved generator base URL for the configured provider."""
        if self.generator_base_url and self.generator_base_url.strip():
            return self.generator_base_url.strip().rstrip("/")
        if self.generator_provider == "nvidia_build":
            return NVIDIA_DEFAULT_BASE_URL
        if self.generator_provider == "nebius_token_factory":
            return NEBIUS_DEFAULT_BASE_URL
        return NVIDIA_DEFAULT_BASE_URL

    @model_validator(mode="after")
    def _validate_runtime_and_provider_matrix(self) -> Settings:
        host = self.alienese_host.strip().lower()
        if not self.alienese_allow_non_loopback and host not in _LOOPBACK_HOSTS:
            raise CompatibilityError(
                f"Refusing to bind unauthenticated development runtime to non-loopback "
                f"host '{self.alienese_host}'. Use 127.0.0.1/localhost or explicitly set "
                "ALIENESE_ALLOW_NON_LOOPBACK=true.",
                param="alienese_host",
                code="non_loopback_binding_forbidden",
            )

        mode = self.alienese_provider_mode.strip().lower()
        if mode not in _SUPPORTED_PROVIDER_MODES:
            raise CompatibilityError(
                f"Unsupported ALIENESE_PROVIDER_MODE '{self.alienese_provider_mode}'. "
                "Supported modes in Phase 2 are 'fake' and 'hybrid' ('remote' is not yet enabled).",
                param="alienese_provider_mode",
                code="unsupported_provider_mode",
            )

        if self.retriever_provider.strip().lower() != "fake":
            raise CompatibilityError(
                f"RETRIEVER_PROVIDER='{self.retriever_provider}' is not enabled in Phase 2; "
                "retriever must remain 'fake'.",
                param="retriever_provider",
                code="invalid_provider_mode_combination",
            )

        if self.controller_provider.strip().lower() != "fake":
            raise CompatibilityError(
                f"CONTROLLER_PROVIDER='{self.controller_provider}' is not enabled in Phase 2; "
                "controller must remain 'fake'.",
                param="controller_provider",
                code="invalid_provider_mode_combination",
            )

        gen_prov = self.generator_provider.strip().lower()
        if gen_prov not in {"fake", "nvidia_build", "nebius_token_factory"}:
            raise CompatibilityError(
                f"Unsupported GENERATOR_PROVIDER '{self.generator_provider}'.",
                param="generator_provider",
                code="invalid_provider_mode_combination",
            )

        if mode == "fake":
            # In fake mode, all three providers remain fake regardless of whether
            # .env is populated with remote generator credentials.
            return self

        # mode == "hybrid"
        if gen_prov not in {"nvidia_build", "nebius_token_factory"}:
            raise CompatibilityError(
                f"ALIENESE_PROVIDER_MODE='hybrid' requires GENERATOR_PROVIDER to be "
                f"'nvidia_build' or 'nebius_token_factory' (got '{self.generator_provider}').",
                param="generator_provider",
                code="invalid_provider_mode_combination",
            )

        raw_key = (
            self.generator_api_key.get_secret_value().strip()
            if self.generator_api_key is not None
            else ""
        )
        if not raw_key:
            raise CompatibilityError(
                f"ALIENESE_PROVIDER_MODE='hybrid' with GENERATOR_PROVIDER='{gen_prov}' "
                "requires a non-empty GENERATOR_API_KEY.",
                param="generator_api_key",
                code="missing_provider_api_key",
            )

        resolved_url = self.effective_generator_base_url
        parsed_url = urlparse(resolved_url)
        scheme = parsed_url.scheme.lower()
        url_host = (parsed_url.hostname or "").lower()
        if scheme != "https" or not url_host:
            raise CompatibilityError(
                f"GENERATOR_BASE_URL for '{gen_prov}' must be a valid HTTPS URL.",
                param="generator_base_url",
                code="insecure_provider_base_url",
            )

        resolved_model = self.effective_generator_model
        if not resolved_model or resolved_model == "nvidia/nemotron":
            raise CompatibilityError(
                f"GENERATOR_MODEL for '{gen_prov}' must be an explicit remote model ID, "
                "not the fake-mode placeholder 'nvidia/nemotron'.",
                param="generator_model",
                code="invalid_generator_model",
            )

        if gen_prov == "nvidia_build":
            if not (url_host == "integrate.api.nvidia.com" or url_host.endswith(".api.nvidia.com")):
                raise CompatibilityError(
                    f"GENERATOR_PROVIDER='nvidia_build' requires an NVIDIA API Catalog origin "
                    f"('integrate.api.nvidia.com'), got '{url_host}'.",
                    param="generator_base_url",
                    code="provider_origin_mismatch",
                )
        elif gen_prov == "nebius_token_factory":
            if raw_key.startswith("nvapi-"):
                raise CompatibilityError(
                    "Refusing to reuse an NVIDIA API key ('nvapi-*') when "
                    "GENERATOR_PROVIDER='nebius_token_factory'.",
                    param="generator_api_key",
                    code="cross_provider_credential_reuse",
                )
            if not (
                url_host == "api.tokenfactory.nebius.com"
                or url_host.endswith(".nebius.com")
                or url_host.endswith(".nebius.ai")
            ):
                raise CompatibilityError(
                    f"GENERATOR_PROVIDER='nebius_token_factory' requires a Nebius origin "
                    f"('api.tokenfactory.nebius.com'), got '{url_host}'.",
                    param="generator_base_url",
                    code="provider_origin_mismatch",
                )
            if not (self.generator_model and self.generator_model.strip()):
                raise CompatibilityError(
                    "GENERATOR_PROVIDER='nebius_token_factory' requires an explicitly "
                    "configured GENERATOR_MODEL.",
                    param="generator_model",
                    code="invalid_generator_model",
                )

        return self

    def safe_dump(self) -> dict[str, Any]:
        """Return a dictionary representation with all secrets masked."""
        raw: dict[str, Any] = {}
        for field_name in self.__class__.model_fields:
            val = getattr(self, field_name)
            if isinstance(val, SecretStr):
                raw[field_name] = REDACTED_PLACEHOLDER if val.get_secret_value() else None
            else:
                raw[field_name] = val
        return redact_mapping(raw)
