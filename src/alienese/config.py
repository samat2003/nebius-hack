"""Environment-driven typed configuration for the Alienese runtime.

Provider API keys use SecretStr so that accidental serialization or logging
never exposes credentials. Fake providers and loopback-only binding are used
by default in development and test environments.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from alienese.api.errors import CompatibilityError
from alienese.observability.redaction import REDACTED_PLACEHOLDER, redact_mapping

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})


class Settings(BaseSettings):
    """Typed runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    # Serving environment (defaults strictly to loopback-only in Phase 1)
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
    alienese_provider_mode: Literal["fake"] = Field(
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

    # Retriever provider (EmbeddingGemma 2 in future phase)
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

    # Controller provider (mini-Jev in future phase)
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

    # Generator provider (NVIDIA Nemotron via Nebius Token Factory in future phase)
    generator_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="GENERATOR_API_KEY",
    )
    generator_base_url: str = Field(
        default="https://api.tokenfactory.nebius.com/v1",
        validation_alias="GENERATOR_BASE_URL",
    )
    generator_model: str = Field(
        default="nvidia/nemotron",
        validation_alias="GENERATOR_MODEL",
    )

    # Optional OpenTelemetry exporter endpoint
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )

    @model_validator(mode="after")
    def _enforce_loopback_binding(self) -> Settings:
        host = self.alienese_host.strip().lower()
        if not self.alienese_allow_non_loopback and host not in _LOOPBACK_HOSTS:
            raise CompatibilityError(
                f"Refusing to bind unauthenticated development runtime to non-loopback "
                f"host '{self.alienese_host}'. Use 127.0.0.1/localhost or explicitly set "
                "ALIENESE_ALLOW_NON_LOOPBACK=true.",
                param="alienese_host",
                code="non_loopback_binding_forbidden",
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
