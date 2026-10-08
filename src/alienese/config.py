"""Environment-driven typed configuration for the Alienese runtime.

Provider API keys use SecretStr so that accidental serialization or logging
never exposes credentials. Fake providers are used by default in development
and test environments.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from alienese.observability.redaction import REDACTED_PLACEHOLDER, redact_mapping


class Settings(BaseSettings):
    """Typed runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # Serving environment
    alienese_env: Literal["development", "test", "production"] = Field(
        default="development",
        validation_alias="ALIENESE_ENV",
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
