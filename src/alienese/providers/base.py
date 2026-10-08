"""Base provider protocols and fault simulation configuration."""

from __future__ import annotations

from enum import StrEnum

from alienese.api.errors import (
    InvalidProviderResponse,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
)
from alienese.contracts.providers import Controller, Generator, Retriever


class ProviderFaultMode(StrEnum):
    """Controlled failure modes for testing provider resilience and error mapping."""

    NONE = "NONE"
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"


def raise_for_fault_mode(provider_name: str, fault_mode: ProviderFaultMode) -> None:
    """Raise the corresponding typed ProviderError if a fault mode is active."""
    if fault_mode == ProviderFaultMode.NONE:
        return
    if fault_mode == ProviderFaultMode.TIMEOUT:
        raise ProviderTimeout(f"Provider '{provider_name}' timed out during request execution.")
    if fault_mode == ProviderFaultMode.UNAVAILABLE:
        raise ProviderUnavailable(f"Provider '{provider_name}' is currently unavailable.")
    if fault_mode == ProviderFaultMode.PROVIDER_ERROR:
        raise ProviderError(f"Provider '{provider_name}' returned an upstream execution error.")
    if fault_mode == ProviderFaultMode.MALFORMED_RESPONSE:
        raise InvalidProviderResponse(
            f"Provider '{provider_name}' returned a malformed or unparseable response."
        )


__all__ = [
    "Controller",
    "Generator",
    "ProviderFaultMode",
    "Retriever",
    "raise_for_fault_mode",
]
