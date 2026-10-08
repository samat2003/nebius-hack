"""Remote Generator provider adapters."""

from __future__ import annotations

from alienese.providers.generator.nebius_token_factory import (
    NEBIUS_DEFAULT_BASE_URL,
    NebiusTokenFactoryGenerator,
)
from alienese.providers.generator.nvidia_build import (
    NVIDIA_DEFAULT_BASE_URL,
    NVIDIA_NEMOTRON_SUPER_MODEL,
    NvidiaBuildGenerator,
)

__all__ = [
    "NEBIUS_DEFAULT_BASE_URL",
    "NVIDIA_DEFAULT_BASE_URL",
    "NVIDIA_NEMOTRON_SUPER_MODEL",
    "NebiusTokenFactoryGenerator",
    "NvidiaBuildGenerator",
]
