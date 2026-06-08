# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Model providers for Tulip.

Native providers (direct API connections):
   - ``OpenAIModel`` — OpenAI (GPT); also fronts OpenAI-compatible gateways via ``base_url``
   - ``AnthropicModel`` — Anthropic (Claude)

Usage:
    # Direct class
    from tulip.models import OpenAIModel
    model = OpenAIModel(model="gpt-4o")

    from tulip.models import AnthropicModel
    model = AnthropicModel(model="claude-sonnet-4-6")

    # String factory — "provider:model"
    from tulip.models import get_model
    model = get_model("anthropic:claude-sonnet-4-6")
"""

from tulip.models.base import (
    ModelConfig,
    ModelProtocol,
    ModelResponse,
    RequestBuilder,
    ResponseParser,
)
from tulip.models.registry import get_model, list_providers, register_provider


__all__ = [
    # Protocols
    "ModelProtocol",
    "RequestBuilder",
    "ResponseParser",
    # Base classes
    "ModelConfig",
    "ModelResponse",
    # Registry
    "get_model",
    "list_providers",
    "register_provider",
    # Native providers (lazy imports)
    "OpenAIModel",
    "OpenAIConfig",
    "AnthropicModel",
    "AnthropicConfig",
]


def __getattr__(name: str) -> object:
    """Lazy import providers to avoid requiring all dependencies."""
    if name in ("OpenAIModel", "OpenAIConfig"):
        from tulip.models.native.openai import OpenAIConfig, OpenAIModel

        return OpenAIModel if name == "OpenAIModel" else OpenAIConfig

    if name in ("AnthropicModel", "AnthropicConfig"):
        from tulip.models.native.anthropic import AnthropicConfig, AnthropicModel

        return AnthropicModel if name == "AnthropicModel" else AnthropicConfig

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
