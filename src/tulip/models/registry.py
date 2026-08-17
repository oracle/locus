# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Model registry and factory - 100% Pydantic."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from tulip.core.protocols import ModelProtocol

# Provider factories: prefix -> factory function
_PROVIDERS: dict[str, Callable[..., ModelProtocol]] = {}


def register_provider(prefix: str, factory: Callable[..., ModelProtocol]) -> None:
    """
    Register a model provider.

    Args:
        prefix: Provider prefix (e.g., "openai", "anthropic")
        factory: Factory function that takes model name and kwargs
    """
    _PROVIDERS[prefix] = factory


def get_model(model_string: str, **kwargs: Any) -> ModelProtocol:
    """
    Get a model from a string identifier.

    Format: "provider:model_name"

    Examples:
        - "openai:gpt-4o"
        - "anthropic:claude-sonnet-4-6"

    Args:
        model_string: Model identifier in "provider:model" format
        **kwargs: Provider-specific configuration

    Returns:
        Model instance

    Raises:
        ValueError: If provider is unknown or model string is invalid
    """
    if ":" not in model_string:
        raise ValueError(
            f"Model string must be 'provider:model', got: {model_string}. "
            f"Available providers: {list(_PROVIDERS.keys())}"
        )

    provider, model_id = model_string.split(":", 1)

    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(_PROVIDERS.keys())}")

    return _PROVIDERS[provider](model_id, **kwargs)


def list_providers() -> list[str]:
    """List available provider prefixes."""
    return list(_PROVIDERS.keys())


def _register_defaults() -> None:
    """Register default providers on import."""
    # OpenAI
    try:
        from tulip.models.native.openai import OpenAIModel

        register_provider(
            "openai",
            # The Pydantic model classes satisfy ``ModelProtocol``
            # structurally, but mypy's Callable-variance check on
            # _PROVIDERS doesn't propagate that structural narrowing —
            # cast at the registration boundary.
            lambda m, **kw: cast("ModelProtocol", OpenAIModel(model=m, **kw)),
        )
    except ImportError:
        pass

    # Anthropic (Claude)
    try:
        from tulip.models.native.anthropic import AnthropicModel

        register_provider(
            "anthropic",
            lambda m, **kw: cast("ModelProtocol", AnthropicModel(model=m, **kw)),
        )
    except ImportError:
        pass

    # Azure OpenAI — OpenAI's models, but deployment-named URLs, an
    # ``api-key`` header and a required ``api-version``, so it cannot ride
    # the OpenAI-compatible table either. Same ``openai`` dependency as the
    # OpenAI provider; no new package.
    try:
        from tulip.models.native.azure import AzureOpenAIModel

        register_provider(
            "azure",
            lambda m, **kw: cast("ModelProtocol", AzureOpenAIModel(model=m, **kw)),
        )
    except ImportError:
        pass

    # Amazon Bedrock — its own wire protocol, so it cannot ride the
    # OpenAI-compatible table below. One Converse-API client covers every
    # model on the service. boto3 is an optional extra and is imported
    # lazily inside the model, so this registration costs nothing to
    # anyone who never names a ``bedrock:`` model.
    try:
        from tulip.models.native.bedrock import BedrockModel

        register_provider(
            "bedrock",
            lambda m, **kw: cast("ModelProtocol", BedrockModel(model=m, **kw)),
        )
    except ImportError:
        pass

    # OpenAI-compatible endpoints — Ollama, vLLM, LM Studio, llama.cpp,
    # LiteLLM, Groq, Together, OpenRouter, DeepSeek, Mistral, xAI, Fireworks,
    # Cerebras, Perplexity, NVIDIA NIM, plus a generic ``openai-compatible``
    # escape hatch. Each is OpenAIModel pointed at a different base_url, so
    # none of them adds a client dependency; the routing table lives in
    # tulip.models.providers.
    try:
        from tulip.models.providers import register_compatible_providers

        register_compatible_providers()
    except ImportError:
        pass


# Register on import
_register_defaults()
