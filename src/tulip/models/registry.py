# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

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


# Register on import
_register_defaults()
