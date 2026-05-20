# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Model registry and factory - 100% Pydantic."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from locus.core.protocols import ModelProtocol

# Provider factories: prefix -> factory function
_PROVIDERS: dict[str, Callable[..., ModelProtocol]] = {}


def register_provider(prefix: str, factory: Callable[..., ModelProtocol]) -> None:
    """
    Register a model provider.

    Args:
        prefix: Provider prefix (e.g., "openai", "oci")
        factory: Factory function that takes model name and kwargs
    """
    _PROVIDERS[prefix] = factory


def get_model(model_string: str, **kwargs: Any) -> ModelProtocol:
    """
    Get a model from a string identifier.

    Format: "provider:model_name"

    Examples:
        - "openai:gpt-4o"
        - "oci:cohere.command-r-plus"

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
    # OpenAI (Oracle partnership)
    try:
        from locus.models.native.openai import OpenAIModel

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
        from locus.models.native.anthropic import AnthropicModel

        register_provider(
            "anthropic",
            lambda m, **kw: cast("ModelProtocol", AnthropicModel(model=m, **kw)),
        )
    except ImportError:
        pass

    # Ollama (local LLMs)
    try:
        from locus.models.native.ollama import OllamaModel

        register_provider(
            "ollama",
            lambda m, **kw: cast("ModelProtocol", OllamaModel(model=m, **kw)),
        )
    except ImportError:
        pass

    # OCI GenAI — pick the right transport per model family.
    #
    # Three transport rules, evaluated top-down:
    #   1. Dedicated AI Cluster (DAC) endpoint OCIDs — strings starting
    #      with ``ocid1.generativeaiendpoint.`` — go through ``OCIModel``
    #      (SDK transport). The DAC endpoint OCID is passed verbatim to
    #      ``DedicatedServingMode(endpoint_id=...)``; the V1 transport
    #      doesn't speak that mode.
    #   2. Cohere R-series (``cohere.command-r-*``) needs the OCI SDK's
    #      proprietary chat shape — also ``OCIModel``.
    #   3. Everything else (OpenAI / Meta / xAI / Mistral / Gemini and
    #      non-R Cohere on-demand) goes through ``OCIChatCompletionsModel``
    #      against ``/openai/v1/chat/completions`` — real SSE streaming,
    #      day-0 model support, no Project OCID required.
    #
    # See docs/how-to/oci-models.md and docs/how-to/oci-dac.md.
    try:
        from locus.models.providers.oci import OCIChatCompletionsModel, OCIModel

        def _make_oci(m: str, **kw: Any) -> ModelProtocol:
            lowered = m.lower()
            # Rule 1: DAC endpoint OCID → SDK transport.
            if lowered.startswith("ocid1.generativeaiendpoint."):
                return cast("ModelProtocol", OCIModel(model_id=m, **kw))
            # Rule 2: Cohere R-series → SDK transport.
            if lowered.startswith("cohere.command-r"):
                # SDK transport: defaults to profile_name="DEFAULT" + API_KEY,
                # so no env-var fallback needed for one-line ergonomics.
                return cast("ModelProtocol", OCIModel(model_id=m, **kw))
            # Rule 3: V1 transport. Strictly requires profile= or auth_type=.
            # Fall back to OCI_PROFILE env var so `Agent(model="oci:...")`
            # works in one line. Explicit kwargs always win.
            if "profile" not in kw and "auth_type" not in kw:
                env_profile = os.environ.get("OCI_PROFILE")
                if env_profile:
                    kw["profile"] = env_profile
            return cast("ModelProtocol", OCIChatCompletionsModel(model=m, **kw))

        register_provider("oci", _make_oci)
    except ImportError:
        pass


# Register on import
_register_defaults()
