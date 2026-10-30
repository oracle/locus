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
        - "anthropic:claude-sonnet-5-5"

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

    # OpenAI-compatible endpoints — Ollama, vLLM, LM Studio, llama.cpp,
    # LiteLLM, Groq, Together, OpenRouter, DeepSeek, Mistral, xAI, Fireworks,
    # Cerebras, Perplexity, NVIDIA NIM, plus a generic ``openai-compatible``
    # escape hatch. Each is OpenAIModel pointed at a different base_url, so
    # none of them adds a client dependency; the routing table lives in
    # locus.models.providers.
    try:
        from locus.models.providers import register_compatible_providers

        register_compatible_providers()
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
    #   2. Cohere chat models (``cohere.command-r-*`` and ``cohere.command-a-*``)
    #      need the OCI SDK's proprietary chat shape — also ``OCIModel``.
    #      OCI does not expose Cohere on ``/openai/v1``.
    #   3. Everything else (OpenAI / Meta / xAI / Mistral / Gemini)
    #      goes through ``OCIChatCompletionsModel``
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
            # Rule 2: Cohere chat models (command-r*, command-a*) → SDK transport.
            # OCI serves Cohere only through its native chat API; /openai/v1
            # answers 400 "Unsupported OpenAI operation" for every cohere.* id.
            if lowered.startswith("cohere."):
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
            # The region the same way, as the provider guide documents:
            # LOCUS_OCI_REGION, then OCI_REGION, else the class default.
            # An explicit region= or base_url= is left alone.
            if "region" not in kw and "base_url" not in kw:
                env_region = os.environ.get("LOCUS_OCI_REGION") or os.environ.get("OCI_REGION")
                if env_region:
                    kw["region"] = env_region
            return cast("ModelProtocol", OCIChatCompletionsModel(model=m, **kw))

        register_provider("oci", _make_oci)
    except ImportError:
        pass


# Register on import
_register_defaults()
