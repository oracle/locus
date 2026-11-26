# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Model registry and factory - 100% Pydantic."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from locus.core.errors import UnknownProviderError


if TYPE_CHECKING:
    from locus.core.protocols import ModelProtocol

# Provider factories: prefix -> factory function
_PROVIDERS: dict[str, Callable[..., ModelProtocol]] = {}


def register_provider(prefix: str, factory: Callable[..., ModelProtocol]) -> None:
    """
    Register a model provider.

    The factory is called as ``factory(model_id, **kwargs)``: ``model_id`` is
    the part of the model string after ``prefix:`` and ``kwargs`` are the
    keyword arguments :func:`get_model` was given. Registering a prefix again
    replaces its factory.

    Args:
        prefix: Provider prefix (e.g., "openai", "anthropic")
        factory: Factory function that takes model name and kwargs

    Example:
        ```python
        from locus.models.registry import get_model, register_provider
        from locus.testing import FunctionModel


        def mine(model_id: str, **kwargs):
            # kwargs are whatever get_model("mine:...", ...) was given
            return FunctionModel(lambda messages, tools: f"{model_id} says hello")


        register_provider("mine", mine)
        model = get_model("mine:small")
        assert isinstance(model, FunctionModel)
        ```

        After that, ``Agent(model="mine:small")`` resolves through the same
        factory.
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
        UnknownProviderError: If the model string is not ``provider:model``
            or the provider is not registered. It is a ``ConfigError`` and a
            ``ValueError``, and carries ``provider`` and ``available``.
    """
    available = list(_PROVIDERS.keys())
    if ":" not in model_string:
        raise UnknownProviderError(
            f"Model string must be 'provider:model', got: {model_string}. "
            f"Available providers: {available}",
            provider=None,
            available=available,
        )

    provider, model_id = model_string.split(":", 1)

    if provider not in _PROVIDERS:
        raise UnknownProviderError(
            f"Unknown provider: {provider}. Available: {available}",
            provider=provider or None,
            available=available,
        )

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
    # Transport rules, evaluated top-down:
    #   0. ``transport="responses"`` — ``OCIResponsesModel``, for models the
    #      chat endpoint serves only without function tools (see
    #      ``OCIToolsNeedResponsesError``) and for server-held threads.
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
    # Every transport takes the profile and the region from the environment
    # the same way when the caller names neither: ``LOCUS_OCI_PROFILE`` then
    # ``OCI_PROFILE``, and ``LOCUS_OCI_REGION`` then ``OCI_REGION``. Explicit
    # kwargs always win.
    #
    # See docs/how-to/oci-models.md and docs/how-to/oci-dac.md.
    try:
        from locus.models.providers.oci import (
            OCIChatCompletionsModel,
            OCIModel,
            OCIResponsesModel,
        )

        def _oci_env_defaults(kw: dict[str, Any], *, endpoint_key: str) -> None:
            if not {"profile", "profile_name", "auth_type"} & kw.keys():
                env_profile = os.environ.get("LOCUS_OCI_PROFILE") or os.environ.get("OCI_PROFILE")
                if env_profile:
                    kw["profile"] = env_profile
            if "region" not in kw and endpoint_key not in kw:
                env_region = os.environ.get("LOCUS_OCI_REGION") or os.environ.get("OCI_REGION")
                if env_region:
                    kw["region"] = env_region

        def _make_oci(m: str, **kw: Any) -> ModelProtocol:
            lowered = m.lower()
            transport = kw.pop("transport", None)
            if transport not in (None, "responses"):
                msg = f'oci: transport must be "responses" or left out, got {transport!r}'
                raise ValueError(msg)
            # Rule 0: the caller asked for the Responses API.
            if transport == "responses":
                _oci_env_defaults(kw, endpoint_key="base_url")
                return cast("ModelProtocol", OCIResponsesModel(model=m, **kw))
            # Rule 1: DAC endpoint OCID → SDK transport.
            # Rule 2: Cohere chat models (command-r*, command-a*) → SDK transport.
            # OCI serves Cohere only through its native chat API; /openai/v1
            # answers 400 "Unsupported OpenAI operation" for every cohere.* id.
            if lowered.startswith(("ocid1.generativeaiendpoint.", "cohere.")):
                # With neither set here nor in the environment, the SDK
                # transport keeps its own defaults: the DEFAULT profile and
                # that profile's region.
                _oci_env_defaults(kw, endpoint_key="service_endpoint")
                return cast("ModelProtocol", OCIModel(model_id=m, **kw))
            # Rule 3: V1 transport. Strictly requires profile= or auth_type=;
            # the environment fallback makes `Agent(model="oci:...")` work in
            # one line. Without a region it uses the class default.
            _oci_env_defaults(kw, endpoint_key="base_url")
            return cast("ModelProtocol", OCIChatCompletionsModel(model=m, **kw))

        register_provider("oci", _make_oci)
    except ImportError:
        pass


# Register on import
_register_defaults()
