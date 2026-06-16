# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Model providers for Locus.

Models are organized into two categories:

1. Native providers (direct API connections):
   - OpenAI (GPT) - Oracle partnership

2. Hosted providers (OCI GenAI):
   - ``OCIChatCompletionsModel`` — OpenAI-compatible ``/openai/v1`` transport.
     For OpenAI / Meta / xAI / Mistral / Gemini families.
     Real SSE streaming and day-0 model support.
   - ``OCIModel`` — OCI SDK transport against ``/20231130/actions/v1``.
     Required for Cohere R-series (``cohere.command-r-*``).

Usage:
    # Native provider (OpenAI)
    from locus.models import OpenAIModel
    model = OpenAIModel(model="gpt-4o")

    # OCI GenAI — V1 transport
    from locus.models import OCIChatCompletionsModel
    model = OCIChatCompletionsModel(model="openai.gpt-5", profile="DEFAULT")

    # OCI GenAI — Cohere R-series
    from locus.models import OCIModel
    model = OCIModel(
        model_id="cohere.command-r-plus",
        profile_name="DEFAULT",
        auth_type="api_key",
    )

    # String factory — auto-routes to the right transport
    from locus.models import get_model
    model = get_model("oci:openai.gpt-5", profile="DEFAULT")
"""

from locus.models.base import (
    ModelConfig,
    ModelProtocol,
    ModelResponse,
    RequestBuilder,
    ResponseParser,
)
from locus.models.registry import get_model, list_providers, register_provider


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
    # OCI GenAI (lazy imports)
    "OCIModel",
    "OCIConfig",
    "OCIAuthType",
    "OCIChatCompletionsModel",
    "OCIChatCompletionsConfig",
]


def __getattr__(name: str) -> object:
    """Lazy import providers to avoid requiring all dependencies."""
    # Native providers - OpenAI (Oracle partnership)
    if name in ("OpenAIModel", "OpenAIConfig"):
        from locus.models.native.openai import OpenAIConfig, OpenAIModel

        return OpenAIModel if name == "OpenAIModel" else OpenAIConfig

    # OCI GenAI
    if name in ("OCIModel", "OCIConfig", "OCIAuthType"):
        from locus.models.providers.oci import OCIAuthType, OCIConfig, OCIModel

        if name == "OCIModel":
            return OCIModel
        if name == "OCIConfig":
            return OCIConfig
        return OCIAuthType

    if name in ("OCIChatCompletionsModel", "OCIChatCompletionsConfig"):
        from locus.models.providers.oci import OCIChatCompletionsConfig, OCIChatCompletionsModel

        return (
            OCIChatCompletionsModel
            if name == "OCIChatCompletionsModel"
            else OCIChatCompletionsConfig
        )

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
