# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible hosted and self-hosted providers.

A large share of the model ecosystem speaks the OpenAI wire protocol: managed
services (Groq, Together, OpenRouter, DeepSeek, Mistral, xAI, Fireworks,
Cerebras, Perplexity, NVIDIA NIM) and self-hosted servers (Ollama, vLLM,
LM Studio, llama.cpp, LiteLLM). They differ only in **base URL** and **which
environment variable holds the key**.

:class:`~tulip.models.native.openai.OpenAIModel` already handles the wire
format, including the important detail that a custom ``base_url`` must never
auto-select ``/v1/responses`` — a gateway serves chat-completions and not the
Responses API. This module supplies the routing table so those endpoints are
reachable by name::

    Agent(model="groq:llama-3.3-70b-versatile")
    Agent(model="ollama:qwen3")
    Agent(model="deepseek:deepseek-chat")

Anything not in the table is still reachable without a code change, by giving
the base URL explicitly::

    Agent(model="openai-compatible:my-model", base_url="https://host/v1")

Every entry resolves its key from the provider's conventional environment
variable, and each accepts a ``TULIP_<PREFIX>_BASE_URL`` override for
self-hosted deployments and proxies. Explicit ``api_key=`` / ``base_url=``
keyword arguments always win over the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from tulip.core.protocols import ModelProtocol


__all__ = [
    "COMPATIBLE_PROVIDERS",
    "CompatibleProvider",
    "provider_table",
    "register_compatible_providers",
]


#: Placeholder key for servers that accept any value. The ``openai`` client
#: refuses to construct without *some* key, but a local Ollama or llama.cpp
#: server never checks it.
_LOCAL_PLACEHOLDER_KEY = "not-needed"


@dataclass(frozen=True)
class CompatibleProvider:
    """One OpenAI-compatible endpoint, addressable by prefix.

    Attributes:
        prefix: The ``provider:`` half of a model string.
        base_url: Default endpoint. ``None`` means the caller must supply one
            (used by the generic ``openai-compatible`` prefix).
        env_key: Environment variable holding the API key.
        label: Human-readable name, used in the provider table and errors.
        env_base_url: Provider-conventional environment variable that
            overrides ``base_url``, if the vendor defines one.
        requires_key: When ``False`` a placeholder key is used, for local
            servers that do not authenticate.
    """

    prefix: str
    base_url: str | None
    env_key: str
    label: str
    env_base_url: str | None = None
    requires_key: bool = True

    @property
    def tulip_base_url_env(self) -> str:
        """The ``TULIP_``-namespaced base-URL override for this provider."""
        return f"TULIP_{self.prefix.replace('-', '_').upper()}_BASE_URL"

    def resolve_base_url(self, explicit: str | None = None) -> str | None:
        """Resolve the endpoint: explicit > TULIP_ override > vendor env > default."""
        if explicit:
            return explicit
        for var in (self.tulip_base_url_env, self.env_base_url):
            if var and (value := os.environ.get(var)):
                return value
        return self.base_url

    def resolve_api_key(self, explicit: str | None = None) -> str | None:
        """Resolve the API key: explicit > provider env > placeholder-or-None."""
        if explicit:
            return explicit
        if key := os.environ.get(self.env_key):
            return key
        return None if self.requires_key else _LOCAL_PLACEHOLDER_KEY


#: The routing table. ``base_url`` values are the vendor's documented
#: OpenAI-compatible endpoint. Self-hosted entries default to the server's
#: conventional local port and are expected to be overridden in most real
#: deployments.
COMPATIBLE_PROVIDERS: tuple[CompatibleProvider, ...] = (
    # --- self-hosted ---------------------------------------------------
    CompatibleProvider(
        prefix="ollama",
        base_url="http://localhost:11434/v1",
        env_key="OLLAMA_API_KEY",
        label="Ollama",
        env_base_url="OLLAMA_BASE_URL",
        requires_key=False,
    ),
    CompatibleProvider(
        prefix="vllm",
        base_url="http://localhost:8000/v1",
        env_key="VLLM_API_KEY",
        label="vLLM",
        env_base_url="VLLM_BASE_URL",
        requires_key=False,
    ),
    CompatibleProvider(
        prefix="lmstudio",
        base_url="http://localhost:1234/v1",
        env_key="LMSTUDIO_API_KEY",
        label="LM Studio",
        env_base_url="LMSTUDIO_BASE_URL",
        requires_key=False,
    ),
    CompatibleProvider(
        prefix="llamacpp",
        base_url="http://localhost:8080/v1",
        env_key="LLAMACPP_API_KEY",
        label="llama.cpp server",
        env_base_url="LLAMACPP_BASE_URL",
        requires_key=False,
    ),
    CompatibleProvider(
        prefix="litellm",
        base_url="http://localhost:4000/v1",
        env_key="LITELLM_API_KEY",
        label="LiteLLM gateway",
        env_base_url="LITELLM_GATEWAY_URL",
    ),
    # --- managed -------------------------------------------------------
    CompatibleProvider(
        prefix="groq",
        base_url="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY",
        label="Groq",
    ),
    CompatibleProvider(
        prefix="together",
        base_url="https://api.together.xyz/v1",
        env_key="TOGETHER_API_KEY",
        label="Together AI",
    ),
    CompatibleProvider(
        prefix="openrouter",
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        label="OpenRouter",
    ),
    CompatibleProvider(
        prefix="deepseek",
        base_url="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
        label="DeepSeek",
    ),
    CompatibleProvider(
        prefix="mistral",
        base_url="https://api.mistral.ai/v1",
        env_key="MISTRAL_API_KEY",
        label="Mistral AI",
    ),
    CompatibleProvider(
        prefix="xai",
        base_url="https://api.x.ai/v1",
        env_key="XAI_API_KEY",
        label="xAI (Grok)",
    ),
    CompatibleProvider(
        prefix="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        env_key="FIREWORKS_API_KEY",
        label="Fireworks AI",
    ),
    CompatibleProvider(
        prefix="cerebras",
        base_url="https://api.cerebras.ai/v1",
        env_key="CEREBRAS_API_KEY",
        label="Cerebras",
    ),
    CompatibleProvider(
        prefix="perplexity",
        base_url="https://api.perplexity.ai",
        env_key="PERPLEXITY_API_KEY",
        label="Perplexity",
    ),
    CompatibleProvider(
        prefix="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        env_key="NVIDIA_API_KEY",
        label="NVIDIA NIM",
    ),
    # --- escape hatch --------------------------------------------------
    CompatibleProvider(
        prefix="openai-compatible",
        base_url=None,
        env_key="OPENAI_COMPATIBLE_API_KEY",
        label="Any OpenAI-compatible endpoint",
        requires_key=False,
    ),
)


def _build(spec: CompatibleProvider, model_id: str, **kwargs: Any) -> ModelProtocol:
    """Construct an :class:`OpenAIModel` pointed at ``spec``'s endpoint."""
    from tulip.models.native.openai import OpenAIModel

    base_url = spec.resolve_base_url(kwargs.pop("base_url", None))
    if base_url is None:
        raise ValueError(
            f"{spec.label} needs an endpoint. Pass base_url=... to the model, "
            f"or set {spec.tulip_base_url_env}. Example: "
            f'Agent(model="{spec.prefix}:{model_id}", base_url="https://your-host/v1")'
        )

    api_key = spec.resolve_api_key(kwargs.pop("api_key", None))
    if api_key is None:
        raise ValueError(
            f"No API key for {spec.label}. Set {spec.env_key}, or pass api_key=... to the model."
        )

    return cast(
        "ModelProtocol",
        OpenAIModel(model=model_id, api_key=api_key, base_url=base_url, **kwargs),
    )


def register_compatible_providers() -> None:
    """Register every entry in :data:`COMPATIBLE_PROVIDERS`.

    Called from the registry's default registration. Safe to call twice —
    registration is a dict assignment keyed by prefix.
    """
    from tulip.models.registry import register_provider

    for spec in COMPATIBLE_PROVIDERS:
        # Bind ``spec`` per iteration; a bare closure would capture the loop
        # variable and give every provider the last table entry.
        def factory(model_id: str, _spec: CompatibleProvider = spec, **kw: Any) -> ModelProtocol:
            return _build(_spec, model_id, **kw)

        register_provider(spec.prefix, factory)


def provider_table() -> str:
    """Render the provider table as Markdown, for docs and the README.

    Kept next to the data so a published table cannot drift from what is
    actually registered.
    """
    rows = [
        "| Prefix | Provider | Endpoint | API key |",
        "|---|---|---|---|",
        "| `openai` | OpenAI | (default) | `OPENAI_API_KEY` |",
        "| `anthropic` | Anthropic | (default) | `ANTHROPIC_API_KEY` |",
    ]
    for spec in COMPATIBLE_PROVIDERS:
        endpoint = f"`{spec.base_url}`" if spec.base_url else "_(supply `base_url`)_"
        key = f"`{spec.env_key}`" + ("" if spec.requires_key else " _(optional)_")
        rows.append(f"| `{spec.prefix}` | {spec.label} | {endpoint} | {key} |")
    return "\n".join(rows)
