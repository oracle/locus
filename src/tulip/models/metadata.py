# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-model metadata registry (context length, pricing, capabilities).

Tulip's :class:`ModelConfig` tracks the *output* ``max_tokens``, but
many agent-time decisions (context compaction thresholds, cost
telemetry, whether to enable prompt caching) need the *input* window
and other per-model capabilities. This module exposes a lightweight
static registry keyed on model ID.

Design:

* **Static by default.** A seed table covers common provider families
  with publicly documented context lengths (as of 2026-04). Entries
  intentionally carry only the fields that drive SDK behaviour — this
  is not a comprehensive spec sheet.
* **Extensible.** Call :func:`register_metadata` to register a custom
  entry, e.g. a fine-tune or a self-hosted model. Later lookups for
  the same model ID return the registered entry.
* **Provider-prefix tolerant.** ``metadata_for("openai:gpt-4o")`` and
  ``metadata_for("gpt-4o")`` both resolve, as do
  ``"anthropic:claude-sonnet-4-6"`` and the bare ``"claude-sonnet-4-6"``.
  Canonical form is stored without a prefix; the lookup normalises inputs.
* **Unknown models** return ``None`` rather than a default — callers
  choose how to handle it (a conservative fallback, a log warning, or
  the existing ``ModelConfig`` values).
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any, Final

from pydantic import BaseModel, Field


__all__ = [
    "ModelMetadata",
    "discover_context_length",
    "known_models",
    "metadata_for",
    "model_id_of",
    "register_metadata",
]


# ---------------------------------------------------------------------------
# Model record
# ---------------------------------------------------------------------------


class ModelMetadata(BaseModel):
    """Frozen per-model capability record."""

    model_config = {"frozen": True}

    model_id: str = Field(
        min_length=1,
        description="Canonical model slug, without provider prefix.",
    )
    family: str = Field(
        description="Provider / vendor family — e.g. 'openai', 'anthropic'.",
    )
    context_length: int = Field(
        ge=1,
        description="Input context window (tokens) as published by the provider.",
    )
    max_output_tokens: int = Field(
        ge=1,
        description="Output cap (tokens). May be further limited per-request.",
    )
    supports_prompt_caching: bool = False
    input_price_per_mtok: Decimal | None = Field(
        default=None,
        description="USD per million input tokens. ``None`` when unknown.",
    )
    output_price_per_mtok: Decimal | None = Field(
        default=None,
        description="USD per million output tokens. ``None`` when unknown.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Provider prefixes stripped at lookup time: the native bindings plus every
# OpenAI-compatible routing prefix in ``tulip.models.providers`` (a unit test
# keeps the two lists in step). What remains is the slug the provider itself
# is called with — the same string a built model carries as ``config.model``,
# so ``"openrouter:deepseek/deepseek-v4-flash"`` and an ``OpenAIModel`` built
# from it resolve to one entry. Slugs stay distinct across routes that price
# differently: OpenRouter's ``deepseek/deepseek-v4-flash`` is not DeepSeek's
# own ``deepseek-v4-flash``. Users supplying a different prefix can register
# metadata under the canonical slug directly.
_PROVIDER_PREFIXES: Final[frozenset[str]] = frozenset(
    {
        "openai",
        "anthropic",
        # tulip.models.providers.COMPATIBLE_PROVIDERS
        "ollama",
        "vllm",
        "lmstudio",
        "llamacpp",
        "litellm",
        "groq",
        "together",
        "openrouter",
        "deepseek",
        "mistral",
        "xai",
        "fireworks",
        "cerebras",
        "perplexity",
        "nvidia",
        "gemini",
        "openai-compatible",
    }
)


def _strip_prefix(model_id: str) -> str:
    if ":" not in model_id:
        return model_id
    prefix, _, rest = model_id.partition(":")
    if prefix.strip().lower() in _PROVIDER_PREFIXES:
        return rest.strip()
    return model_id


_lock = threading.Lock()
_registry: dict[str, ModelMetadata] = {}


def register_metadata(md: ModelMetadata) -> None:
    """Register or overwrite a :class:`ModelMetadata` entry.

    Call at import time from user code to add fine-tunes, regional
    aliases, or self-hosted models that Tulip doesn't ship
    seed data for.
    """
    with _lock:
        _registry[md.model_id] = md


def metadata_for(model_id: str) -> ModelMetadata | None:
    """Return the metadata record for ``model_id`` or ``None``.

    Accepts both bare (``"gpt-4o"``) and prefixed (``"openai:gpt-4o"``)
    forms. Only prefixes Tulip's providers use are stripped; anything
    else is treated as part of the slug.
    """
    key = _strip_prefix(model_id.strip())
    with _lock:
        return _registry.get(key)


def model_id_of(model: object) -> str | None:
    """The model slug a model object (or a string id) is called with.

    A string is returned as-is. An object is read the way the rest of the
    SDK reads it: ``model.config.model`` (every native binding), else a
    string ``model.model``. Wrappers that proxy attribute access to an inner
    model — a ``FallbackChain`` (its ``config`` is the primary tier's), or a
    caller's per-turn view that forwards ``__getattr__`` — resolve through
    the proxy. ``None`` when nothing names the model.
    """
    if isinstance(model, str):
        return model
    name = getattr(getattr(model, "config", None), "model", None)
    if not isinstance(name, str) or not name:
        name = getattr(model, "model", None)
    return name if isinstance(name, str) and name else None


async def discover_context_length(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    request_timeout: float = 10.0,
    register: bool = True,
) -> int | None:
    """Read a served model's context window from its ``/models`` listing.

    vLLM lists ``max_model_len`` for each model it serves, the window it
    was started with — which can be smaller than the model's published
    one, so it is the number the server will actually enforce. OpenRouter
    and some other gateways list ``context_length`` instead; both are read.

    The call is explicit and async on purpose: agent construction stays
    offline, and a caller that wants the window asks for it once, before
    building agents. With ``register`` (the default) the window is stored
    via :func:`register_metadata`, keeping any prices already registered
    for the model, so every agent built afterwards counts tokens against it.

    Args:
        base_url: The OpenAI-compatible base URL, e.g. ``http://host:8000/v1``.
        model: The served model id; a provider prefix (``vllm:``) is dropped.
        api_key: Bearer token, when the server requires one.
        request_timeout: Request timeout in seconds.
        register: Store the discovered window in the metadata registry.

    Returns:
        The window in tokens, or ``None`` when the server is unreachable or
        does not list one for ``model``. Never raises for those cases:
        discovery is best effort and the caller decides the fallback.
    """
    import httpx

    slug = _strip_prefix(model.strip())
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            listed = response.json().get("data") or []
    except (httpx.HTTPError, ValueError, AttributeError):
        return None

    window: int | None = None
    for entry in listed:
        if not isinstance(entry, dict) or entry.get("id") != slug:
            continue
        window = _positive_int(entry.get("max_model_len")) or _positive_int(
            entry.get("context_length")
        )
        break
    if window is None:
        return None
    if register:
        existing = metadata_for(slug)
        if existing is not None:
            register_metadata(existing.model_copy(update={"context_length": window}))
        else:
            register_metadata(
                ModelMetadata(
                    model_id=slug,
                    family="discovered",
                    context_length=window,
                    max_output_tokens=window,
                )
            )
    return window


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def known_models() -> list[str]:
    """Snapshot of all registered model IDs, sorted."""
    with _lock:
        return sorted(_registry)


# ---------------------------------------------------------------------------
# Seed table
# ---------------------------------------------------------------------------
#
# Seed values reflect publicly documented specs as of 2026-04. Keep this
# list tight — only models that users of Tulip are likely to touch.
# Anything else registers via :func:`register_metadata` at user import.


def _seed(
    model_id: str,
    *,
    family: str,
    context_length: int,
    max_output_tokens: int,
    supports_prompt_caching: bool = False,
    input_price_per_mtok: str | None = None,
    output_price_per_mtok: str | None = None,
) -> None:
    _registry[model_id] = ModelMetadata(
        model_id=model_id,
        family=family,
        context_length=context_length,
        max_output_tokens=max_output_tokens,
        supports_prompt_caching=supports_prompt_caching,
        input_price_per_mtok=Decimal(input_price_per_mtok)
        if input_price_per_mtok is not None
        else None,
        output_price_per_mtok=Decimal(output_price_per_mtok)
        if output_price_per_mtok is not None
        else None,
    )


# OpenAI
_seed(
    "gpt-4o",
    family="openai",
    context_length=128_000,
    max_output_tokens=16_384,
    supports_prompt_caching=True,
    input_price_per_mtok="2.50",
    output_price_per_mtok="10.00",
)
_seed(
    "gpt-4o-mini",
    family="openai",
    context_length=128_000,
    max_output_tokens=16_384,
    supports_prompt_caching=True,
    input_price_per_mtok="0.15",
    output_price_per_mtok="0.60",
)
_seed(
    "gpt-4.1",
    family="openai",
    context_length=1_000_000,
    max_output_tokens=32_768,
    supports_prompt_caching=True,
    input_price_per_mtok="2.00",
    output_price_per_mtok="8.00",
)
_seed(
    "gpt-4.1-mini",
    family="openai",
    context_length=1_000_000,
    max_output_tokens=32_768,
    supports_prompt_caching=True,
    input_price_per_mtok="0.40",
    output_price_per_mtok="1.60",
)
_seed(
    "gpt-5",
    family="openai",
    context_length=400_000,
    max_output_tokens=128_000,
    supports_prompt_caching=True,
)
_seed(
    "gpt-5-mini",
    family="openai",
    context_length=400_000,
    max_output_tokens=64_000,
    supports_prompt_caching=True,
)
_seed(
    "o1",
    family="openai",
    context_length=200_000,
    max_output_tokens=100_000,
)
_seed(
    "o3",
    family="openai",
    context_length=200_000,
    max_output_tokens=100_000,
)
_seed(
    "o4-mini",
    family="openai",
    context_length=200_000,
    max_output_tokens=100_000,
)

# Anthropic
_seed(
    "claude-opus-4",
    family="anthropic",
    context_length=1_000_000,
    max_output_tokens=64_000,
    supports_prompt_caching=True,
    input_price_per_mtok="15.00",
    output_price_per_mtok="75.00",
)
_seed(
    "claude-sonnet-4",
    family="anthropic",
    context_length=1_000_000,
    max_output_tokens=64_000,
    supports_prompt_caching=True,
    input_price_per_mtok="3.00",
    output_price_per_mtok="15.00",
)
_seed(
    "claude-haiku-4",
    family="anthropic",
    context_length=200_000,
    max_output_tokens=16_384,
    supports_prompt_caching=True,
    input_price_per_mtok="0.80",
    output_price_per_mtok="4.00",
)

# Qwen (Alibaba) — open-weight reasoning models commonly served via
# vLLM with ``--reasoning-parser qwen``. Context windows as published
# for the Qwen3.5 / Qwen3.6 family (2026); pricing is None because
# these are almost always self-hosted.
_seed(
    "qwen3.6-35b",
    family="qwen",
    context_length=262_000,
    max_output_tokens=32_768,
)
_seed(
    "qwen3.6-35b-a3b",
    family="qwen",
    context_length=262_000,
    max_output_tokens=32_768,
)

# DeepSeek V4 — two routes, priced separately because they bill differently.
#
# OpenRouter slugs (``openrouter:deepseek/...``): context length, output cap
# (``top_provider.max_completion_tokens``) and price are OpenRouter's listed
# values from ``GET https://openrouter.ai/api/v1/models`` on 2026-09-28.
# OpenRouter routes each request to one of ~15 hosts whose prices differ
# (V4 Flash output: $0.10-$1.28/M on that date), so the listed price is what
# OpenRouter quotes, not a ceiling. A hard ``max_cost_usd`` cap that must
# never under-count should register the highest price among the providers it
# allows, or pin providers via ``extra_body={"provider": {...}}``.
_seed(
    "deepseek/deepseek-v4-flash",
    family="deepseek",
    context_length=1_048_576,
    max_output_tokens=131_072,
    input_price_per_mtok="0.05152",
    output_price_per_mtok="0.10304",
)
_seed(
    "deepseek/deepseek-v4.1-flash",
    family="deepseek",
    context_length=1_048_576,
    max_output_tokens=943_718,
    input_price_per_mtok="0.30",
    output_price_per_mtok="1.20",
)
_seed(
    "deepseek/deepseek-v4-pro",
    family="deepseek",
    context_length=1_048_576,
    max_output_tokens=384_000,
    input_price_per_mtok="0.783",
    output_price_per_mtok="1.566",
)

# DeepSeek's own API (``deepseek:...``), per api-docs.deepseek.com
# "Models & Pricing" on 2026-09-28: 1M context, 384K max output. Prices are
# the PEAK (cache-miss input) rates — off-peak is half — so a budget never
# under-counts. ``deepseek-flash`` is DeepSeek-V4.1-Flash; the legacy
# ``deepseek-v4-flash`` name is still accepted and served (and billed) as it.
for _slug in ("deepseek-flash", "deepseek-v4-flash"):
    _seed(
        _slug,
        family="deepseek",
        context_length=1_000_000,
        max_output_tokens=384_000,
        input_price_per_mtok="0.30",
        output_price_per_mtok="1.20",
    )
del _slug
_seed(
    "deepseek-v4-pro",
    family="deepseek",
    context_length=1_000_000,
    max_output_tokens=384_000,
    input_price_per_mtok="1.32",
    output_price_per_mtok="3.96",
)
