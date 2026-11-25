# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Per-model metadata registry (context length, pricing, capabilities).

Locus's :class:`ModelConfig` tracks the *output* ``max_tokens``, but
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
* **OCI ids resolve the model they serve.** OCI Generative AI names a
  model ``<vendor>.<model>``: ``metadata_for("oci:openai.gpt-4.1")`` and
  the bare ``"openai.gpt-4.1"`` an ``OCIChatCompletionsModel`` carries both
  resolve the ``gpt-4.1`` entry, and the models only OCI serves this way
  (``meta.llama-3.3-70b-instruct``, ``cohere.command-a-03-2025``, ...) have
  entries of their own.
* **Unknown models** return ``None`` rather than a default — callers
  choose how to handle it (a conservative fallback, a log warning, or
  the existing ``ModelConfig`` values).
"""

from __future__ import annotations

import re
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
# OpenAI-compatible routing prefix in ``locus.models.providers`` (a unit test
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
        # OCI Generative AI; ``oci-responses`` for a Responses binding named
        # through it. What remains is the ``<vendor>.<model>`` id.
        "oci",
        "oci-responses",
        # locus.models.providers.COMPATIBLE_PROVIDERS
        "ollama",
        "vllm",
        "lmstudio",
        "llamacpp",
        "litellm",
        "groq",
        "together",
        "openrouter",
        "runpod",
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


#: The vendors OCI Generative AI names in a model id, ``<vendor>.<model>``:
#: ``openai.gpt-4.1`` is OpenAI's ``gpt-4.1`` served by OCI. Only these
#: segments are dropped, so a dot inside a slug (``qwen3.6-35b``) is kept.
_OCI_VENDORS: Final[frozenset[str]] = frozenset(
    {"openai", "meta", "cohere", "google", "xai", "mistral", "anthropic"}
)


def _oci_bare(key: str) -> str | None:
    """The model's own slug from an OCI ``<vendor>.<model>`` id, else ``None``."""
    vendor, dot, rest = key.partition(".")
    if dot and rest and vendor.lower() in _OCI_VENDORS:
        return rest
    return None


#: A snapshot suffix on a model slug: ``-20250929``, ``-2025-09-29``,
#: Vertex's ``@20251101``, or ``-latest``.
_SNAPSHOT_SUFFIX: Final[re.Pattern[str]] = re.compile(
    r"(?:[-@](?:\d{8}|\d{4}-\d{2}-\d{2})|-latest)$"
)

#: A Claude slug's ``-<major>.<minor>`` version, as OpenRouter writes it.
_CLAUDE_DOTTED: Final[re.Pattern[str]] = re.compile(r"^(claude-[a-z]+-\d+)\.(\d)$")

_lock = threading.Lock()
_registry: dict[str, ModelMetadata] = {}


def register_metadata(md: ModelMetadata) -> None:
    """Register or overwrite a :class:`ModelMetadata` entry.

    Call at import time from user code to add fine-tunes, regional
    aliases, or self-hosted models that Locus doesn't ship
    seed data for, or to price a model at the rate your route bills
    (an OCI id such as ``"oci:openai.gpt-5.5"`` at OCI's rate rather
    than the vendor's list price).

    The entry is stored under the same key :func:`metadata_for` looks up:
    a provider prefix Locus knows (``openai:``, ``oci:``, ``openrouter:``
    and the rest) is stripped, so ``"oci:openai.gpt-5.5"`` and
    ``"openai.gpt-5.5"`` name one entry, and that entry wins over the
    built-in record of the model the OCI id serves (``gpt-5.5``). An
    unknown prefix is part of the slug and is kept.
    """
    key = _strip_prefix(md.model_id.strip())
    if key != md.model_id:
        md = md.model_copy(update={"model_id": key})
    with _lock:
        _registry[key] = md


def metadata_for(model_id: str) -> ModelMetadata | None:
    """Return the metadata record for ``model_id`` or ``None``.

    Accepts both bare (``"gpt-4o"``) and prefixed (``"openai:gpt-4o"``)
    forms. Only prefixes Locus's providers use are stripped; anything
    else is treated as part of the slug. An OCI Generative AI id,
    ``<vendor>.<model>`` with or without ``oci:``, resolves the entry of
    the model it serves: ``"oci:openai.gpt-4.1"`` is ``gpt-4.1`` — for its
    window and output cap only. OCI bills at its own rates, so the prices
    come back ``None`` unless a rate was registered under the OCI id.
    """
    key = _strip_prefix(model_id.strip())
    with _lock:
        found = _lookup(key)
        if found is None:
            bare = _oci_bare(key)
            if bare is not None:
                served = _lookup(bare)
                if served is not None:
                    found = _unpriced(served)
        return found


def _unpriced(md: ModelMetadata) -> ModelMetadata:
    """``md`` with its prices dropped.

    An OCI Generative AI id answers with the window and output cap of the
    model it serves, never with that model's list price: OCI bills at its
    own rates, which the built-in table does not carry. A rate registered
    under the OCI id itself is found before this step and keeps its prices.
    """
    if md.input_price_per_mtok is None and md.output_price_per_mtok is None:
        return md
    return md.model_copy(update={"input_price_per_mtok": None, "output_price_per_mtok": None})


def oci_served_slug(model_id: str) -> str | None:
    """The slug an OCI Generative AI id serves, or ``None`` for any other id.

    ``"oci:openai.gpt-5.5"`` and ``"openai.gpt-5.5"`` give ``"gpt-5.5"``;
    ``"openai:gpt-5.5"`` gives ``None``.
    """
    return _oci_bare(_strip_prefix(model_id.strip()))


def _lookup(key: str) -> ModelMetadata | None:
    """``key``'s entry, its base entry for a snapshot, or a Claude respelling's."""
    found = _registry.get(key)
    if found is None:
        # A dated snapshot or a ``-latest`` alias is the same model as its
        # base slug, priced the same. Only those exact suffix shapes are
        # stripped: a looser prefix match would let ``gpt-5`` answer for
        # ``gpt-5.5``, which is a different model at a different price —
        # and a wrong price under a hard ``max_cost_usd`` is worse than
        # none.
        base = _SNAPSHOT_SUFFIX.sub("", key)
        if base != key:
            found = _registry.get(base)
    if found is None:
        # OpenRouter's spelling of a Claude model, with or without its
        # ``anthropic/`` vendor part, as the Anthropic binding sends it to
        # OpenRouter's Messages API: the same model at the same list price
        # (``anthropic/claude-sonnet-5.5`` is ``claude-sonnet-5-5``).
        bare = key.removeprefix("anthropic/")
        if _CLAUDE_DOTTED.match(bare) or bare != key:
            found = _registry.get(_CLAUDE_DOTTED.sub(r"\1-\2", bare))
    return found


def model_id_of(model: object) -> str | None:
    """The model slug a model object (or a string id) is called with.

    A string is returned as-is. An object is read the way the rest of the
    SDK reads it: ``model.config.model`` (every native binding), else
    ``model.config.model_id`` (the OCI SDK transport, ``OCIModel``, names its
    model there), else a string ``model.model``. Wrappers that proxy attribute
    access to an inner model — a ``FallbackChain`` (its ``config`` is the
    primary tier's), or a caller's per-turn view that forwards
    ``__getattr__`` — resolve through the proxy. ``None`` when nothing names
    the model.
    """
    if isinstance(model, str):
        return model
    config = getattr(model, "config", None)
    for name in (
        getattr(config, "model", None),
        getattr(config, "model_id", None),
        getattr(model, "model", None),
    ):
        if isinstance(name, str) and name:
            return name
    return None


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
# list tight — only models that users of Locus are likely to touch.
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
    input_price_per_mtok="1.25",
    output_price_per_mtok="10.00",
)
_seed(
    "gpt-5-mini",
    family="openai",
    context_length=400_000,
    max_output_tokens=64_000,
    supports_prompt_caching=True,
    input_price_per_mtok="0.25",
    output_price_per_mtok="2.00",
)
_seed(
    "o1",
    family="openai",
    context_length=200_000,
    max_output_tokens=100_000,
    input_price_per_mtok="15.00",
    output_price_per_mtok="60.00",
)
_seed(
    "o3",
    family="openai",
    context_length=200_000,
    max_output_tokens=100_000,
    input_price_per_mtok="2.00",
    output_price_per_mtok="8.00",
)
_seed(
    "o4-mini",
    family="openai",
    context_length=200_000,
    max_output_tokens=100_000,
    input_price_per_mtok="1.10",
    output_price_per_mtok="4.40",
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

# OpenAI GPT-5.1: the GPT-5 generation's window and output cap, as OpenAI
# publishes them. OCI Generative AI serves it as ``openai.gpt-5.1``.
_seed(
    "gpt-5.1",
    family="openai",
    context_length=400_000,
    max_output_tokens=128_000,
    supports_prompt_caching=True,
    input_price_per_mtok="1.25",
    output_price_per_mtok="10.00",
)

# OpenAI GPT-5.5. Prices and window are the listed rates locus-code's price
# table carried (2026-08); the output cap follows the GPT-5 generation.
for _slug, _in, _out in (
    ("gpt-5.5", "1.25", "10.00"),
    ("gpt-5.5-mini", "0.25", "2.00"),
    ("gpt-5.5-nano", "0.05", "0.40"),
):
    _seed(
        _slug,
        family="openai",
        context_length=400_000,
        max_output_tokens=128_000,
        supports_prompt_caching=True,
        input_price_per_mtok=_in,
        output_price_per_mtok=_out,
    )

# OpenAI GPT-5.6 and GPT-6 generations — windows, output caps and standard-tier
# list prices as the model pages publish them (read 2026-11-24): every one takes
# a 1,050,000-token window (922,000 of input) and returns up to 128,000 tokens.
# Prices are the short-context rates; above 272,000 input tokens OpenAI bills
# input at twice and output at 1.5 times these figures for the whole request,
# which this table does not model. OCI Generative AI serves the same ids as
# ``openai.<slug>`` at its own rates; through an OCI id these entries answer
# with window and cap only. ``gpt-5.6-sol``'s rate is the promotional one the page
# lists through 2026-11-30; refresh it when the standard rate is published.
for _slug, _in, _out in (
    ("gpt-5.6-luna", "0.20", "1.20"),
    ("gpt-5.6-sol", "4.00", "20.00"),
    ("gpt-5.6-terra", "2.00", "12.00"),
    ("gpt-6-astra", "10.00", "50.00"),
    ("gpt-6-sol", "2.00", "10.00"),
    ("gpt-6-luna", "0.10", "0.50"),
    ("gpt-6.1-sol", "2.00", "10.00"),
    ("gpt-6.1-terra", "1.00", "5.00"),
    ("gpt-6.2-astra", "8.00", "40.00"),
    ("gpt-6.2-sol", "2.00", "10.00"),
):
    _seed(
        _slug,
        family="openai",
        context_length=1_050_000,
        max_output_tokens=128_000,
        supports_prompt_caching=True,
        input_price_per_mtok=_in,
        output_price_per_mtok=_out,
    )

# The rest of the OpenAI line that OCI Generative AI serves as ``openai.<slug>``
# (windows, output caps and standard-tier short-context list prices as the
# OpenAI model pages publish them, read 2026-11-24). Several of these are
# deprecated or shut down on OpenAI's own API and still served by OCI; dated
# snapshots (``gpt-5.2-2025-12-11``) resolve to these base records. The
# ``-codex`` and ``-pro`` models answer only on the Responses endpoint. The two
# audio models carry their text rates; audio tokens are billed separately.
for _slug, _window, _output, _in, _out in (
    ("gpt-4.1-nano", 1_047_576, 32_768, "0.10", "0.40"),
    ("gpt-4o-search-preview", 128_000, 16_384, "2.50", "10.00"),
    ("gpt-4o-mini-search-preview", 128_000, 16_384, "0.15", "0.60"),
    ("gpt-5-codex", 400_000, 128_000, "1.25", "10.00"),
    ("gpt-5-nano", 400_000, 128_000, "0.05", "0.40"),
    ("gpt-5.1-chat-latest", 128_000, 16_384, "1.25", "10.00"),
    ("gpt-5.1-codex", 400_000, 128_000, "1.25", "10.00"),
    ("gpt-5.1-codex-max", 400_000, 128_000, "1.25", "10.00"),
    ("gpt-5.1-codex-mini", 400_000, 128_000, "0.25", "2.00"),
    ("gpt-5.2", 400_000, 128_000, "1.75", "14.00"),
    ("gpt-5.2-chat-latest", 128_000, 16_384, "1.75", "14.00"),
    ("gpt-5.2-codex", 400_000, 128_000, "1.75", "14.00"),
    ("gpt-5.2-pro", 400_000, 128_000, "21.00", "168.00"),
    ("gpt-5.3-codex", 400_000, 128_000, "1.75", "14.00"),
    ("gpt-5.4", 1_050_000, 128_000, "2.50", "15.00"),
    ("gpt-5.4-mini", 400_000, 128_000, "0.75", "4.50"),
    ("gpt-5.4-nano", 400_000, 128_000, "0.20", "1.25"),
    ("gpt-5.4-pro", 1_050_000, 128_000, "30.00", "180.00"),
    ("gpt-5.5-pro", 1_050_000, 128_000, "30.00", "180.00"),
    ("o3-mini", 200_000, 100_000, "1.10", "4.40"),
    ("gpt-audio", 128_000, 16_384, "2.50", "10.00"),
    ("gpt-audio-mini", 128_000, 16_384, "0.60", "2.40"),
):
    _seed(
        _slug,
        family="openai",
        context_length=_window,
        max_output_tokens=_output,
        supports_prompt_caching=True,
        input_price_per_mtok=_in,
        output_price_per_mtok=_out,
    )
del _slug, _window, _output, _in, _out

# OpenAI's open-weight models, served by OCI as ``openai.gpt-oss-120b`` and
# ``openai.gpt-oss-20b``: 131,072-token window and output cap per the OpenAI
# model pages (2026-11-24). No prices — they have no OpenAI list price and OCI
# bills at its own rates, which a budget registers itself.
for _slug in ("gpt-oss-120b", "gpt-oss-20b"):
    _seed(_slug, family="openai", context_length=131_072, max_output_tokens=131_072)
del _slug

# Anthropic, current generation — ids, windows, output caps and first-party
# list prices as published 2026-09-25. Every one of these takes the full 1M
# window by default and caps output at 128K, except Haiku 4.5 (200K / 64K).
for _slug, _in, _out in (
    ("claude-fable-5-1", "10.00", "50.00"),
    ("claude-fable-5", "10.00", "50.00"),
    ("claude-opus-5-5", "4.00", "20.00"),
    ("claude-opus-5", "5.00", "25.00"),
    ("claude-opus-4-8", "5.00", "25.00"),
    ("claude-opus-4-7", "5.00", "25.00"),
    ("claude-opus-4-6", "5.00", "25.00"),
    ("claude-sonnet-5-5", "2.00", "10.00"),
    ("claude-sonnet-5", "2.00", "10.00"),
    ("claude-sonnet-4-6", "3.00", "15.00"),
):
    _seed(
        _slug,
        family="anthropic",
        context_length=1_000_000,
        max_output_tokens=128_000,
        supports_prompt_caching=True,
        input_price_per_mtok=_in,
        output_price_per_mtok=_out,
    )
_seed(
    "claude-haiku-4-5",
    family="anthropic",
    context_length=200_000,
    max_output_tokens=64_000,
    supports_prompt_caching=True,
    input_price_per_mtok="1.00",
    output_price_per_mtok="5.00",
)
del _slug, _in, _out

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

# Models OCI Generative AI serves under ``<vendor>.<model>`` ids and that no
# other entry covers. Windows are the ones the OCI Generative AI documentation
# states per model (read 2026-11-16); where another route lists a larger one
# (xAI's own API for Grok 4) the OCI figure is the smaller and safe for both.
# Output caps: the OCI pages say on-demand serving caps a Meta or Cohere reply
# at 4,000 tokens, but the service does not hold the Meta models to that on
# the OpenAI-compatible endpoint. Measured 2026-11-24: Llama 3.3 70B produced
# 6,000 tokens when asked and names 131,072 as its model length, Llama 4
# Maverick names 524,288 and Scout 196,608 ("supports at most N completion
# tokens"), so those three carry the service's model length as window and cap;
# the Cohere Command A models refuse ``maxTokens`` above 4,096, which is their
# cap. Unmeasured models keep the documented figures. Gemini defaults to 65,536
# and Grok 4's playground cap is 16,000. No prices: OCI bills at its own rates,
# which a budget registers itself.
for _slug, _family, _window, _output in (
    ("llama-3.3-70b-instruct", "meta", 131_072, 131_072),
    ("llama-4-maverick-17b-128e-instruct-fp8", "meta", 524_288, 524_288),
    ("command-a-03-2025", "cohere", 256_000, 4_096),
    ("command-r-plus-08-2024", "cohere", 128_000, 4_000),
    ("gemini-2.5-pro", "google", 1_048_576, 65_536),
    ("gemini-2.5-flash", "google", 1_048_576, 65_536),
    ("grok-4", "xai", 128_000, 16_000),
    # Added 2026-11-24 from the OCI model pages of that day. Llama 4 Scout
    # carries the model length the service names (and it produced 8,000
    # tokens when asked); Command A Vision refuses maxTokens above 4,096 like
    # Command A. Llama 3.1 405B and Command A Reasoning were not served
    # on-demand in the regions measured and keep the documented figures. Grok
    # 4.3 and 4.20 pages give a 131,000 playground cap; the Grok 4.7 and 4.6
    # pages give none and the service accepted any value asked, so they carry
    # their siblings' figure. The 4.20 variants are the ids OCI lists for one
    # model.
    ("command-a-reasoning", "cohere", 256_000, 4_000),
    ("command-a-vision", "cohere", 128_000, 4_096),
    ("llama-3.1-405b-instruct", "meta", 128_000, 4_000),
    ("llama-4-scout-17b-16e-instruct", "meta", 196_608, 196_608),
    ("gemini-2.5-flash-lite", "google", 1_048_576, 65_536),
    ("grok-4.7", "xai", 500_000, 131_000),
    ("grok-4.6", "xai", 500_000, 131_000),
    ("grok-4.3", "xai", 1_000_000, 131_000),
    ("grok-4.20", "xai", 1_000_000, 131_000),
    ("grok-4.20-reasoning", "xai", 1_000_000, 131_000),
    ("grok-4.20-non-reasoning", "xai", 1_000_000, 131_000),
    ("grok-4.20-0309-reasoning", "xai", 1_000_000, 131_000),
    ("grok-4.20-0309-non-reasoning", "xai", 1_000_000, 131_000),
    ("grok-4.20-multi-agent", "xai", 1_000_000, 131_000),
    ("grok-4.20-multi-agent-0309", "xai", 1_000_000, 131_000),
):
    _seed(_slug, family=_family, context_length=_window, max_output_tokens=_output)
del _slug, _family, _window, _output
