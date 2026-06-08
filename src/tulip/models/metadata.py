# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

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
from typing import Final

from pydantic import BaseModel, Field


__all__ = [
    "ModelMetadata",
    "known_models",
    "metadata_for",
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


# Provider prefixes stripped at lookup time. Trimmed to the ones Tulip
# actually ships bindings for — users supplying a different prefix can
# register metadata under the canonical slug directly.
_PROVIDER_PREFIXES: Final[frozenset[str]] = frozenset(
    {
        "openai",
        "anthropic",
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
