# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Prompt-cache breakpoint helpers.

Anthropic (and, as of 2026, a growing set of providers including
OCI-hosted Claude, Bedrock, and Gemini) supports ephemeral prompt
caching: mark a message as a cache checkpoint and subsequent requests
that share the same prefix reuse the provider's computation at a
fraction of the input cost.

This module does **not** speak any provider-specific protocol. It
exposes two helpers that stamp a cache-control marker on messages so
provider-adapter code can translate the marker into the appropriate
wire format when it exists, and ignore it when it doesn't.

Usage::

    from locus.models.caching import mark_cache_breakpoint
    from locus.models.metadata import metadata_for

    system = Message.system("You are a helpful assistant …")
    meta = metadata_for(model_id)
    if meta and meta.supports_prompt_caching:
        system = mark_cache_breakpoint(system)

Provider adapters check ``message.metadata.get("cache_control")`` and
emit the provider's native breakpoint representation (for Anthropic:
``{"type": "ephemeral"}`` on the last content block). Adapters that
don't support caching ignore the field — messages with the marker
remain valid Pydantic models everywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from locus.core.messages import Message


__all__ = [
    "CACHE_CONTROL_KEY",
    "is_cache_breakpoint",
    "mark_cache_breakpoint",
]


#: Key used in :attr:`Message.metadata` to signal a cache breakpoint.
#: Kept stable so provider adapters can rely on it.
CACHE_CONTROL_KEY = "cache_control"


def mark_cache_breakpoint(
    message: Message,
    *,
    ttl: Literal["ephemeral"] = "ephemeral",
) -> Message:
    """Return a copy of ``message`` tagged as a cache checkpoint.

    Args:
        message: The message to mark. Must be a :class:`Message` —
            but since it's immutable, a new instance is returned with
            the same fields plus a ``cache_control`` metadata entry.
        ttl: Cache tier. Currently only ``"ephemeral"`` is defined
            (Anthropic's 5-minute cache). Reserved for future cache
            tiers (Anthropic 1-hour, Bedrock 24-hour, …).

    Returns:
        A new :class:`Message` with ``metadata[CACHE_CONTROL_KEY]``
        set. Providers that don't honour caching ignore the field.
    """
    existing = dict(message.metadata or {})
    existing[CACHE_CONTROL_KEY] = {"type": ttl}
    return message.model_copy(update={"metadata": existing})


def is_cache_breakpoint(message: Message) -> bool:
    """Return True when ``message`` carries a cache-breakpoint marker."""
    if not message.metadata:
        return False
    marker = message.metadata.get(CACHE_CONTROL_KEY)
    return isinstance(marker, dict) and "type" in marker
