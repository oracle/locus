# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Auxiliary (cheap / fast) model helper.

Some agent-side operations — context compaction, claim extraction,
multi-hop query refinement — need an LLM call but don't need the
quality of the primary reasoning model. Running them on a
``gpt-4o-mini`` / ``claude-haiku-4`` tier is both cheaper and faster
and frees primary-model headroom for the actual task.

:func:`resolve_auxiliary` consolidates the "if auxiliary is set use
it, otherwise fall back to the primary" pattern so compaction,
reflexion, and other helpers don't each reinvent it.

The primary model resolution remains the caller's responsibility.
This helper is a few lines of glue, not a parallel model registry.
"""

from __future__ import annotations

from typing import Any


__all__ = ["resolve_auxiliary"]


def resolve_auxiliary(
    primary: Any,
    auxiliary: Any | None,
) -> Any:
    """Return the auxiliary model to use, falling back to ``primary``.

    Args:
        primary: The agent's primary model (string or ModelProtocol
            instance). Used as the fallback when ``auxiliary`` is
            ``None``.
        auxiliary: The auxiliary model from
            :attr:`AgentConfig.auxiliary_model`. May be ``None`` (use
            primary), a string (``'openai:gpt-4o-mini'``), or a
            ModelProtocol instance.

    Returns:
        The model to use for the helper call. Never returns ``None``
        — callers can trust the return value.

    Raises:
        ValueError: When ``primary`` is ``None`` and ``auxiliary`` is
            also ``None``. A caller that reaches this helper without a
            primary model is misconfigured.
    """
    chosen = auxiliary if auxiliary is not None else primary
    if chosen is None:
        raise ValueError("no auxiliary or primary model configured — AgentConfig.model must be set")
    return chosen
