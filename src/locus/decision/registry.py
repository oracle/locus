# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Decision providers by name: ``"oci:<model>"``, ``"openai:<model>"``.

The same ``provider:model`` spelling :func:`~locus.models.registry.get_model`
reads, for decision models. It is a separate registry on purpose: a decision
model answers typed questions, not a conversation, so
``get_decider("oci:openai.gpt-6-luna")`` is a decider and
``Agent(model="oci:openai.gpt-6-luna")`` is not.

    from locus.decision import get_decider

    oci = get_decider("oci:openai.gpt-6-luna", profile="DEFAULT", region="eu-frankfurt-1")
    oai = get_decider("openai:gpt-6-luna")   # key from OPENAI_API_KEY
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from locus.decision.provider import DecisionProvider


__all__ = ["get_decider", "list_deciders", "register_decider"]

#: prefix -> factory, called as ``factory(model_id, **kwargs)``.
_DECIDERS: dict[str, Callable[..., DecisionProvider]] = {}


def register_decider(prefix: str, factory: Callable[..., DecisionProvider]) -> None:
    """Register a decision provider under ``prefix``.

    The factory is called as ``factory(model_id, **kwargs)``: ``model_id`` is
    the part of the string after ``prefix:`` and ``kwargs`` are the keyword
    arguments :func:`get_decider` was given. Registering a prefix again
    replaces its factory.
    """
    if not prefix or ":" in prefix:
        raise ValueError(f"a decider prefix is a non-empty name without ':', got {prefix!r}")
    _DECIDERS[prefix] = factory


def get_decider(spec: str, **kwargs: Any) -> DecisionProvider:
    """The decision provider named by ``spec``, a ``"provider:model"`` string.

    Raises:
        ValueError: If ``spec`` has no ``provider:`` prefix, names no model,
            or names an unknown provider.
    """
    provider, sep, model_id = spec.partition(":")
    if not sep or not provider or not model_id:
        raise ValueError(f"a decider is named 'provider:model', got {spec!r}")
    factory = _DECIDERS.get(provider)
    if factory is None:
        raise ValueError(f"unknown decider provider {provider!r}; registered: {list_deciders()}")
    return factory(model_id, **kwargs)


def list_deciders() -> list[str]:
    """The registered provider prefixes."""
    return sorted(_DECIDERS)


def _register_defaults() -> None:
    from locus.decision.api import OCIDecider, OpenAIDecider  # noqa: PLC0415

    register_decider("openai", OpenAIDecider)
    register_decider("oci", OCIDecider)


_register_defaults()
