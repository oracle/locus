# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Deriving the :class:`Action` a policy is weighed against.

A :class:`~tulip.security.policy.ControlPolicy` matches on
``Action.labels()`` — ``{environment, kind, *tags}``. The action's *name* and
*asset* are deliberately not labels, so a policy that says "hold anything in
production" or "deny anything irreversible" can only work if something puts
those labels on the action in the first place.

That is what this module is for. Every caller that governs a tool call — the
gateway's run loop, the registry's run engine, the framework adapters — needs
the same derivation, and the failure mode when they each invent their own is
severe and quiet: an action labelled ``environment="staging"`` by default never
matches ``require_human_for={"production"}``, so the gate returns ALLOW and
records that all policy checks passed.

**Fail safe, not open.** With nothing known about an action, the environment is
``"unknown"`` rather than a plausible-looking guess. Combined with the stock
policy — which requires a verification score — an unlabelled consequential call
lands on ``require_human`` instead of sliding through.

Originally written for ``tulip-frameworks``; promoted here so the SDK, the
gateway and the registry share one implementation rather than three.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from tulip.security.policy import Action


#: An :class:`Action`, or a callable that derives one from ``(name, kwargs)``.
ActionSpec = Action | Callable[[str, Mapping[str, Any]], Action]

#: Argument names that commonly name the asset an action touches, in priority
#: order. Best-effort: an unrecognised signature yields an empty asset, which
#: costs an audit record some detail but never changes a policy decision.
_ASSET_KEYS = (
    "asset",
    "host",
    "hostname",
    "target",
    "resource",
    "resource_id",
    "account",
    "user",
    "username",
    "email",
    "order_id",
    "id",
    "ip",
    "url",
)


def asset_from_args(kwargs: Mapping[str, Any]) -> str:
    """Best-effort asset label from a tool call's arguments."""
    for key in _ASSET_KEYS:
        value = kwargs.get(key)
        if value:
            return str(value)
    return ""


def default_action(
    name: str,
    kwargs: Mapping[str, Any],
    *,
    environment: str = "unknown",
    kind: str = "",
    blast_radius: int = 1,
    tags: frozenset[str] | None = None,
) -> Action:
    """A conservative :class:`Action` for ``name`` when none was supplied.

    Fail-safe by construction: ``environment="unknown"`` plus the stock
    :class:`~tulip.security.policy.ControlPolicy` (which requires a verification
    score) lands an un-verified call on ``require_human`` rather than
    auto-allowing it.

    ``tags`` defaults to the action's own name, so a policy can always gate one
    specific tool by naming it — the one thing that worked before labels were
    derived at all.
    """
    return Action(
        name=name,
        asset=asset_from_args(kwargs),
        blast_radius=blast_radius,
        environment=environment,
        kind=kind,
        tags=tags if tags is not None else frozenset({name}),
    )


def resolve_action(spec: ActionSpec | None, name: str, kwargs: Mapping[str, Any]) -> Action:
    """Resolve an :class:`ActionSpec` (or ``None``) into a concrete :class:`Action`."""
    if spec is None:
        return default_action(name, kwargs)
    if isinstance(spec, Action):
        return spec
    return spec(name, kwargs)


def action_from_labels(
    name: str,
    kwargs: Mapping[str, Any],
    *,
    labels: Mapping[str, Any] | None = None,
    environment: str | None = None,
    blast_radius: int = 1,
) -> Action:
    """Build an :class:`Action` from a tool's declared labels.

    ``labels`` is what a tool definition declares about the actions it performs
    — ``environment``, ``kind``, ``blast_radius``, ``tags``. Anything absent
    falls back: the caller's ``environment`` (the agent's, or the deployment's),
    then ``"unknown"``.

    The tool's own name is always among the tags, so naming a tool in
    ``require_human_for`` keeps working regardless of what it declares.
    """
    declared = dict(labels or {})
    declared_env = str(declared.get("environment") or "").strip()
    declared_kind = str(declared.get("kind") or "").strip()
    declared_tags = declared.get("tags") or []
    if isinstance(declared_tags, str):
        declared_tags = [declared_tags]

    radius = declared.get("blast_radius")
    try:
        resolved_radius = int(radius) if radius is not None else blast_radius
    except (TypeError, ValueError):
        resolved_radius = blast_radius

    return default_action(
        name,
        kwargs,
        environment=declared_env or (environment or "").strip() or "unknown",
        kind=declared_kind,
        blast_radius=resolved_radius,
        tags=frozenset({name, *(str(t) for t in declared_tags if t)}),
    )


__all__ = [
    "ActionSpec",
    "action_from_labels",
    "asset_from_args",
    "default_action",
    "resolve_action",
]
