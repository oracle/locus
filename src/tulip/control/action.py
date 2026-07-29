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

**Labels can depend on the arguments.** A tool's declared labels are static, but
``refund(amount=10)`` and ``refund(amount=10_000)`` are not the same action. A
tool definition may therefore carry a ``derive`` list — declarative rules,
evaluated against the call's arguments, that add tags or raise the blast radius
(see :func:`derive_labels`). The rules are data, never code: comparisons only,
no ``eval``, no callables, so a tool cannot talk its own labels down.

Originally written for ``tulip-frameworks``; promoted here so the SDK, the
gateway and the registry share one implementation rather than three.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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


#: Tag added when a rule could not be evaluated — the argument it names was
#: absent, was of an unusable type, or the rule itself was malformed. Never
#: silently treat "we could not tell" as safe: a policy can gate on this tag.
UNDETERMINED_TAG = "undetermined"

#: Effect keys a ``derive`` rule may carry alongside its ``when``.
_EFFECT_KEYS = frozenset({"add_tags", "set_kind", "set_environment"})


def _number(value: Any) -> float | None:
    """``value`` as a number, or ``None`` if it is not one.

    ``bool`` is excluded on purpose: ``True > 500`` is not a comparison anyone
    meant to write, and silently reading it as ``1`` would let a flag argument
    decide a threshold rule.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _compare_ordered(arg: Any, want: Any, op: str) -> bool | None:
    left, right = _number(arg), _number(want)
    if left is None or right is None:
        return None
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "lt":
        return left < right
    return left <= right


def _compare_in(arg: Any, want: Any) -> bool | None:
    if isinstance(want, str) or not isinstance(want, Sequence | set | frozenset):
        return None
    return any(arg == candidate for candidate in want)


def _compare_contains(arg: Any, want: Any) -> bool | None:
    if isinstance(arg, str):
        return want in arg if isinstance(want, str) else None
    if isinstance(arg, Sequence | set | frozenset):
        return any(want == item for item in arg)
    return None


#: The whole comparison vocabulary. Each returns ``True``/``False``, or ``None``
#: meaning "undeterminable" — an unusable argument or an unusable rule value.
_COMPARATORS: dict[str, Callable[[Any, Any], bool | None]] = {
    "gt": lambda a, w: _compare_ordered(a, w, "gt"),
    "gte": lambda a, w: _compare_ordered(a, w, "gte"),
    "lt": lambda a, w: _compare_ordered(a, w, "lt"),
    "lte": lambda a, w: _compare_ordered(a, w, "lte"),
    "eq": lambda a, w: bool(a == w),
    "ne": lambda a, w: bool(a != w),
    "in": _compare_in,
    "contains": _compare_contains,
}


def _evaluate_when(when: Any, kwargs: Mapping[str, Any]) -> bool | None:
    """Evaluate a ``when`` clause. ``None`` = could not tell (fail safe)."""
    if not isinstance(when, Mapping):
        return None
    arg = when.get("arg")
    operators = [key for key in when if key in _COMPARATORS]
    if not isinstance(arg, str) or not arg or len(operators) != 1 or len(when) != 2:
        return None
    if arg not in kwargs:
        return None
    return _COMPARATORS[operators[0]](kwargs[arg], when[operators[0]])


def _string_list(value: Any) -> list[str] | None:
    """A list of non-empty strings, or ``None`` if ``value`` is not that."""
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list | tuple):
        return None
    if not items or any(not isinstance(item, str) or not item for item in items):
        return None
    return [str(item) for item in items]


class DerivedLabels:
    """Accumulator for what a ``derive`` list adds to an action."""

    def __init__(self) -> None:
        self.tags: set[str] = set()
        self.kind: str = ""
        self.environment: str = ""
        self.blast_radius: int | None = None
        self.undetermined: bool = False

    def mark_undetermined(self) -> None:
        self.undetermined = True

    def raise_radius(self, value: int) -> None:
        """Deriving may raise the blast radius; it may never lower it."""
        self.blast_radius = value if self.blast_radius is None else max(self.blast_radius, value)


def _apply_effects(rule: Mapping[str, Any], derived: DerivedLabels) -> None:
    """Apply a matched rule's effects, or mark it undetermined if malformed."""
    tags = rule.get("add_tags")
    kind = rule.get("set_kind")
    environment = rule.get("set_environment")

    resolved_tags: list[str] = []
    if tags is not None:
        checked = _string_list(tags)
        if checked is None:
            derived.mark_undetermined()
            return
        resolved_tags = checked
    for value in (kind, environment):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            derived.mark_undetermined()
            return

    derived.tags.update(resolved_tags)
    if isinstance(kind, str):
        derived.kind = kind.strip()
    if isinstance(environment, str):
        derived.environment = environment.strip()


def _count_of(value: Any) -> int | None:
    """``len()`` of a collection argument. Strings are not collections here."""
    if isinstance(value, str) or not isinstance(value, Sequence | set | frozenset | Mapping):
        return None
    return len(value)


def _value_of(value: Any) -> int | None:
    """``int()`` of a numeric argument, or ``None`` if it is not one."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _apply_blast_radius(spec: Any, kwargs: Mapping[str, Any], derived: DerivedLabels) -> None:
    """Apply a ``{"blast_radius": {...}}`` rule."""
    if not isinstance(spec, Mapping):
        derived.mark_undetermined()
        return
    arg = spec.get("arg")
    agg = spec.get("agg")
    default = spec.get("default")
    known = {"arg", "agg", "default"}
    bad_default = default is not None and (
        isinstance(default, bool) or not isinstance(default, int)
    )
    if (
        not isinstance(arg, str)
        or not arg
        or agg not in ("count", "value")
        or bad_default
        or any(key not in known for key in spec)
    ):
        derived.mark_undetermined()
        return

    read = _count_of if agg == "count" else _value_of
    resolved = read(kwargs[arg]) if arg in kwargs else None
    if resolved is None:
        derived.mark_undetermined()
        if isinstance(default, int):
            derived.raise_radius(default)
        return
    derived.raise_radius(resolved)


def _apply_rule(rule: Any, kwargs: Mapping[str, Any], derived: DerivedLabels) -> None:
    if not isinstance(rule, Mapping):
        derived.mark_undetermined()
        return

    if "blast_radius" in rule:
        if any(key not in ("blast_radius", "when") for key in rule):
            derived.mark_undetermined()
            return
        if "when" in rule:
            matched = _evaluate_when(rule["when"], kwargs)
            if matched is None:
                derived.mark_undetermined()
                return
            if not matched:
                return
        _apply_blast_radius(rule["blast_radius"], kwargs, derived)
        return

    if (
        "when" not in rule
        or not (_EFFECT_KEYS & set(rule))
        or any(key != "when" and key not in _EFFECT_KEYS for key in rule)
    ):
        derived.mark_undetermined()
        return

    matched = _evaluate_when(rule["when"], kwargs)
    if matched is None:
        derived.mark_undetermined()
        return
    if matched:
        _apply_effects(rule, derived)


def derive_labels(rules: Any, kwargs: Mapping[str, Any]) -> DerivedLabels:
    """Evaluate a tool's ``derive`` rules against one call's arguments.

    Declarative and total: comparisons only, never ``eval``, never a callable,
    so nothing a tool receives can execute during labelling. Rules apply in
    order and all matching rules apply. Anything that cannot be evaluated — a
    missing argument, an argument of the wrong type, a malformed rule — is
    skipped *and* records :data:`UNDETERMINED_TAG`, so "we could not tell"
    reaches the policy as a fact rather than as silence.
    """
    derived = DerivedLabels()
    if rules is None:
        return derived
    if not isinstance(rules, list | tuple):
        derived.mark_undetermined()
        return derived
    for rule in rules:
        _apply_rule(rule, kwargs, derived)
    return derived


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

    ``labels["derive"]`` may carry argument-derived rules (see
    :func:`derive_labels`) — the only part of this that reads ``kwargs`` for
    labelling. Derived tags join the declared ones, a derived ``set_kind`` /
    ``set_environment`` wins over the declared value (it describes *this* call),
    and a derived blast radius only ever raises the declared one. With no
    ``derive`` key the result is exactly what it was before.
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

    derived = derive_labels(declared.get("derive"), kwargs)
    if derived.blast_radius is not None:
        resolved_radius = max(resolved_radius, derived.blast_radius)

    tags = {name, *(str(t) for t in declared_tags if t), *derived.tags}
    if derived.undetermined:
        tags.add(UNDETERMINED_TAG)

    return default_action(
        name,
        kwargs,
        environment=(
            derived.environment or declared_env or (environment or "").strip() or "unknown"
        ),
        kind=derived.kind or declared_kind,
        blast_radius=resolved_radius,
        tags=frozenset(tags),
    )


__all__ = [
    "UNDETERMINED_TAG",
    "ActionSpec",
    "DerivedLabels",
    "action_from_labels",
    "asset_from_args",
    "default_action",
    "derive_labels",
    "resolve_action",
]
