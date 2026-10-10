# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Who approves a step's tool calls: a step's approval rules, and which one applies.

A step's ``approval`` names who must approve its tool calls (the registry's
``StepApproval``). The old form names one group, ``by``; the new form is a list of
``rules``, first match wins, the default rule (one without ``when``) last::

    approval:
      ask: "Confirm the new account was verified with the vendor."
      show: [vendor_id, vendor_name]
      rules:
        - when: "inputs.amount > 10000"
          all_of: [{by: finance, count: 1}, {by: cfo, count: 1}]
          due: 4h
          escalate_to: finance-leads
        - by: finance
          count: 1
          due: 24h
      only_these_approvers: true

A rule's ``when`` is the playbook's own condition language
(:mod:`tulip.playbooks.v2.when`), read against the step's typed context -- the run's
``inputs``, the earlier steps' ``outputs`` -- plus the held call's arguments under
``args``. :func:`pick_rule` chooses one rule for one call:

* the first rule whose ``when`` is true applies (a rule without ``when`` always is);
* a rule whose ``when`` is ``unknown`` -- it reads a value the run cannot see, orders
  money in two currencies, orders an argument against a number when the call's argument
  is not a number, or does not parse -- is never a reason to ask fewer people: the
  STRICTEST of it and the rules after it applies (the most approvers in all; on a tie
  the first), never a laxer one;
* when no rule matches and the step names no default, the strictest rule applies.

What it returns, :class:`ResolvedApproval`, says who must approve (``groups``: each group
must give its ``count`` of approvals, all of them), by when (``due_seconds``), who it
escalates to, whether only those groups may decide (``only_named``), and why, in one plain
sentence. The gateway sends it with the hold (:meth:`ResolvedApproval.hold_fields`);
a local, open-source run enforces it with :func:`authority_from_resolved`.

Reading is tolerant, as for the rest of a v2 playbook: the registry validated the shape
strictly at publish, so a malformed rule is left out rather than refusing a run -- except
that a malformed ``when`` keeps its rule and reads as ``unknown``, so a typo can only
make a call need more approvers, never fewer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from tulip.playbooks.v2.fields import Field, validate_value
from tulip.playbooks.v2.when import (
    FALSE,
    TRUE,
    UNKNOWN,
    AllOf,
    Always,
    AnyOf,
    Compare,
    Contains,
    Empty,
    Literal,
    Money,
    Node,
    Not,
    Path,
    Truthy,
    WhenSyntaxError,
    parse_when,
    when_verdict,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from datetime import datetime

    from tulip.control.approvals import ApprovalAuthority, Delegation


#: How a rule was chosen (:attr:`ResolvedApproval.matched`).
MATCHED_WHEN = "when"
MATCHED_DEFAULT = "default"
MATCHED_UNKNOWN = "unknown"
MATCHED_NO_RULE = "no_rule"

#: The units a ``due`` is written in, in seconds.
_UNITS: Mapping[str, int] = {"m": 60, "h": 3600, "d": 86400}
_DURATION = re.compile(r"^\s*(\d{1,6})\s*([mhd])\s*$", re.IGNORECASE)

#: The registry's ``group_name`` exceptions: short words that are not acronyms.
_NOT_ACRONYMS = frozenset({"and", "for", "of", "the", "ops", "tax", "pay", "law", "new", "all"})
_ORDERING = frozenset({">", "<", ">=", "<="})
_MONEY = Field(name="amount", type="money")


def parse_duration(text: str) -> int:
    """Seconds in a ``due``: ``"30m"``, ``"4h"``, ``"2d"`` (minutes, hours, days).

    One whole number and one unit; case and surrounding spaces do not matter.

    Raises:
        ValueError: Anything else, or zero.
    """
    match = _DURATION.match(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError(
            f"a duration is a whole number of minutes, hours or days (30m, 4h, 2d); got {text!r}"
        )
    seconds = int(match.group(1)) * _UNITS[match.group(2).lower()]
    if seconds <= 0:
        raise ValueError(f"a duration is longer than zero; got {text!r}")
    return seconds


def group_name(label: str) -> str:
    """A group label as a person reads it: ``finance`` is "Finance", ``cfo`` is "CFO".

    The registry's ``group_name``, word for word, so a reason reads as its notices do.
    """
    words = [w for w in re.split(r"[-_.:\s]+", label) if w]
    return (
        " ".join(
            w.upper()
            if len(w) <= 3 and w.isalpha() and w.lower() not in _NOT_ACRONYMS
            else w[:1].upper() + w[1:]
            for w in words
        )
        or label
    )


@dataclass(frozen=True)
class ApprovalRule:
    """One rule of a step's approval: when it applies, and who must approve then.

    ``groups`` are ``(grant label, count)`` pairs, and every group must give its count
    (``by`` + ``count`` is one group, ``all_of`` several). ``when`` empty = the default
    rule, which always applies. ``when_error`` says why a ``when`` could not be read; such
    a rule reads as ``unknown``.
    """

    groups: tuple[tuple[str, int], ...]
    when: str = ""
    due_seconds: int | None = None
    escalate_to: str | None = None
    when_error: str = ""

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("an approval rule names at least one group")
        for label, count in self.groups:
            if not label or count < 1:
                raise ValueError("an approval group has a label and a count of at least 1")

    @property
    def default(self) -> bool:
        """Whether this rule always applies (it has no ``when``)."""
        return not self.when and not self.when_error

    @property
    def total(self) -> int:
        """How many approvals the rule needs in all."""
        return sum(count for _, count in self.groups)


def _default_rule(rules: Sequence[ApprovalRule]) -> bool:
    return any(rule.default for rule in rules)


@dataclass(frozen=True)
class StepApproval:
    """Who approves a step's tool calls, and what they are asked (``approval`` on a step).

    ``ask`` is the plain text the approver reads; ``show`` names the call's arguments to
    put in front of them first. ``rules`` say who must approve, first match wins
    (:func:`pick_rule`); ``only_these_approvers`` restricts the decision to the groups the
    applying rule names (a break-glass admin aside).

    ``by`` is the old form: one grant label, one approval. It is the default rule: when
    ``rules`` has no rule without ``when``, a ``by`` is appended as one, so
    ``StepApproval(by="finance")`` and ``StepApproval(rules=(...))`` read alike. When only
    ``rules`` are given, ``by`` is the default rule's first group (the last rule's, when
    none is the default), for readers that know only ``by``.
    """

    by: str = ""
    ask: str = ""
    show: tuple[str, ...] = ()
    rules: tuple[ApprovalRule, ...] = ()
    only_these_approvers: bool = False

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if self.by and not _default_rule(rules):
            rules = (*rules, ApprovalRule(groups=((self.by, 1),)))
        if not rules:
            raise ValueError("a step approval names who approves: by, or rules")
        object.__setattr__(self, "rules", rules)
        if not self.by:
            fallback = next((r for r in rules if r.default), rules[-1])
            object.__setattr__(self, "by", fallback.groups[0][0])


@dataclass(frozen=True)
class ResolvedApproval:
    """Who must approve one held call, chosen from its step's rules (:func:`pick_rule`).

    ``groups`` must each give their ``count`` of approvals, from distinct people;
    ``due_seconds`` is how long they have before the hold escalates to ``escalate_to``
    (``None`` = the deployment's default); ``only_named`` = nobody outside the groups
    decides it (a break-glass admin aside). ``rule_index`` is the position in
    :attr:`StepApproval.rules` of the rule that applies, ``matched`` how it was chosen
    (``when``, ``default``, ``unknown``, ``no_rule``), and ``reason`` says so in a sentence
    a person reads ("The amount is more than $10,000, so Finance and CFO must both approve.").
    """

    groups: tuple[tuple[str, int], ...]
    due_seconds: int | None
    escalate_to: str | None
    only_named: bool
    ask: str
    show: tuple[str, ...]
    rule_index: int
    reason: str
    matched: str = MATCHED_WHEN
    step: str = ""
    #: The ``when`` of the rule that could not be told, when ``matched`` is ``unknown``.
    unknown_when: str = ""

    @property
    def total(self) -> int:
        """How many approvals the hold needs in all."""
        return sum(count for _, count in self.groups)

    @property
    def labels(self) -> list[str]:
        """The groups' grant labels, in order."""
        return [label for label, _ in self.groups]

    def hold_fields(self) -> dict[str, Any]:
        """The registry hold's fields for this approval (``POST /v1/approvals``).

        ``approver_groups`` and ``only_named_groups``; with a ``due``, ``ttl_seconds`` --
        twice the due when the hold escalates, so the escalation gets its turn, the due
        otherwise -- and, with ``escalate_to``, ``escalate_to_label`` and
        ``escalate_after_seconds`` (the due). Never ``approver_labels``: on the registry
        an approver label makes its holders general approvers who count for ANY group, so
        two people of one group could satisfy "Finance and CFO".
        """
        fields: dict[str, Any] = {
            "approver_groups": [{"label": label, "count": count} for label, count in self.groups],
            "only_named_groups": self.only_named,
        }
        if self.escalate_to:
            fields["escalate_to_label"] = self.escalate_to
        if self.due_seconds is not None:
            if self.escalate_to:
                fields["ttl_seconds"] = 2 * self.due_seconds
                fields["escalate_after_seconds"] = self.due_seconds
            else:
                fields["ttl_seconds"] = self.due_seconds
        return fields


# ── reading a definition ─────────────────────────────────────────────────────


def _label(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _group_count(value: Any) -> int:
    """A group's count, tolerantly: absent or unreadable is 1, never fewer."""
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return max(value, 1)
    if isinstance(value, str) and value.strip().isdigit():
        return max(int(value.strip()), 1)
    return 1


def _groups(raw: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    """A rule's groups: ``all_of`` when it is a list, else ``by`` + ``count``.

    A group named twice is one group with both counts, so the labels stay distinct.
    """
    pairs: list[tuple[str, int]] = []
    all_of = raw.get("all_of")
    if isinstance(all_of, list):
        for item in all_of:
            if isinstance(item, Mapping) and (label := _label(item.get("by"))):
                pairs.append((label, _group_count(item.get("count"))))
    elif label := _label(raw.get("by")):
        pairs.append((label, _group_count(raw.get("count"))))
    merged: dict[str, int] = {}
    for label, count in pairs:
        merged[label] = merged.get(label, 0) + count
    return tuple(merged.items())


def _due(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        return parse_duration(value)
    except ValueError:
        return None


def parse_rule(raw: Any) -> ApprovalRule | None:
    """One rule of ``approval.rules``, or ``None`` when it names no group.

    A ``when`` that is not text, or does not parse, keeps the rule and reads as
    ``unknown`` (:attr:`ApprovalRule.when_error`); an unreadable ``due`` reads as none.
    """
    if not isinstance(raw, Mapping):
        return None
    groups = _groups(raw)
    if not groups:
        return None
    when = raw.get("when")
    text, error = "", ""
    if isinstance(when, str):
        text = when.strip()
        if text:
            try:
                parse_when(text)
            except WhenSyntaxError as exc:
                error = str(exc)
    elif when is not None:
        error = "the condition is not text"
    return ApprovalRule(
        groups=groups,
        when=text,
        due_seconds=_due(raw.get("due")),
        escalate_to=_label(raw.get("escalate_to")) or None,
        when_error=error,
    )


def parse_step_approval(value: Any) -> StepApproval | None:
    """Read a step's ``approval`` tolerantly: no group anywhere, no approval.

    The registry validates the shape strictly at publish; here a malformed value reads as
    absent rather than refusing the run. ``by`` alone is the old form; ``rules`` the new.
    """
    if not isinstance(value, Mapping):
        return None
    by = _label(value.get("by"))
    raw_rules = value.get("rules")
    rules = tuple(
        rule
        for raw in (raw_rules if isinstance(raw_rules, list) else [])
        if (rule := parse_rule(raw)) is not None
    )
    if not by and not rules:
        return None
    ask = value.get("ask")
    show = value.get("show")
    return StepApproval(
        by=by,
        ask=ask.strip() if isinstance(ask, str) else "",
        show=tuple(s.strip() for s in show if isinstance(s, str) and s.strip())
        if isinstance(show, list)
        else (),
        rules=rules,
        only_these_approvers=value.get("only_these_approvers") is True,
    )


# ── choosing a rule ──────────────────────────────────────────────────────────


def call_context(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """A call's arguments as a rule's ``when`` reads them (``args.<name>``).

    Values are read as the call carries them, except that a money value
    (``{"amount", "currency"}``, valid) compares as money, as a money input does.
    """
    context: dict[str, Any] = {}
    for name, value in (arguments or {}).items():
        if isinstance(value, Mapping) and validate_value(_MONEY, value) is None:
            context[str(name)] = Money(value["amount"], value["currency"])
        else:
            context[str(name)] = value
    return context


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _resolve(context: Mapping[str, Any], dotted: str) -> tuple[bool, Any]:
    current: Any = context
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _orders_an_argument_blindly(node: Node, context: Mapping[str, Any]) -> bool:
    """Whether ``node`` orders a call argument against a number when that argument is
    not a number (missing, text, a list): the comparison would be false, and a call that
    spells ``"20000"`` would get a laxer rule than one that sends ``20000``."""
    if isinstance(node, Not):
        return _orders_an_argument_blindly(node.operand, context)
    if isinstance(node, AllOf | AnyOf):
        return any(_orders_an_argument_blindly(part, context) for part in node.operands)
    if not isinstance(node, Compare) or node.op not in _ORDERING:
        return False
    for side, other in ((node.left, node.right), (node.right, node.left)):
        if not isinstance(side, Path) or not side.dotted.startswith("args."):
            continue
        if isinstance(other, Literal):
            numeric = _number(other.value)
        else:
            found, value = _resolve(context, other.dotted)
            numeric = found and (_number(value) or isinstance(value, Money))
        if not numeric:
            continue
        found, value = _resolve(context, side.dotted)
        if not found or not (_number(value) or isinstance(value, Money)):
            return True
    return False


def rule_verdict(rule: ApprovalRule, context: Mapping[str, Any]) -> str:
    """``"true"``, ``"false"`` or ``"unknown"``: whether ``rule`` applies in ``context``."""
    if rule.when_error:
        return UNKNOWN
    if not rule.when:
        return TRUE
    node = parse_when(rule.when)
    if _orders_an_argument_blindly(node, context):
        return UNKNOWN
    return when_verdict(node, context)


def _strictest(rules: Sequence[ApprovalRule], start: int = 0) -> int:
    """The position of the rule needing the most approvals from ``start`` on; first on a tie."""
    best = start
    for i in range(start + 1, len(rules)):
        if rules[i].total > rules[best].total:
            best = i
    return best


def pick_rule(
    approval: StepApproval, context: Mapping[str, Any], *, step: str = ""
) -> ResolvedApproval:
    """The rule of ``approval`` that applies in ``context``, and why (module docstring).

    ``context`` is what the rule's ``when`` reads: the step's typed context
    (:meth:`~tulip.playbooks.v2.engine.StepGraph.context`) with the call's arguments under
    ``args`` (:func:`call_context`). Never raises.
    """
    rules = approval.rules
    for i, rule in enumerate(rules):
        verdict = rule_verdict(rule, context)
        if verdict == FALSE:
            continue
        if verdict == TRUE:
            matched = MATCHED_DEFAULT if rule.default else MATCHED_WHEN
            return _resolved(approval, i, matched, context, step=step, earlier=i > 0)
        chosen = _strictest(rules, i)
        return _resolved(approval, chosen, MATCHED_UNKNOWN, context, step=step, unknown=rule)
    return _resolved(approval, _strictest(rules), MATCHED_NO_RULE, context, step=step)


def _resolved(
    approval: StepApproval,
    index: int,
    matched: str,
    context: Mapping[str, Any],
    *,
    step: str,
    earlier: bool = False,
    unknown: ApprovalRule | None = None,
) -> ResolvedApproval:
    rule = approval.rules[index]
    return ResolvedApproval(
        groups=rule.groups,
        due_seconds=rule.due_seconds,
        escalate_to=rule.escalate_to,
        only_named=approval.only_these_approvers,
        ask=approval.ask,
        show=approval.show,
        rule_index=index,
        reason=_reason(rule, matched, context, earlier=earlier, unknown=unknown),
        matched=matched,
        step=step,
        unknown_when=unknown.when if unknown is not None else "",
    )


# ── saying why, in plain words ───────────────────────────────────────────────


def who_must_approve(groups: Sequence[tuple[str, int]]) -> str:
    """``groups`` in words: "someone from Finance must approve", "Finance and CFO must
    both approve", "2 people from Finance and someone from CFO must approve"."""
    names = [group_name(label) for label, _ in groups]
    if all(count == 1 for _, count in groups) and len(groups) > 1:
        both = "both" if len(groups) == 2 else "all"
        return f"{_joined(names)} must {both} approve"
    parts = [
        f"someone from {name}" if count == 1 else f"{count} people from {name}"
        for name, (_, count) in zip(names, groups, strict=True)
    ]
    return f"{_joined(parts)} must approve"


def _joined(parts: Sequence[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")


def _reason(
    rule: ApprovalRule,
    matched: str,
    context: Mapping[str, Any],
    *,
    earlier: bool,
    unknown: ApprovalRule | None,
) -> str:
    who = who_must_approve(rule.groups)
    if matched == MATCHED_WHEN:
        return _sentence(f"{describe_when(parse_when(rule.when), context)}, so {who}")
    if matched == MATCHED_DEFAULT:
        return _sentence(f"no other rule applies, so {who}" if earlier else who)
    if matched == MATCHED_UNKNOWN and unknown is not None:
        if unknown.when_error:
            doubt = "this step's approval rule cannot be read"
        else:
            doubt = (
                f"this run cannot tell whether {describe_when(parse_when(unknown.when), context)}"
            )
        return _sentence(f"{doubt}, so the strictest rule applies: {who}")
    return _sentence(
        f"no rule applies and the step names no default, so the strictest rule applies: {who}"
    )


_OPS: Mapping[str, str] = {
    ">": "is more than",
    "<": "is less than",
    ">=": "is at least",
    "<=": "is at most",
    "==": "is",
    "!=": "is not",
}
_SYMBOLS: Mapping[str, str] = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}


def _path_words(dotted: str) -> str:
    parts = dotted.split(".")
    prefix = ""
    if parts[0] == "inputs" and len(parts) > 1:
        parts = parts[1:]
    elif parts[0] == "args" and len(parts) > 1:
        parts, prefix = parts[1:], "the call's "
    elif parts[0] == "outputs" and len(parts) > 2:
        parts = parts[2:]
    words = [re.sub(r"[_\-:]+", " ", p).strip() or p for p in parts]
    if prefix:
        return prefix + "'s ".join(words)
    return "the " + "'s ".join(words)


def _number_words(value: float, currency: str | None) -> str:
    text = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    if currency is None:
        return text
    symbol = _SYMBOLS.get(currency.upper())
    return f"{symbol}{text}" if symbol else f"{text} {currency.upper()}"


def _currency(context: Mapping[str, Any], operand: Any) -> str | None:
    if isinstance(operand, Path):
        found, value = _resolve(context, operand.dotted)
        if found and isinstance(value, Money):
            return value.currency
    return None


def _operand_words(operand: Any, context: Mapping[str, Any], currency: str | None = None) -> str:
    if isinstance(operand, Path):
        return _path_words(operand.dotted)
    value = operand.value
    if value is None:
        return "empty"
    if isinstance(value, bool):
        return "true" if value else "false"
    if _number(value):
        return _number_words(value, currency)
    return f'"{value}"'


def describe_when(node: Node, context: Mapping[str, Any]) -> str:
    """A condition in plain words: ``inputs.amount > 10000`` is "the amount is more than
    $10,000" when the amount is in dollars."""
    if isinstance(node, Always):
        return "it always applies"
    if isinstance(node, Truthy):
        return f"{_operand_words(node.operand, context)} is set"
    if isinstance(node, Compare):
        left_currency = _currency(context, node.left)
        right_currency = _currency(context, node.right)
        left = _operand_words(node.left, context, right_currency)
        right = _operand_words(node.right, context, left_currency)
        return f"{left} {_OPS.get(node.op, node.op)} {right}"
    if isinstance(node, Contains):
        return (
            f"{_operand_words(node.left, context)} includes {_operand_words(node.right, context)}"
        )
    if isinstance(node, Empty):
        return f"{_operand_words(node.operand, context)} is {'not ' if node.negate else ''}empty"
    if isinstance(node, Not):
        return f"it is not so that {describe_when(node.operand, context)}"
    joiner = " and " if isinstance(node, AllOf) else " or "
    parts = [
        f"({describe_when(part, context)})"
        if isinstance(part, AllOf | AnyOf)
        else describe_when(part, context)
        for part in node.operands
    ]
    return joiner.join(parts)


# ── enforcing it locally: the open-source approval stores ────────────────────


def authority_from_resolved(
    resolved: ResolvedApproval,
    *,
    roles_of: Callable[[str], Iterable[str]] | None = None,
    members: Mapping[str, Iterable[str]] | None = None,
    also: Iterable[str] = (),
    break_glass: Iterable[str] = (),
    delegations: Sequence[Delegation] = (),
    clock: Callable[[], datetime] | None = None,
) -> ApprovalAuthority:
    """An :class:`~tulip.control.approvals.ApprovalAuthority` that enforces ``resolved``.

    One rule per group, in order, each with the group's count as its quorum, so the call
    is approved only once every group has given its count (all of them), from distinct
    people: each approval counts toward one group only (``count_once``), the first of the
    approver's groups still short of its count. Whoever is in a group: the people whose
    ``roles_of`` include its grant label, or the names ``members`` lists for it.

    ``also`` are roles that may count for any group (the deployment's general approvers),
    unless the approval is ``only_named``; ``break_glass`` are roles that may count for
    any group even then (a tenant admin). The requester never approves its own call.

    Deadlines and escalation (``due_seconds``, ``escalate_to``) are the broker's, not the
    authority's. Hand the result to a store per held call::

        authorities: dict[str, ApprovalAuthority] = {}
        store = InMemoryApprovals(
            authority=lambda record: authorities.get(record.approval_id)
        )
        approval_id = store.submit(principal, tool, args)
        authorities[approval_id] = authority_from_resolved(
            runtime.approval_for(tool, args), roles_of=directory.roles_for
        )
    """
    from tulip.control.approvals import ApprovalAuthority, ApproverRule

    anyone = frozenset(break_glass) | (frozenset() if resolved.only_named else frozenset(also))
    rules = tuple(
        ApproverRule(
            roles=frozenset({label}) | anyone,
            approvers=frozenset((members or {}).get(label, ())),
            quorum=count,
            name=group_name(label),
        )
        for label, count in resolved.groups
    )
    authority = ApprovalAuthority(
        rules=rules, delegations=tuple(delegations), roles_of=roles_of, count_once=True
    )
    return authority if clock is None else replace(authority, clock=clock)


__all__ = [
    "MATCHED_DEFAULT",
    "MATCHED_NO_RULE",
    "MATCHED_UNKNOWN",
    "MATCHED_WHEN",
    "ApprovalRule",
    "ResolvedApproval",
    "StepApproval",
    "authority_from_resolved",
    "call_context",
    "describe_when",
    "group_name",
    "parse_duration",
    "parse_rule",
    "parse_step_approval",
    "pick_rule",
    "rule_verdict",
    "who_must_approve",
]
