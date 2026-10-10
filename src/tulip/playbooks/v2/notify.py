# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A playbook's ``notify`` rules: who is told when a step starts, waits, ends or fails.

::

    notify:
      - on: [step.waiting, step.failed]   # step.started | step.done | process.done | process.failed
        steps: [confirm_account]          # omit = every step
        include: [order_id]               # values a notice may carry
        to: [{by: finance}, requester, {channel: slack:finance-alerts}]

The registry sends the notices (its notice dispatcher reads the run's pinned playbook);
the engine sends nothing. This module only reads the rules, the way the registry does
(``tulip_registry.notices.rules_of``), so a local run, a test or a tool can see them:
:attr:`~tulip.playbooks.v2.engine.PlaybookV2.notify_rules`.

Reading is tolerant: the top-level ``notify`` first, else ``metadata.notify``; a rule that
is not a mapping, names no known event or nobody to tell is left out.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


#: Every event a ``notify`` rule may name.
NOTIFY_EVENTS: tuple[str, ...] = (
    "step.started",
    "step.waiting",
    "step.done",
    "step.failed",
    "process.done",
    "process.failed",
)

TargetKind = Literal["by", "requester", "slack"]


@dataclass(frozen=True)
class NotifyTarget:
    """One ``to`` entry: the approvers of a label (``by``), the ``requester``, or a Slack
    channel of the workspace (``slack``, by name)."""

    kind: TargetKind
    name: str = ""


@dataclass(frozen=True)
class NotifyRule:
    """One ``notify`` rule. ``steps`` is ``None`` for every step."""

    on: tuple[str, ...]
    to: tuple[NotifyTarget, ...]
    steps: tuple[str, ...] | None = None
    include: tuple[str, ...] = ()

    def applies(self, event: str, step_id: str | None = None) -> bool:
        """Whether this rule asks to be told of ``event`` (on ``step_id``)."""
        if event not in self.on:
            return False
        return not event.startswith("step.") or self.steps is None or step_id in self.steps


def _names(raw: Any) -> tuple[str, ...]:
    items = [raw] if isinstance(raw, str) else raw if isinstance(raw, list | tuple) else []
    return tuple(i.strip() for i in items if isinstance(i, str) and i.strip())


def _target(raw: Any) -> NotifyTarget | None:
    if raw == "requester" or (isinstance(raw, Mapping) and raw.get("requester") is True):
        return NotifyTarget("requester")
    if not isinstance(raw, Mapping):
        return None
    by = raw.get("by")
    if isinstance(by, str) and by.strip():
        return NotifyTarget("by", by.strip())
    channel = raw.get("channel")
    if isinstance(channel, str) and channel.startswith("slack:") and channel[6:].strip():
        return NotifyTarget("slack", channel[6:].strip())
    return None


def parse_notify_rules(definition: Mapping[str, Any]) -> tuple[NotifyRule, ...]:
    """A playbook's ``notify`` rules: the top-level field, else ``metadata.notify``."""
    raw = definition.get("notify")
    if raw is None:
        metadata = definition.get("metadata")
        raw = metadata.get("notify") if isinstance(metadata, Mapping) else None
    rules: list[NotifyRule] = []
    for entry in raw if isinstance(raw, list | tuple) else []:
        if not isinstance(entry, Mapping):
            continue
        on = tuple(e for e in _names(entry.get("on")) if e in NOTIFY_EVENTS)
        targets = entry.get("to")
        to = tuple(
            t
            for t in (_target(r) for r in (targets if isinstance(targets, list | tuple) else []))
            if t is not None
        )
        if not on or not to:
            continue
        steps = _names(entry.get("steps")) if entry.get("steps") is not None else None
        rules.append(NotifyRule(on=on, to=to, steps=steps, include=_names(entry.get("include"))))
    return tuple(rules)


__all__ = ["NOTIFY_EVENTS", "NotifyRule", "NotifyTarget", "TargetKind", "parse_notify_rules"]
