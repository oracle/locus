# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""What a finished tool call's result says about whether it ran.

The playbook engine credits a step only with calls that executed, so it has to read
two answers that complete normally but did nothing: the gate's refusal and an allowed
call with no body. Their shapes are a contract with whoever runs the gate (the gateway
writes both), copied here byte for byte so the engine reads them the same wherever the
run's loop is.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Mapping


#: How the gate's refusal of a call starts, as the model sees it.
DENIAL_PREFIX = "⛔ denied:"


def refused(result: Any) -> bool:
    """Whether a completed call's result is the gate refusing it."""
    return isinstance(result, str) and result.lstrip().startswith(DENIAL_PREFIX)


def not_executed_result(name: str, reason: str, arguments: Mapping[str, Any]) -> str:
    """The model-visible answer to an ALLOWED call that has no body to run.

    Structured, and explicit that nothing happened: the old ``[name] executed``
    echo was read by models as success, and a model narrated a refund the
    payment system never received. The arguments are returned so the model can
    see exactly which call this answer is about.
    """
    return json.dumps(
        {
            "executed": False,
            "tool": name,
            "decision": "allow",
            "reason": reason,
            "arguments": dict(arguments),
        },
        default=str,
    )


def not_executed_reason(result: Any) -> str | None:
    """Why a completed call ran nothing, read off its result, or ``None`` if it ran.

    The reader of :func:`not_executed_result`. An allowed call with no body
    completes normally, with no error, so every consumer that asks "did this
    call happen?" has to recognise that answer, and they all recognise it here.
    Only that exact shape counts: ``executed: false`` on an ``allow``. Anything
    else, including a real tool's own JSON, is a call that ran.
    """
    if not isinstance(result, str):
        return None
    text = result.strip()
    if not text.startswith("{") or '"executed"' not in text:
        return None
    try:
        body = json.loads(text)
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("executed") is not False:
        return None
    if body.get("decision") != "allow":
        return None
    return str(body.get("reason") or "").strip() or "nothing ran"


__all__ = ["DENIAL_PREFIX", "not_executed_reason", "not_executed_result", "refused"]
