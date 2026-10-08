# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit: the two result shapes the engine reads -- a refusal, and a call that ran nothing."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tulip.playbooks.v2.results import (
    DENIAL_PREFIX,
    not_executed_reason,
    not_executed_result,
    refused,
)


def test_a_refusal_is_the_gate_s_prefix_only() -> None:
    assert refused(f"{DENIAL_PREFIX} by policy")
    assert refused(f"  {DENIAL_PREFIX} by policy")
    assert not refused("denied")
    assert not refused(None)
    assert not refused({"text": DENIAL_PREFIX})


def test_the_not_executed_answer_round_trips() -> None:
    text = not_executed_result("refund", "label-only tool", {"amount": 3})
    assert json.loads(text) == {
        "executed": False,
        "tool": "refund",
        "decision": "allow",
        "reason": "label-only tool",
        "arguments": {"amount": 3},
    }
    assert not_executed_reason(text) == "label-only tool"
    assert not_executed_reason(not_executed_result("refund", "  ", {})) == "nothing ran"


@pytest.mark.parametrize(
    "result",
    [
        None,
        {"executed": False},
        "plain text",
        '{"ok": true}',
        '{"executed": false',  # not JSON
        '["executed"]',  # not an object
        '{"executed": true, "decision": "allow"}',
        '{"executed": false, "decision": "deny", "reason": "x"}',
    ],
)
def test_anything_else_is_a_call_that_ran(result: Any) -> None:
    assert not_executed_reason(result) is None
