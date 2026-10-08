# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity: the gateway evaluates `when` exactly as the registry parses it.

The vectors below are the registry's own (``tulip_registry`` tests/unit/test_when.py),
copied verbatim. A condition the registry accepted at publish must mean the same thing
when a run reaches the branch; if either side changes, both copies change.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.playbooks.v2.when import (
    MAX_WHEN_DEPTH,
    MAX_WHEN_LENGTH,
    WhenSyntaxError,
    evaluate_when,
    parse_when,
    when_paths,
)


CTX: dict[str, Any] = {
    "selected_branch_ids": ["tx_locking", "single_block_io"],
    "scope": "CLUSTER-WIDE",
    "error_count": 12,
    "ratio": 0.5,
    "triggered": True,
    "findings": [],
    "notes": "",
    "summary": "pool exhausted on db-2",
    "labels": {"tier": "gold"},
    "outputs": {"router": {"top": [{"event": "enq: TX"}], "scope": "LOCAL"}},
    "nothing": None,
}


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        # observai's authored shape: a bare word on the right of `contains` is a literal
        ("selected_branch_ids contains tx_locking", True),
        ("selected_branch_ids contains rac_cache_fusion", False),
        ("selected_branch_ids contains 'single_block_io'", True),
        ("summary contains 'exhausted'", True),
        ("labels contains tier", True),
        ("error_count contains 1", False),  # not a container
        # comparisons, as observai's evaluator has them
        ("scope == 'CLUSTER-WIDE'", True),
        ("scope != 'CLUSTER-WIDE'", False),
        ("scope equals CLUSTER-WIDE", True),
        ("error_count > 10", True),
        ("error_count >= 12 AND ratio < 1", True),
        ("error_count <= 11 OR ratio <= 0.5", True),
        ("error_count > 'ten'", False),  # unorderable: false, not an error
        ("nothing > 1", False),
        ("triggered > 0", False),  # booleans are not ordered
        ("scope == outputs.router.scope", False),  # a bare word on a symbolic side is a path
        ("outputs.router.top.0.event == 'enq: TX'", True),
        ("outputs.router.top.5.event is empty", True),
        ("outputs.router.scope.deeper is empty", True),
        ("triggered", True),
        ("NOT triggered", False),
        ("not (error_count > 100) and triggered", True),
        ("findings is empty", True),
        ("findings is not empty", False),
        ("notes is empty AND nothing is empty AND missing.path is empty", True),
        ("error_count is empty", False),
        ("always", True),
        ("ALWAYS", True),
        ("null == nothing", True),
        ("true", True),
        ("-1 < 0", True),
    ],
)
def test_conditions_evaluate(condition: str, expected: bool) -> None:
    assert evaluate_when(condition, CTX) is expected
    # Parsed once, evaluated many times: the tree gives the same answer.
    assert evaluate_when(parse_when(condition), CTX) is expected


@pytest.mark.parametrize(
    ("condition", "fragment"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("scope ==", "a value, found the end"),
        ("scope = 'x'", "unexpected '='"),
        ("(scope == 'x'", "expected ')'"),
        ("scope == 'x')", "AND, OR or the end"),
        ("scope is full", "'empty' after 'is'"),
        ("AND scope", "a path or a value"),
        ("__import__('os').system('x')", "unexpected"),
        ("1abc == 1", "unexpected"),
        ("x contains", "a value, found the end"),
    ],
)
def test_bad_conditions_are_refused_with_a_reason(condition: str, fragment: str) -> None:
    with pytest.raises(WhenSyntaxError, match=fragment.replace("(", r"\(").replace(")", r"\)")):
        parse_when(condition)
    assert evaluate_when(condition, CTX) is False


def test_limits_on_length_and_depth() -> None:
    with pytest.raises(WhenSyntaxError, match="longer than"):
        parse_when("a" * (MAX_WHEN_LENGTH + 1))
    with pytest.raises(WhenSyntaxError, match="nests deeper"):
        parse_when("(" * (MAX_WHEN_DEPTH + 1) + "x" + ")" * (MAX_WHEN_DEPTH + 1))
    with pytest.raises(WhenSyntaxError, match="nests deeper"):
        parse_when("NOT " * (MAX_WHEN_DEPTH + 1) + "x")
    with pytest.raises(WhenSyntaxError, match="is text"):
        parse_when(42)  # type: ignore[arg-type]


def test_paths_a_condition_reads() -> None:
    node = parse_when(
        "selected_branch_ids contains tx AND (scope == other.scope OR NOT findings is empty)"
        " AND scope != 'x'"
    )
    assert when_paths(node) == ["selected_branch_ids", "scope", "other.scope", "findings"]
    assert when_paths(parse_when("always")) == []
