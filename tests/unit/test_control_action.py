# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Deriving the Action a policy is weighed against.

A ControlPolicy matches on ``Action.labels()`` — environment, kind, tags — and
never on the action's name or asset. So a policy saying "hold anything in
production" only works if something puts that label on the action. These tests
pin that derivation, and the fail-safe when nothing is known.
"""

from __future__ import annotations

import pytest

from tulip.control import (
    Action,
    ControlPolicy,
    action_from_labels,
    approve,
    asset_from_args,
    default_action,
    resolve_action,
)


def test_declared_labels_reach_the_policy() -> None:
    action = action_from_labels(
        "issue_refund",
        {"order_id": "INV-4821"},
        labels={"environment": "production", "kind": "payment"},
    )
    assert action.labels() == {"production", "payment", "issue_refund"}
    assert action.asset == "INV-4821"


def test_a_production_action_is_held_rather_than_allowed() -> None:
    """The exact case that returned ALLOW on the live stack."""
    action = action_from_labels(
        "issue_refund", {"amount_usd": 1200}, labels={"environment": "production"}
    )
    decision = approve(action, policy=ControlPolicy(require_human_for=frozenset({"production"})))
    assert decision.outcome == "require_human"


def test_an_unlabelled_action_fails_safe_not_open() -> None:
    """`unknown`, never a plausible-looking default like `staging`."""
    action = default_action("mystery_tool", {})
    assert "unknown" in action.labels()
    assert "staging" not in action.labels()


def test_the_tool_name_is_always_a_tag() -> None:
    """Gating one specific tool by name keeps working whatever it declares."""
    action = action_from_labels("wipe_disk", {}, labels={"environment": "staging"})
    assert "wipe_disk" in action.labels()
    decision = approve(action, policy=ControlPolicy(deny_for=frozenset({"wipe_disk"})))
    assert decision.outcome == "deny"


def test_declared_tags_join_the_name() -> None:
    action = action_from_labels("deploy", {}, labels={"tags": ["irreversible", "infra"]})
    assert action.labels() >= {"deploy", "irreversible", "infra"}


def test_a_single_tag_string_is_accepted() -> None:
    assert "irreversible" in action_from_labels("x", {}, labels={"tags": "irreversible"}).labels()


def test_the_callers_environment_is_the_fallback() -> None:
    """An agent's or deployment's environment applies when a tool declares none."""
    action = action_from_labels("read_ledger", {}, environment="production")
    assert "production" in action.labels()


def test_a_declared_environment_beats_the_fallback() -> None:
    action = action_from_labels(
        "read_ledger", {}, labels={"environment": "sandbox"}, environment="production"
    )
    assert "sandbox" in action.labels()
    assert "production" not in action.labels()


@pytest.mark.parametrize("declared", [None, "", "   "])
def test_blank_declarations_fall_through_to_unknown(declared: str | None) -> None:
    action = action_from_labels("x", {}, labels={"environment": declared})
    assert "unknown" in action.labels()


def test_blast_radius_is_declared_or_defaulted() -> None:
    assert action_from_labels("x", {}, labels={"blast_radius": 50}).blast_radius == 50
    assert action_from_labels("x", {}, blast_radius=3).blast_radius == 3
    # A nonsense declaration must not crash the gate — fall back, don't raise.
    assert action_from_labels("x", {}, labels={"blast_radius": "lots"}).blast_radius == 1


def test_blast_radius_over_the_policy_maximum_holds() -> None:
    action = action_from_labels("bulk_refund", {}, labels={"blast_radius": 50})
    decision = approve(action, policy=ControlPolicy(max_blast_radius=1))
    assert decision.outcome == "require_human"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"asset": "a", "host": "b"}, "a"),  # priority order
        ({"host": "web-1"}, "web-1"),
        ({"order_id": "INV-1"}, "INV-1"),
        ({"unrecognised": "x"}, ""),
        ({}, ""),
        ({"asset": ""}, ""),  # empty values are skipped
    ],
)
def test_asset_from_args(kwargs: dict[str, str], expected: str) -> None:
    assert asset_from_args(kwargs) == expected


def test_resolve_action_passes_through_and_derives() -> None:
    explicit = Action(name="x", environment="production")
    assert resolve_action(explicit, "ignored", {}) is explicit
    assert resolve_action(None, "x", {}).environment == "unknown"
    assert resolve_action(lambda n, k: Action(name=n, kind="payment"), "x", {}).kind == "payment"
