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
    derive_labels,
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


# ---------------------------------------------------------------------------
# Argument-derived labels
#
# A tool's declared labels are static, so `refund(amount=10)` and
# `refund(amount=10_000)` used to produce byte-identical actions and no policy
# could separate them. `labels["derive"]` closes that: declarative rules,
# evaluated against the call's arguments. Data, never code.
# ---------------------------------------------------------------------------

HIGH_VALUE_LABELS = {
    "environment": "production",
    "kind": "payment",
    "derive": [{"when": {"arg": "amount", "gt": 500}, "add_tags": ["high_value"]}],
}


def test_the_amount_over_the_threshold_is_what_gets_held() -> None:
    """The PRFAQ promise: the policy weighs the *arguments*, not just the tool."""
    policy = ControlPolicy(
        require_verification_score=0.0, require_human_for=frozenset({"high_value"})
    )

    small = action_from_labels("refund_customer", {"amount": 10}, labels=HIGH_VALUE_LABELS)
    large = action_from_labels("refund_customer", {"amount": 10_000}, labels=HIGH_VALUE_LABELS)

    assert "high_value" not in small.labels()
    assert "high_value" in large.labels()
    assert approve(small, policy=policy).outcome != "require_human"
    assert approve(large, policy=policy).outcome == "require_human"


@pytest.mark.parametrize(
    ("rule", "kwargs", "fires"),
    [
        ({"arg": "n", "gt": 500}, {"n": 501}, True),
        ({"arg": "n", "gt": 500}, {"n": 500}, False),
        ({"arg": "n", "gte": 500}, {"n": 500}, True),
        ({"arg": "n", "gte": 500}, {"n": 499}, False),
        ({"arg": "n", "lt": 500}, {"n": 499}, True),
        ({"arg": "n", "lt": 500}, {"n": 500}, False),
        ({"arg": "n", "lte": 500}, {"n": 500}, True),
        ({"arg": "n", "lte": 500}, {"n": 501}, False),
        ({"arg": "n", "gt": 500}, {"n": 500.5}, True),  # floats compare too
        ({"arg": "r", "eq": "chargeback"}, {"r": "chargeback"}, True),
        ({"arg": "r", "eq": "chargeback"}, {"r": "duplicate"}, False),
        ({"arg": "r", "ne": "chargeback"}, {"r": "duplicate"}, True),
        ({"arg": "r", "ne": "chargeback"}, {"r": "chargeback"}, False),
        ({"arg": "r", "in": ["fraud", "chargeback"]}, {"r": "fraud"}, True),
        ({"arg": "r", "in": ["fraud", "chargeback"]}, {"r": "typo"}, False),
        ({"arg": "t", "contains": "prod"}, {"t": ["prod", "eu"]}, True),
        ({"arg": "t", "contains": "prod"}, {"t": ["dev"]}, False),
        ({"arg": "t", "contains": "prod"}, {"t": "acct-prod-1"}, True),  # substring
        ({"arg": "t", "contains": "prod"}, {"t": "acct-dev-1"}, False),
        ({"arg": "t", "contains": "prod"}, {"t": ("prod",)}, True),  # tuples count
    ],
)
def test_every_comparator(rule: dict[str, object], kwargs: dict[str, object], fires: bool) -> None:
    action = action_from_labels(
        "x", kwargs, labels={"derive": [{"when": rule, "add_tags": ["hit"]}]}
    )
    assert ("hit" in action.labels()) is fires
    # A determinable rule — fired or not — is never `undetermined`.
    assert "undetermined" not in action.labels()


def test_a_missing_argument_is_undetermined_not_safe() -> None:
    """Never silently read "we could not tell" as "below the threshold"."""
    action = action_from_labels("refund", {"currency": "usd"}, labels=HIGH_VALUE_LABELS)
    assert "high_value" not in action.labels()
    assert "undetermined" in action.labels()
    policy = ControlPolicy(require_human_for=frozenset({"undetermined"}))
    assert approve(action, policy=policy).outcome == "require_human"


@pytest.mark.parametrize("amount", ["500", None, True, [1, 2], {"a": 1}])
def test_a_wrongly_typed_argument_is_undetermined(amount: object) -> None:
    """No coercion games — a bool or a numeric string is not a threshold."""
    action = action_from_labels("refund", {"amount": amount}, labels=HIGH_VALUE_LABELS)
    assert "high_value" not in action.labels()
    assert "undetermined" in action.labels()


@pytest.mark.parametrize(
    "rule",
    [
        "not-a-rule",  # not a mapping
        {"when": {"arg": "amount", "gt": 500}},  # no effect
        {"add_tags": ["x"]},  # no `when`
        {"when": {"arg": "amount"}, "add_tags": ["x"]},  # no comparator
        {"when": {"arg": "amount", "gt": 1, "lt": 9}, "add_tags": ["x"]},  # two comparators
        {"when": {"arg": "amount", "wat": 1}, "add_tags": ["x"]},  # unknown comparator
        {"when": {"arg": "", "gt": 1}, "add_tags": ["x"]},  # empty arg name
        {"when": {"arg": 7, "gt": 1}, "add_tags": ["x"]},  # non-string arg name
        {"when": "amount > 500", "add_tags": ["x"]},  # not an expression language
        {"when": {"arg": "amount", "gt": "lots"}, "add_tags": ["x"]},  # non-numeric bound
        {"when": {"arg": "amount", "in": "fraud"}, "add_tags": ["x"]},  # `in` wants a list
        {"when": {"arg": "amount", "contains": 1}, "add_tags": ["x"]},  # int cannot contain
        {"when": {"arg": "amount", "gt": 1}, "add_tags": {"a": 1}},  # tags not a list
        {"when": {"arg": "amount", "gt": 1}, "add_tags": [7]},  # non-string tag
        {"when": {"arg": "amount", "gt": 1}, "add_tags": []},  # empty tag list
        {"when": {"arg": "amount", "gt": 1}, "set_kind": ""},  # blank kind
        {"when": {"arg": "amount", "gt": 1}, "set_environment": 3},  # non-string environment
        {"when": {"arg": "amount", "gt": 1}, "add_tags": ["x"], "run": "rm -rf /"},  # extra key
    ],
)
def test_a_malformed_rule_is_ignored_and_undetermined(rule: object) -> None:
    """A crash in the labeller must never become a bypass."""
    action = action_from_labels("refund", {"amount": 10_000}, labels={"derive": [rule]})
    assert action.labels() == {"refund", "unknown", "undetermined"}


@pytest.mark.parametrize("rules", ["nope", 7, {"when": {}}])
def test_a_derive_block_that_is_not_a_list_is_undetermined(rules: object) -> None:
    assert "undetermined" in action_from_labels("x", {}, labels={"derive": rules}).labels()


def test_a_derive_of_none_derives_nothing() -> None:
    assert action_from_labels("x", {}, labels={"derive": None}).labels() == {"x", "unknown"}


def test_multiple_rules_compose_in_order() -> None:
    action = action_from_labels(
        "refund_customer",
        {"amount": 900, "reason": "chargeback", "line_items": [1, 2, 3]},
        labels={
            "environment": "production",
            "kind": "payment",
            "tags": ["irreversible"],
            "derive": [
                {"when": {"arg": "amount", "gt": 500}, "add_tags": ["high_value"]},
                {"when": {"arg": "reason", "eq": "chargeback"}, "add_tags": ["chargeback"]},
                {"blast_radius": {"arg": "line_items", "agg": "count", "default": 1}},
            ],
        },
    )
    assert action.labels() == {
        "refund_customer",
        "production",
        "payment",
        "irreversible",
        "high_value",
        "chargeback",
    }
    assert action.blast_radius == 3


def test_effects_can_set_kind_and_environment() -> None:
    action = action_from_labels(
        "deploy",
        {"target": "prod-eu"},
        labels={
            "environment": "staging",
            "kind": "config",
            "derive": [
                {
                    "when": {"arg": "target", "contains": "prod"},
                    "set_environment": "production",
                    "set_kind": "infra",
                    "add_tags": ["irreversible"],
                }
            ],
        },
    )
    assert action.labels() == {"deploy", "production", "infra", "irreversible"}
    assert "staging" not in action.labels()


def test_a_single_derived_tag_string_is_accepted() -> None:
    action = action_from_labels(
        "x", {"n": 2}, labels={"derive": [{"when": {"arg": "n", "gt": 1}, "add_tags": "hot"}]}
    )
    assert "hot" in action.labels()


def test_blast_radius_counts_a_collection_argument() -> None:
    rules = [{"blast_radius": {"arg": "hosts", "agg": "count", "default": 1}}]
    assert (
        action_from_labels("patch", {"hosts": ["a", "b", "c"]}, labels={"derive": rules})
    ).blast_radius == 3
    assert (
        action_from_labels("patch", {"hosts": {"a": 1, "b": 2}}, labels={"derive": rules})
    ).blast_radius == 2


def test_blast_radius_reads_a_numeric_argument() -> None:
    rules = [{"blast_radius": {"arg": "count", "agg": "value"}}]
    assert action_from_labels("x", {"count": 42}, labels={"derive": rules}).blast_radius == 42
    assert action_from_labels("x", {"count": 4.7}, labels={"derive": rules}).blast_radius == 4
    assert action_from_labels("x", {"count": "9"}, labels={"derive": rules}).blast_radius == 9


def test_a_derived_radius_over_the_maximum_holds() -> None:
    action = action_from_labels(
        "bulk_refund",
        {"orders": list(range(50))},
        labels={"derive": [{"blast_radius": {"arg": "orders", "agg": "count"}}]},
    )
    assert approve(action, policy=ControlPolicy(max_blast_radius=10)).outcome == "require_human"


def test_deriving_never_lowers_the_blast_radius() -> None:
    """A derived radius may raise a declared one, never weaken it."""
    labels = {
        "blast_radius": 50,
        "derive": [{"blast_radius": {"arg": "hosts", "agg": "count"}}],
    }
    assert action_from_labels("x", {"hosts": ["a"]}, labels=labels).blast_radius == 50
    assert action_from_labels("x", {"hosts": list(range(80))}, labels=labels).blast_radius == 80
    # …nor the caller-supplied one.
    assert (
        action_from_labels(
            "x",
            {"hosts": ["a"]},
            labels={"derive": [{"blast_radius": {"arg": "hosts", "agg": "count"}}]},
            blast_radius=9,
        ).blast_radius
        == 9
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, 7),  # argument absent -> the declared default
        ({"hosts": "web-1"}, 7),  # a string is not a collection to count
        ({"hosts": 3}, 7),  # nor is a bare int
    ],
)
def test_an_uncountable_argument_falls_back_to_the_default_and_is_undetermined(
    kwargs: dict[str, object], expected: int
) -> None:
    action = action_from_labels(
        "patch",
        kwargs,
        labels={"derive": [{"blast_radius": {"arg": "hosts", "agg": "count", "default": 7}}]},
    )
    assert action.blast_radius == expected
    assert "undetermined" in action.labels()


def test_an_unusable_value_argument_is_undetermined() -> None:
    rules = [{"blast_radius": {"arg": "n", "agg": "value"}}]
    for value in ("many", True, [1], None):
        action = action_from_labels("x", {"n": value}, labels={"derive": rules})
        assert action.blast_radius == 1
        assert "undetermined" in action.labels()


@pytest.mark.parametrize(
    "spec",
    [
        "count",  # not a mapping
        {"agg": "count"},  # no arg
        {"arg": "hosts"},  # no agg
        {"arg": "hosts", "agg": "sum"},  # unsupported agg
        {"arg": "hosts", "agg": "count", "default": "1"},  # non-int default
        {"arg": "hosts", "agg": "count", "default": True},  # a bool is not a radius
        {"arg": "hosts", "agg": "count", "wat": 1},  # unknown key
        {"arg": "", "agg": "count"},  # empty arg name
    ],
)
def test_a_malformed_blast_radius_rule_is_ignored_and_undetermined(spec: object) -> None:
    action = action_from_labels(
        "patch", {"hosts": [1, 2, 3]}, labels={"derive": [{"blast_radius": spec}]}
    )
    assert action.blast_radius == 1
    assert "undetermined" in action.labels()


def test_a_blast_radius_rule_takes_an_unknown_key_as_malformed() -> None:
    action = action_from_labels(
        "patch",
        {"hosts": [1, 2, 3]},
        labels={"derive": [{"blast_radius": {"arg": "hosts", "agg": "count"}, "add_tags": ["x"]}]},
    )
    assert action.blast_radius == 1
    assert action.labels() == {"patch", "unknown", "undetermined"}


def test_a_blast_radius_rule_can_be_gated_by_when() -> None:
    rules = [
        {
            "when": {"arg": "mode", "eq": "bulk"},
            "blast_radius": {"arg": "orders", "agg": "count"},
        }
    ]
    bulk = action_from_labels(
        "refund", {"mode": "bulk", "orders": [1, 2, 3]}, labels={"derive": rules}
    )
    single = action_from_labels(
        "refund", {"mode": "one", "orders": [1, 2, 3]}, labels={"derive": rules}
    )
    assert bulk.blast_radius == 3
    assert single.blast_radius == 1
    assert "undetermined" not in single.labels()
    # …and an unevaluatable gate is undetermined rather than skipped quietly.
    assert "undetermined" in action_from_labels("refund", {}, labels={"derive": rules}).labels()


def test_no_derive_key_is_byte_identical_to_before() -> None:
    """The regression guard: this is additive, existing behaviour is untouched."""
    labels = {"environment": "production", "kind": "payment", "tags": ["irreversible"]}
    action = action_from_labels(
        "issue_refund", {"order_id": "INV-1", "amount": 10_000}, labels=labels
    )
    assert action == Action(
        name="issue_refund",
        asset="INV-1",
        blast_radius=1,
        environment="production",
        kind="payment",
        tags=frozenset({"issue_refund", "irreversible"}),
    )


def test_derive_labels_is_usable_on_its_own() -> None:
    """The gateway and registry consume the derivation directly, too."""
    derived = derive_labels(
        [
            {"when": {"arg": "amount", "gt": 500}, "add_tags": ["high_value"]},
            {"blast_radius": {"arg": "items", "agg": "count"}},
        ],
        {"amount": 900, "items": [1, 2]},
    )
    assert derived.tags == {"high_value"}
    assert derived.blast_radius == 2
    assert derived.undetermined is False
    assert derived.kind == ""
    assert derived.environment == ""
