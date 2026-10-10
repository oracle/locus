# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit: typed process data -- validated outputs, inputs in conditions, sensitive fields.

A v2 playbook declares its data as typed fields (the registry's ``PlaybookField``): its
inputs, each step's ``outputs`` beside the old ``expected_outputs``, and the answer to a
declared question. ``complete_step`` checks every typed value with the registry's own
sentences; a ``when`` reads ``inputs.<name>`` and compares money by amount (never across
currencies), dates as dates and numbers as numbers; a value the run cannot use is
UNAVAILABLE, so the step is not routed on a guess. ``public_outputs`` / ``public_inputs``
hand the gateway exactly the ``sensitive: false`` values it may mirror in clear.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tulip.playbooks.v2 import (
    ROUTING_UNKNOWN,
    UNAVAILABLE,
    UNKNOWN,
    Field,
    evaluate_when,
    validate_inputs,
    validate_value,
    when_verdict,
)
from tulip.playbooks.v2.engine import (
    ACTIVE,
    DONE,
    WAIVED,
    PlaybookRuntime,
    parse_playbook_v2,
    playbook_prose,
    step_brief,
)
from tulip.playbooks.v2.fields import (
    describe,
    is_iso_date,
    label_from_name,
    parse_field,
    parse_fields,
    text_field,
    typed_value,
)
from tulip.playbooks.v2.when import Day, Money


def _invoice(branches: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """An invoice playbook: ``check`` routes to ``cfo`` or ``clerk``."""
    return {
        "version": "playbook.v2",
        "id": "invoice-approval",
        "title": "Invoice approval",
        "inputs": [
            {"name": "amount", "type": "money", "label": "Invoice amount", "sensitive": False},
            {"name": "vendor_name", "sensitive": False},
            {"name": "due", "type": "date", "required": False},
            {"name": "order_id", "description": "The order.", "required": True},
        ],
        "step_groups": [
            {
                "id": "g",
                "title": "Review",
                "steps": [
                    {
                        "id": "check",
                        "title": "Check the invoice",
                        "allowed_tools": [],
                        "expected_outputs": ["note"],
                        "outputs": [
                            {"name": "total", "type": "money", "sensitive": False},
                            {"name": "paid_on", "type": "date", "required": False},
                            {
                                "name": "vendor_email",
                                "type": "email",
                                "required": False,
                                "sensitive": False,
                            },
                        ],
                        "required_from_user": [
                            {
                                "name": "approved_on",
                                "question": "When did the buyer approve it?",
                                "field": {"type": "date", "sensitive": False},
                            },
                        ],
                        "branches": branches
                        if branches is not None
                        else [
                            {"id": "big", "when": "inputs.amount > 10000", "next_step_id": "cfo"},
                            {
                                "id": "small",
                                "when": "NOT (inputs.amount > 10000)",
                                "next_step_id": "clerk",
                            },
                        ],
                    },
                    {"id": "cfo", "title": "CFO review", "allowed_tools": [], "after": ["check"]},
                    {
                        "id": "clerk",
                        "title": "Clerk review",
                        "allowed_tools": [],
                        "after": ["check"],
                    },
                ],
            }
        ],
    }


def _money(amount: float, currency: str = "USD") -> dict[str, Any]:
    return {"amount": amount, "currency": currency}


def _outputs(**extra: Any) -> dict[str, Any]:
    """Valid outputs of ``check``."""
    return {"note": "looks fine", "total": _money(500), "approved_on": "2026-10-01", **extra}


def _runtime(
    definition: dict[str, Any] | None = None, events: list[dict[str, Any]] | None = None
) -> PlaybookRuntime:
    playbook = parse_playbook_v2(definition or _invoice())
    sink: list[dict[str, Any]] = [] if events is None else events
    return PlaybookRuntime(playbook, emit=sink.append)


_INPUTS: dict[str, Any] = {
    "amount": _money(12000),
    "vendor_name": "Acme",
    "order_id": "o-1",
}


# ── reading the definition ───────────────────────────────────────────────────


def test_an_old_input_reads_as_a_sensitive_text_field() -> None:
    playbook = parse_playbook_v2(_invoice())
    order = next(f for f in playbook.input_fields if f.name == "order_id")
    assert order == Field(
        name="order_id", label="Order id", type="text", description="The order.", sensitive=True
    )
    # ``inputs`` keeps its old shape: (name, description) pairs.
    assert ("order_id", "The order.") in playbook.inputs
    assert [f.name for f in playbook.input_fields] == ["amount", "vendor_name", "due", "order_id"]
    amount = playbook.input_fields[0]
    assert (amount.type, amount.label, amount.sensitive, amount.required) == (
        "money",
        "Invoice amount",
        False,
        True,
    )


def test_expected_outputs_merge_with_typed_outputs() -> None:
    step = parse_playbook_v2(_invoice()).step("check")
    assert step is not None
    assert [(f.name, f.type) for f in step.output_fields()] == [
        ("note", "text"),
        ("total", "money"),
        ("paid_on", "date"),
        ("vendor_email", "email"),
    ]
    assert step.output_fields()[0] == text_field("note")
    # The answer takes the question's name and its declared type.
    assert step.answer_fields == (
        Field(name="approved_on", label="Approved on", type="date", sensitive=False),
    )
    assert step.data_fields()["approved_on"].type == "date"
    assert [f.name for f in step.typed_fields()] == [
        "total",
        "paid_on",
        "vendor_email",
        "approved_on",
    ]


def test_a_name_in_both_lists_is_one_typed_output() -> None:
    body = _invoice()
    check = body["step_groups"][0]["steps"][0]
    check["expected_outputs"] = ["total", "note", "total"]
    check["required_from_user"] = [{"name": "total", "field": {"type": "money"}}]
    step = parse_playbook_v2(body).step("check")
    assert step is not None
    assert [f.name for f in step.output_fields()] == [
        "total",
        "note",
        "paid_on",
        "vendor_email",
    ]
    assert step.output_fields()[0].type == "money"
    assert [f.name for f in step.typed_fields()] == ["total", "paid_on", "vendor_email"]


def test_a_question_without_a_field_has_a_sensitive_text_answer() -> None:
    body = _invoice()
    check = body["step_groups"][0]["steps"][0]
    check["required_from_user"] = [
        {"name": "why", "question": "Why?"},
        {"name": "who", "field": "not a mapping"},
        {"name": "when", "field": {"name": "other_name", "type": "date"}},
    ]
    step = parse_playbook_v2(body).step("check")
    assert step is not None
    assert step.required_from_user == (("why", "Why?"), ("who", ""), ("when", ""))
    # A field written under another name still takes the question's.
    assert [f.name for f in step.answer_fields] == ["when"]
    assert step.data_fields()["why"] == text_field("why")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"name": "x", "type": "colour"}, Field(name="x", label="X")),
        ({"name": "x", "type": "choice"}, Field(name="x", label="X")),
        ({"name": "x", "type": "choice", "choices": ["", 3]}, Field(name="x", label="X")),
        (
            {"name": "x", "type": "choice", "choices": ["a", "b"]},
            Field(name="x", label="X", type="choice", choices=("a", "b")),
        ),
        # Only an explicit false switches either flag off.
        ({"name": "x", "sensitive": "false", "required": 0}, Field(name="x", label="X")),
        ({"name": "x", "label": "  Shown  "}, Field(name="x", label="Shown")),
    ],
)
def test_a_field_reads_tolerantly(raw: dict[str, Any], expected: Field) -> None:
    assert parse_field(raw) == expected


@pytest.mark.parametrize("raw", [None, "amount", 3, {"name": "  "}, {"type": "money"}])
def test_a_field_that_names_nothing_is_skipped(raw: Any) -> None:
    assert parse_field(raw) is None


def test_fields_read_once_each() -> None:
    assert parse_fields("not a list") == ()
    fields = parse_fields([{"name": "a"}, {"name": "a", "type": "number"}, None, {"name": "b"}])
    assert [(f.name, f.type) for f in fields] == [("a", "text"), ("b", "text")]


def test_labels_are_made_from_names() -> None:
    assert label_from_name("vendor_id") == "Vendor id"
    assert label_from_name("due-date:local") == "Due date local"
    assert label_from_name("_") == "_"
    assert Field(name="vendor_id").shown_as == "Vendor id"


# ── each type, validated with the registry's sentences ───────────────────────


def _f(kind: str, **extra: Any) -> Field:
    return Field(name="x", label="X", type=kind, **extra)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (_f("text"), "hello"),
        (_f("number"), 3),
        (_f("number"), 2.5),
        (_f("money"), {"amount": 10, "currency": "EUR"}),
        (_f("date"), "2026-02-28"),
        (_f("boolean"), False),
        (_f("choice", choices=("a", "b")), "b"),
        (_f("email"), "ap@example.com"),
        (_f("url"), "https://example.com/x"),
        (_f("file"), {"ref": "files/1", "name": "invoice.pdf"}),
        (
            _f("file"),
            {"ref": "files/1", "name": "i.pdf", "size": 10, "media_type": "application/pdf"},
        ),
        (_f("text", required=False), None),
    ],
)
def test_a_valid_value_passes(field: Field, value: Any) -> None:
    assert validate_value(field, value) is None


@pytest.mark.parametrize(
    ("field", "value", "sentence"),
    [
        (_f("text"), None, "X is required."),
        (_f("text"), 3, "X must be text."),
        (_f("email"), ["a"], "X must be text."),
        (_f("choice", choices=("a", "b")), "c", "X must be one of: a, b."),
        (_f("email"), "not-an-email", "X must be an email address."),
        (
            _f("url"),
            "ftp://example.com",
            "X must be a web address starting with http:// or https://.",
        ),
        (_f("url"), "http://[::1", "X must be a web address starting with http:// or https://."),
        (_f("number"), True, "X must be a number."),
        (_f("number"), float("nan"), "X must be a number."),
        (_f("number"), "12", "X must be a number."),
        (
            _f("money"),
            12000,
            "X must be an amount of money: an amount and a three-letter currency code.",
        ),
        (
            _f("money"),
            {"amount": 1, "currency": "USD", "note": "x"},
            "X must be an amount of money: an amount and a three-letter currency code.",
        ),
        (_f("money"), {"amount": "1", "currency": "USD"}, "X must have a number as its amount."),
        (
            _f("money"),
            {"amount": 1, "currency": "usd"},
            "X must have an ISO 4217 currency code, such as USD or EUR.",
        ),
        (_f("date"), "2026-02-30", "X must be a date written as YYYY-MM-DD."),
        (_f("date"), "28/02/2026", "X must be a date written as YYYY-MM-DD."),
        (_f("boolean"), "yes", "X must be yes or no (true or false)."),
        (
            _f("file"),
            b"%PDF",
            "X must be a reference to a stored file, never the file itself.",
        ),
        (_f("file"), "files/1", "X must be a reference to a stored file, with its ref and name."),
        (
            _f("file"),
            {"ref": "r", "name": "n", "bytes": "AAAA"},
            "X must be a reference to a stored file (ref, name, size, media_type); "
            "it cannot carry bytes.",
        ),
        (_f("file"), {"ref": " ", "name": "n"}, "X must name the stored file's ref."),
        (_f("file"), {"ref": "r"}, "X must name the stored file's name."),
        (
            _f("file"),
            {"ref": "r", "name": "n", "size": -1},
            "X must give its size as a whole number of bytes.",
        ),
        (
            _f("file"),
            {"ref": "r", "name": "n", "media_type": "pdf"},
            "X must give its media type as type/subtype, such as application/pdf.",
        ),
    ],
)
def test_an_invalid_value_is_refused_in_plain_words(
    field: Field, value: Any, sentence: str
) -> None:
    assert validate_value(field, value) == sentence


def test_iso_dates() -> None:
    assert is_iso_date("2026-10-10")
    assert not is_iso_date("2026-13-01")
    assert not is_iso_date(20261010)


def test_typed_values_are_what_conditions_compare() -> None:
    money = typed_value(_f("money"), {"amount": 5, "currency": "USD"})
    assert isinstance(money, Money)
    assert (money.amount, money.currency, dict(money), len(money)) == (5, "USD", _money(5), 2)
    assert money == _money(5)
    assert repr(money) == "Money(5, 'USD')"
    with pytest.raises(KeyError):
        money["rate"]
    assert isinstance(typed_value(_f("date"), "2026-01-02"), Day)
    # Invalid, missing and untyped values are left as they are.
    assert typed_value(_f("money"), 5) == 5
    assert typed_value(_f("money", required=False), None) is None
    assert typed_value(_f("number"), 7) == 7


@pytest.mark.parametrize(
    ("kind", "extra", "text"),
    [
        ("text", {}, ""),
        ("money", {}, 'money: {"amount": <number>, "currency": "<ISO 4217 code>"}'),
        ("date", {}, "date: YYYY-MM-DD"),
        ("boolean", {}, "true or false"),
        ("choice", {"choices": ("a", "b")}, "one of: a, b"),
        ("file", {}, 'file: {"ref", "name"} of a stored file, never its bytes'),
        ("number", {}, "number"),
        ("email", {}, "email"),
    ],
)
def test_each_type_is_described_to_the_model(kind: str, extra: dict[str, Any], text: str) -> None:
    assert describe(_f(kind, **extra)) == text


# ── typed comparisons in the when language ───────────────────────────────────


def _ctx(**values: Any) -> dict[str, Any]:
    return {"inputs": values}


@pytest.mark.parametrize(
    ("condition", "context", "verdict"),
    [
        ("inputs.amount > 10000", _ctx(amount=Money(12000, "USD")), "true"),
        ("inputs.amount > 10000", _ctx(amount=Money(9000, "USD")), "false"),
        ("10000 < inputs.amount", _ctx(amount=Money(12000, "USD")), "true"),
        ("inputs.amount == 12000", _ctx(amount=Money(12000, "USD")), "true"),
        ("inputs.amount.currency == 'USD'", _ctx(amount=Money(1, "USD")), "true"),
        ("inputs.amount.amount >= 1", _ctx(amount=Money(1, "USD")), "true"),
        ("inputs.a > inputs.b", _ctx(a=Money(2, "EUR"), b=Money(1, "EUR")), "true"),
        ("inputs.a <= inputs.b", _ctx(a=Money(2, "EUR"), b=Money(1, "EUR")), "false"),
        # Two currencies cannot be ordered; they are never equal.
        ("inputs.a > inputs.b", _ctx(a=Money(2, "EUR"), b=Money(1, "USD")), "unknown"),
        ("inputs.a < inputs.b", _ctx(a=Money(2, "EUR"), b=Money(1, "USD")), "unknown"),
        ("inputs.a == inputs.b", _ctx(a=Money(1, "EUR"), b=Money(1, "USD")), "false"),
        ("inputs.a != inputs.b", _ctx(a=Money(1, "EUR"), b=Money(1, "USD")), "true"),
        # Unknown wherever it is, whatever the rest says.
        ("always OR inputs.a > inputs.b", _ctx(a=Money(2, "EUR"), b=Money(1, "USD")), "unknown"),
        ("NOT (inputs.a > inputs.b)", _ctx(a=Money(2, "EUR"), b=Money(1, "USD")), "unknown"),
        (
            "inputs.a > 1 AND inputs.a > inputs.b",
            _ctx(a=Money(2, "EUR"), b=Money(1, "USD")),
            "unknown",
        ),
        # Money against text is never ordered.
        ("inputs.amount > 'big'", _ctx(amount=Money(1, "USD")), "false"),
        # Dates are chronological, and order only against dates.
        ("inputs.due < '2026-11-01'", _ctx(due=Day("2026-10-31")), "true"),
        ("inputs.due > inputs.paid", _ctx(due=Day("2026-10-31"), paid=Day("2026-02-01")), "true"),
        ("inputs.due == '2026-10-31'", _ctx(due=Day("2026-10-31")), "true"),
        ("inputs.due != 'soon'", _ctx(due=Day("2026-10-31")), "true"),
        ("inputs.due > 'soon'", _ctx(due=Day("2026-10-31")), "false"),
        ("inputs.due > 20261001", _ctx(due=Day("2026-10-31")), "false"),
        ("inputs.due > '2026-02-30'", _ctx(due=Day("2026-10-31")), "false"),
        # Numbers stay numbers.
        ("inputs.n > 9.5", _ctx(n=10), "true"),
        # Without typed values the answers are the untyped ones: a plain dict is not ordered.
        ("inputs.amount > 10000", _ctx(amount=_money(12000)), "false"),
    ],
)
def test_typed_values_compare_as_their_type_says(
    condition: str, context: dict[str, Any], verdict: str
) -> None:
    assert when_verdict(condition, context) == verdict


def test_an_undecidable_condition_does_not_hold() -> None:
    context = _ctx(a=Money(2, "EUR"), b=Money(1, "USD"))
    assert when_verdict("inputs.a > inputs.b", context) == UNKNOWN
    assert evaluate_when("inputs.a > inputs.b", context) is False


# ── complete_step checks every typed value ───────────────────────────────────


def test_complete_step_refuses_every_bad_field_at_once() -> None:
    rt = _runtime()
    rt.start(inputs=_INPUTS)
    result = rt.complete_step(
        "check",
        _outputs(
            total=_money(500, "dollars"),
            paid_on="last week",
            vendor_email="nope",
            approved_on="2026-10-01",
        ),
    )
    assert result["ok"] is False
    assert result["invalid"] == {
        "total": "Total must have an ISO 4217 currency code, such as USD or EUR.",
        "paid_on": "Paid on must be a date written as YYYY-MM-DD.",
        "vendor_email": "Vendor email must be an email address.",
    }
    assert result["error"].startswith("step check cannot close: Total must have")
    assert "Fix each of these outputs" in result["error"]
    # Nothing moved.
    assert rt.graph.status["check"] == ACTIVE
    assert "check" not in rt.graph.outputs


def test_a_declared_answer_is_checked_against_its_field() -> None:
    rt = _runtime()
    rt.start(inputs=_INPUTS)
    result = rt.complete_step("check", _outputs(approved_on="yesterday"))
    assert result["invalid"] == {"approved_on": "Approved on must be a date written as YYYY-MM-DD."}


def test_a_required_typed_output_must_be_present() -> None:
    rt = _runtime()
    rt.start(inputs=_INPUTS)
    result = rt.complete_step("check", {"note": "n", "approved_on": "2026-10-01"})
    assert result["ok"] is False
    assert result["missing"] == ["total"]
    # Present but null is refused by its field.
    again = rt.complete_step("check", _outputs(total=None))
    assert again["invalid"] == {"total": "Total is required."}


def test_optional_outputs_may_be_absent_or_null_and_untyped_ones_take_anything() -> None:
    rt = _runtime()
    rt.start(inputs=_INPUTS)
    result = rt.complete_step(
        "check", _outputs(note={"any": ["shape"]}, paid_on=None, extra=object())
    )
    assert result["ok"] is True, result
    assert result["branches_taken"] == ["big"]


# ── inputs in conditions ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("amount", "taken", "active"), [(12000, "big", "cfo"), (900, "small", "clerk")]
)
def test_a_branch_on_a_money_input_routes(amount: float, taken: str, active: str) -> None:
    rt = _runtime()
    problems = rt.set_inputs({**_INPUTS, "amount": _money(amount)})
    assert problems == []
    rt.start()
    result = rt.complete_step("check", _outputs())
    assert result["ok"] is True
    assert result["branches_taken"] == [taken]
    assert result["active"] == [active]
    assert rt.graph.status["cfo" if active == "clerk" else "clerk"] == WAIVED


def test_inputs_given_at_start() -> None:
    rt = _runtime()
    rt.start(inputs={**_INPUTS, "amount": _money(20000)})
    assert rt.complete_step("check", _outputs())["active"] == ["cfo"]


def test_money_in_two_currencies_does_not_route() -> None:
    branches = [
        {"id": "over", "when": "total > inputs.amount", "next_step_id": "cfo"},
        {"id": "within", "when": "NOT (total > inputs.amount)", "next_step_id": "clerk"},
    ]
    rt = _runtime(_invoice(branches))
    rt.start(inputs={**_INPUTS, "amount": _money(100, "EUR")})
    result = rt.complete_step("check", _outputs(total=_money(500, "USD")))
    assert result["ok"] is False
    assert result["routing"] == ROUTING_UNKNOWN
    assert result["branches_unknown"] == ["over", "within"]
    assert result["unavailable"] == []
    assert result["unavailable_inputs"] == []
    assert "different currencies" in result["error"]
    assert "a person must choose" in result["error"].lower()
    assert rt.graph.status["check"] == ACTIVE
    # Same currency: it routes.
    assert rt.complete_step("check", _outputs(total=_money(50, "EUR")))["active"] == ["clerk"]


def test_a_date_output_routes_chronologically() -> None:
    branches = [
        {"id": "late", "when": "paid_on > inputs.due", "next_step_id": "cfo"},
        {"id": "on_time", "when": "NOT (paid_on > inputs.due)", "next_step_id": "clerk"},
    ]
    rt = _runtime(_invoice(branches))
    rt.start(inputs={**_INPUTS, "due": "2026-09-30"})
    assert rt.complete_step("check", _outputs(paid_on="2026-10-02"))["active"] == ["cfo"]


def test_an_output_of_an_earlier_step_is_typed_too() -> None:
    body = _invoice()
    steps = body["step_groups"][0]["steps"]
    steps[0]["branches"] = []
    steps[1]["branches"] = [
        {"id": "big", "when": "outputs.check.total >= 1000", "next_step_id": "clerk"}
    ]
    steps[2]["after"] = ["cfo"]
    rt = _runtime(body)
    rt.start(inputs=_INPUTS)
    assert rt.complete_step("check", _outputs(total=_money(1000)))["active"] == ["cfo"]
    assert rt.complete_step("cfo", {})["active"] == ["clerk"]


def test_a_required_input_not_given_is_unknown_not_null() -> None:
    rt = _runtime()
    rt.start()
    assert rt.unavailable_inputs() == ["amount", "vendor_name", "order_id"]
    result = rt.complete_step("check", _outputs())
    assert result["routing"] == ROUTING_UNKNOWN
    assert result["unavailable_inputs"] == ["amount"]
    assert "reads inputs this run does not have (amount)" in result["error"]
    # Once given, the same step routes.
    rt.set_inputs(_INPUTS)
    assert rt.complete_step("check", _outputs())["active"] == ["cfo"]


def test_an_invalid_input_is_reported_and_unknown() -> None:
    rt = _runtime()
    problems = rt.set_inputs({**_INPUTS, "amount": 12000, "colour": "red"})
    assert problems == [
        "The playbook has no input named 'colour'.",
        "Invoice amount must be an amount of money: an amount and a three-letter currency code.",
    ]
    assert "colour" not in rt.graph.inputs
    assert rt.graph.inputs["amount"] is UNAVAILABLE
    rt.start()
    assert rt.complete_step("check", _outputs())["routing"] == ROUTING_UNKNOWN


def test_an_optional_input_not_given_reads_null() -> None:
    rt = _runtime()
    rt.set_inputs(_INPUTS)
    assert "due" not in rt.graph.inputs
    assert rt.unavailable_inputs() == []


def test_a_withheld_input_is_unknown_and_not_checked() -> None:
    digest = {"redacted": True, "sha256": "ab" * 32, "bytes": 30}
    rt = _runtime()
    assert rt.set_inputs({**_INPUTS, "amount": digest}) == []
    assert rt.unavailable_inputs() == ["amount"]
    rt.start()
    assert rt.complete_step("check", _outputs())["routing"] == ROUTING_UNKNOWN


def test_validate_inputs_mirrors_the_registry() -> None:
    playbook = parse_playbook_v2(_invoice())
    assert validate_inputs(playbook, _INPUTS) == []
    assert validate_inputs(playbook, None) == [
        "Invoice amount is required.",
        "Vendor name is required.",
        "Order id is required.",
    ]
    assert validate_inputs(playbook, ["amount"]) == [
        "The inputs must be given as names and their values."
    ]


def test_a_playbook_without_inputs_reads_no_inputs_key() -> None:
    body = _invoice([{"id": "b", "when": "inputs > 3", "next_step_id": "cfo"}])
    body["inputs"] = []
    body["step_groups"][0]["steps"][2]["after"] = ["cfo"]
    rt = _runtime(body)
    rt.start()
    step = rt.playbook.step("check")
    assert step is not None
    # An old step output named ``inputs`` still reads as it always did.
    assert "inputs" not in rt.graph.context(step, {})
    assert rt.complete_step("check", _outputs(inputs=5))["branches_taken"] == ["b"]


# ── a restored run and its inputs ────────────────────────────────────────────


def _moved(*, inputs_before: dict[str, Any] | None = None) -> PlaybookRuntime:
    events: list[dict[str, Any]] = []
    original = _runtime(events=events)
    original.start(inputs=_INPUTS)
    moved = _runtime()
    if inputs_before is not None:
        moved.set_inputs(inputs_before)
    moved.restore(copy.deepcopy(events))
    return moved


def test_a_restored_run_does_not_route_on_inputs_it_was_not_given() -> None:
    moved = _moved()
    assert moved.unavailable_inputs() == ["amount", "vendor_name", "due", "order_id"]
    assert moved.public_inputs() == {}
    result = moved.complete_step("check", _outputs())
    assert result["routing"] == ROUTING_UNKNOWN
    moved.set_inputs(_INPUTS)
    assert moved.complete_step("check", _outputs())["active"] == ["cfo"]


def test_inputs_set_before_a_restore_carry_over() -> None:
    moved = _moved(inputs_before=_INPUTS)
    assert moved.unavailable_inputs() == []
    assert moved.complete_step("check", _outputs())["active"] == ["cfo"]


# ── what may be shown in clear ───────────────────────────────────────────────


def test_public_outputs_are_only_the_non_sensitive_fields() -> None:
    rt = _runtime()
    rt.start(inputs=_INPUTS)
    assert rt.public_outputs("check") == {}
    rt.complete_step(
        "check", _outputs(paid_on="2026-10-03", vendor_email="ap@acme.test", private="s")
    )
    assert rt.graph.status["check"] == DONE
    # total, vendor_email and the answer are sensitive: false; note, paid_on and the
    # undeclared ``private`` are sensitive.
    assert rt.public_outputs("check") == {
        "total": _money(500),
        "vendor_email": "ap@acme.test",
        "approved_on": "2026-10-01",
    }
    assert rt.public_outputs("ghost") == {}
    assert rt.public_outputs("cfo") == {}


def test_a_public_value_the_run_cannot_see_is_not_shown() -> None:
    rt = _runtime()
    rt.graph.outputs["check"] = {"total": UNAVAILABLE, "vendor_email": "bad", "approved_on": None}
    assert rt.public_outputs("check") == {}


def test_public_inputs_are_only_the_non_sensitive_valid_ones() -> None:
    rt = _runtime()
    assert rt.public_inputs() == {}
    rt.set_inputs({**_INPUTS, "due": "2026-12-01"})
    assert rt.public_inputs() == {"amount": _money(12000), "vendor_name": "Acme"}
    rt.set_inputs({**_INPUTS, "amount": 5})
    assert rt.public_inputs() == {"vendor_name": "Acme"}


def test_no_new_event_fields_carry_process_data() -> None:
    events: list[dict[str, Any]] = []
    rt = _runtime(events=events)
    rt.start(inputs=_INPUTS)
    rt.complete_step("check", _outputs())
    assert all("inputs" not in e and "public" not in str(sorted(e)) for e in events)


# ── what the model is told ───────────────────────────────────────────────────


def test_the_brief_says_how_to_write_each_typed_value() -> None:
    playbook = parse_playbook_v2(_invoice())
    step = playbook.step("check")
    assert step is not None
    brief = step_brief(step, {})
    assert (
        "Expected outputs (pass each to complete_step): note, "
        'total (money: {"amount": <number>, "currency": "<ISO 4217 code>"}), '
        "paid_on (date: YYYY-MM-DD; optional), vendor_email (email; optional)"
    ) in brief
    assert "- approved_on: When did the buyer approve it? (answer as date: YYYY-MM-DD)" in brief
    prose = playbook_prose(playbook, {}, [])
    assert (
        "Inputs:\n"
        '- amount (money: {"amount": <number>, "currency": "<ISO 4217 code>"})\n'
        "- vendor_name\n"
        "- due (date: YYYY-MM-DD; optional)\n"
        "- order_id: The order."
    ) in prose
