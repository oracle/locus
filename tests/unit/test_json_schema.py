# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The dependency-free JSON Schema validator behind structured output."""

from __future__ import annotations

import pytest

from tulip.core.json_schema import SchemaError, check_schema, validate


PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0},
        "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


def test_a_valid_value_has_no_errors() -> None:
    assert validate({"name": "Ada", "age": 36, "tags": ["math"]}, PERSON) == []


def test_errors_name_the_path_of_the_bad_value() -> None:
    errors = validate({"name": "", "age": -1, "tags": ["a", 3]}, PERSON)
    assert "$.name: needs at least 1 characters" in errors
    assert "$.age: must be >= 0" in errors
    assert "$.tags[1]: expected string, got integer" in errors


def test_missing_and_unexpected_properties() -> None:
    errors = validate({"age": 3, "extra": True}, PERSON)
    assert "$: missing required property 'name'" in errors
    assert "$: unexpected property 'extra'" in errors


@pytest.mark.parametrize(
    ("value", "kind", "ok"),
    [
        (1, "integer", True),
        (1.0, "integer", True),
        (1.5, "integer", False),
        (True, "integer", False),
        (True, "number", False),
        (2, "number", True),
        (None, "null", True),
        ([], "array", True),
        ({}, "object", True),
        ("x", ["string", "null"], True),
        (3, ["string", "null"], False),
    ],
)
def test_types(value: object, kind: object, ok: bool) -> None:
    assert (validate(value, {"type": kind}) == []) is ok


def test_enum_and_const_use_json_equality() -> None:
    assert validate(1.0, {"enum": [1, 2]}) == []
    assert validate(True, {"enum": [1]}) != []
    assert validate("b", {"const": "a"}) == ["$: must be 'a'"]
    assert validate({"a": [1]}, {"const": {"a": [1.0]}}) == []


def test_unique_items_and_contains() -> None:
    assert validate([1, 1], {"uniqueItems": True}) != []
    assert validate([1, "x"], {"contains": {"type": "string"}}) == []
    assert validate([1, 2], {"contains": {"type": "string"}}) != []


def test_array_bounds_prefix_items_and_tuple_items() -> None:
    schema = {"prefixItems": [{"type": "string"}], "items": {"type": "integer"}, "maxItems": 3}
    assert validate(["a", 1, 2], schema) == []
    assert validate([1, 1], schema) == ["$[0]: expected string, got integer"]
    assert validate(["a", 1, 2, 3], schema) == ["$: allows at most 3 items"]
    assert validate([], {"minItems": 1}) == ["$: needs at least 1 items"]
    assert validate([1, "x"], {"items": [{"type": "integer"}, {"type": "integer"}]}) != []


def test_strings_and_numbers() -> None:
    assert validate("abc", {"maxLength": 2}) != []
    assert validate("abc", {"pattern": "^a"}) == []
    assert validate("abc", {"pattern": "^b"}) != []
    assert validate(5, {"exclusiveMinimum": 5}) == ["$: must be > 5"]
    assert validate(5, {"exclusiveMaximum": 5}) == ["$: must be < 5"]
    assert validate(6, {"maximum": 5}) == ["$: must be <= 5"]
    assert validate(0.3, {"multipleOf": 0.1}) == []
    assert validate(7, {"multipleOf": 2}) != []


def test_object_counts_and_pattern_properties() -> None:
    schema = {
        "patternProperties": {"^x_": {"type": "integer"}},
        "additionalProperties": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 2,
    }
    assert validate({"x_a": 1, "b": "s"}, schema) == []
    assert validate({"x_a": "s"}, schema) == ["$.x_a: expected integer, got string"]
    assert validate({"b": 1}, schema) == ["$.b: expected string, got integer"]
    assert validate({}, schema) == ["$: needs at least 1 properties"]
    assert validate({"a": "1", "b": "2", "c": "3"}, schema) == ["$: allows at most 2 properties"]


def test_combinators() -> None:
    assert validate(3, {"anyOf": [{"type": "string"}, {"type": "integer"}]}) == []
    assert "matches none of anyOf" in validate(1.5, {"anyOf": [{"type": "string"}]})[0]
    one_of = {"oneOf": [{"type": "integer"}, {"type": "number"}]}
    assert validate(1, one_of) == ["$: must match exactly one of oneOf, matched 2"]
    assert validate(1.5, one_of) == []
    assert validate(1, {"allOf": [{"type": "integer"}, {"minimum": 2}]}) == ["$: must be >= 2"]
    assert validate("x", {"not": {"type": "string"}}) != []
    cond = {"if": {"type": "string"}, "then": {"minLength": 2}, "else": {"minimum": 0}}
    assert validate("a", cond) != []
    assert validate(-1, cond) != []
    assert validate("ab", cond) == []


def test_a_non_json_value_is_named_by_its_python_type() -> None:
    assert validate({1, 2}, {"type": "array"}) == ["$: expected array, got set"]


def test_if_without_the_matching_branch_constrains_nothing() -> None:
    assert validate(-1, {"if": {"type": "string"}, "then": {"minLength": 2}}) == []


def test_boolean_schemas() -> None:
    assert validate(1, True) == []
    assert validate(1, False) == ["$: no value is allowed here"]
    assert validate({"a": 1}, {"properties": {"a": False}}) == ["$.a: no value is allowed here"]


def test_local_refs_resolve() -> None:
    schema = {
        "$defs": {"item": {"type": "object", "required": ["id"]}},
        "type": "array",
        "items": {"$ref": "#/$defs/item"},
    }
    assert validate([{"id": 1}], schema) == []
    assert validate([{}], schema) == ["$[0]: missing required property 'id'"]
    tree = {"type": "object", "properties": {"child": {"$ref": "#"}}}
    assert validate({"child": {"child": {}}}, tree) == []
    assert validate({"child": 3}, tree) == ["$.child: expected object, got integer"]


def test_bad_schemas_are_refused_up_front() -> None:
    with pytest.raises(SchemaError, match="unknown type"):
        check_schema({"type": "text"})
    with pytest.raises(SchemaError, match="does not resolve"):
        check_schema({"$ref": "#/$defs/missing"})
    with pytest.raises(SchemaError, match="only local"):
        check_schema({"$ref": "https://example.com/s.json"})
    with pytest.raises(SchemaError, match="bad pattern"):
        check_schema({"pattern": "("})
    with pytest.raises(SchemaError, match="object or a boolean"):
        check_schema({"properties": {"a": 3}})
    with pytest.raises(SchemaError):
        validate(1, 3)
    check_schema({"$defs": {"a": {"items": [{"type": "string"}]}}, "anyOf": [True]})
    check_schema({"$ref": "#/anyOf/0", "anyOf": [{"type": "string"}]})
