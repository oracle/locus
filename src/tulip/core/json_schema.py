# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate a JSON value against a JSON Schema, without a dependency.

Structured output from a caller-supplied schema (``tulip-code --json-schema``,
an HTTP API taking a schema per request) has no Pydantic model to validate
against — only the schema document. ``jsonschema`` would do it, but it is a
dependency tree for something the SDK needs a few hundred lines of, and a core
install stays at httpx + pydantic.

Covered: the keywords structured-output schemas use in practice —

* ``type`` (one or a list; ``integer`` accepts ``1.0``, ``number`` and
  ``integer`` reject booleans), ``enum``, ``const``
* objects: ``properties``, ``required``, ``additionalProperties`` (boolean or
  schema), ``patternProperties``, ``minProperties``/``maxProperties``
* arrays: ``items`` (one schema), ``prefixItems``, ``minItems``/``maxItems``,
  ``uniqueItems``, ``contains``
* strings: ``minLength``/``maxLength``, ``pattern`` (``re.search``)
* numbers: ``minimum``/``maximum``, ``exclusiveMinimum``/``exclusiveMaximum``
  (draft 2020-12 numeric form), ``multipleOf``
* ``allOf``/``anyOf``/``oneOf``/``not``, ``if``/``then``/``else``
* local ``$ref`` (``#``, ``#/$defs/...``, ``#/definitions/...``, any JSON
  pointer into the root document)

Annotations (``title``, ``description``, ``default``, ``examples``,
``format``…) and keywords outside that list are ignored, as the specification
allows for vocabularies a validator does not implement. Remote ``$ref`` is
refused rather than fetched.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any


__all__ = ["SchemaError", "check_schema", "validate"]


class SchemaError(ValueError):
    """The schema itself is unusable — not the value being checked."""


_TYPES = frozenset({"object", "array", "string", "number", "integer", "boolean", "null"})


def check_schema(schema: Any) -> None:
    """Raise :class:`SchemaError` when ``schema`` cannot be validated against.

    Checks the shape this module relies on — an object or boolean schema,
    known ``type`` names, resolvable local ``$ref`` s, compilable patterns —
    so a bad schema is reported once, up front, instead of as a confusing
    validation failure on every value.
    """
    _check(schema, schema, "#")


def validate(instance: Any, schema: Any) -> list[str]:
    """Every way ``instance`` fails ``schema``, as readable messages.

    An empty list means the value is valid. Each message starts with the JSON
    path of the offending value (``$.items[2].name``), because the reader is
    often a model that has to fix exactly that value and try again.

    Raises:
        SchemaError: The schema is unusable (see :func:`check_schema`).
    """
    errors: list[str] = []
    _validate(instance, schema, schema, "$", errors)
    return errors


# --------------------------------------------------------------------------
# schema checking
# --------------------------------------------------------------------------


def _check(node: Any, root: Any, where: str) -> None:
    if isinstance(node, bool):
        return
    if not isinstance(node, Mapping):
        raise SchemaError(f"{where}: a schema must be an object or a boolean")
    kinds = node.get("type")
    if kinds is not None:
        for kind in kinds if isinstance(kinds, list) else [kinds]:
            if kind not in _TYPES:
                raise SchemaError(f"{where}: unknown type {kind!r}")
    ref = node.get("$ref")
    if ref is not None:
        _resolve(str(ref), root)
    pattern = node.get("pattern")
    if pattern is not None:
        try:
            re.compile(str(pattern))
        except re.error as exc:
            raise SchemaError(f"{where}: bad pattern {pattern!r}: {exc}") from exc
    for key in ("properties", "patternProperties", "$defs", "definitions"):
        for name, sub in (node.get(key) or {}).items():
            _check(sub, root, f"{where}/{key}/{name}")
    for key in ("items", "additionalProperties", "not", "contains", "if", "then", "else"):
        if key in node and not isinstance(node[key], list):
            _check(node[key], root, f"{where}/{key}")
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        for i, sub in enumerate(node.get(key) or []):
            _check(sub, root, f"{where}/{key}/{i}")


def _resolve(ref: str, root: Any) -> Any:
    """The schema a local ``$ref`` points at."""
    if not ref.startswith("#"):
        raise SchemaError(f"only local $ref is supported, not {ref!r}")
    node = root
    for raw in ref[1:].split("/")[1:] if ref != "#" else []:
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise SchemaError(f"$ref {ref!r} does not resolve")
    return node


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def _type_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list | tuple):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _is_type(value: Any, kind: str) -> bool:
    actual = _type_of(value)
    if kind == "number":
        return actual in ("number", "integer")
    if kind == "integer":
        return actual == "integer" or (
            actual == "number" and math.isfinite(value) and float(value).is_integer()
        )
    return actual == kind


def _equal(a: Any, b: Any) -> bool:
    """JSON equality: ``1 == 1.0``, but ``True`` is not ``1``."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return bool(a == b)


def _validate(value: Any, node: Any, root: Any, path: str, errors: list[str]) -> None:
    if node is True:
        return
    if node is False:
        errors.append(f"{path}: no value is allowed here")
        return
    if not isinstance(node, Mapping):
        raise SchemaError(f"{path}: a schema must be an object or a boolean")

    if "$ref" in node:
        _validate(value, _resolve(str(node["$ref"]), root), root, path, errors)

    kinds = node.get("type")
    if kinds is not None:
        allowed = kinds if isinstance(kinds, list) else [kinds]
        if not any(_is_type(value, k) for k in allowed):
            errors.append(f"{path}: expected {' or '.join(allowed)}, got {_type_of(value)}")
            return  # the rest would only repeat the same mistake in other words

    if "enum" in node and not any(_equal(value, option) for option in node["enum"]):
        errors.append(f"{path}: must be one of {node['enum']!r}")
    if "const" in node and not _equal(value, node["const"]):
        errors.append(f"{path}: must be {node['const']!r}")

    kind = _type_of(value)
    if kind == "object":
        _validate_object(value, node, root, path, errors)
    elif kind == "array":
        _validate_array(list(value), node, root, path, errors)
    elif kind == "string":
        _validate_string(value, node, path, errors)
    elif kind in ("number", "integer"):
        _validate_number(value, node, path, errors)

    _validate_combinators(value, node, root, path, errors)


def _validate_object(
    value: Mapping[str, Any], node: Mapping[str, Any], root: Any, path: str, errors: list[str]
) -> None:
    properties = node.get("properties") or {}
    patterns = node.get("patternProperties") or {}
    for name in node.get("required") or []:
        if name not in value:
            errors.append(f"{path}: missing required property {name!r}")
    for name, item in value.items():
        where = f"{path}.{name}"
        matched = False
        if name in properties:
            matched = True
            _validate(item, properties[name], root, where, errors)
        for pattern, sub in patterns.items():
            if re.search(pattern, name):
                matched = True
                _validate(item, sub, root, where, errors)
        if not matched and "additionalProperties" in node:
            extra = node["additionalProperties"]
            if extra is False:
                errors.append(f"{path}: unexpected property {name!r}")
            else:
                _validate(item, extra, root, where, errors)
    if "minProperties" in node and len(value) < node["minProperties"]:
        errors.append(f"{path}: needs at least {node['minProperties']} properties")
    if "maxProperties" in node and len(value) > node["maxProperties"]:
        errors.append(f"{path}: allows at most {node['maxProperties']} properties")


def _validate_array(
    value: list[Any], node: Mapping[str, Any], root: Any, path: str, errors: list[str]
) -> None:
    prefix = node.get("prefixItems") or []
    for i, sub in enumerate(prefix[: len(value)]):
        _validate(value[i], sub, root, f"{path}[{i}]", errors)
    items = node.get("items")
    if isinstance(items, list):  # draft-07 tuple form
        for i, sub in enumerate(items[: len(value)]):
            _validate(value[i], sub, root, f"{path}[{i}]", errors)
    elif items is not None:
        for i in range(len(prefix), len(value)):
            _validate(value[i], items, root, f"{path}[{i}]", errors)
    if "minItems" in node and len(value) < node["minItems"]:
        errors.append(f"{path}: needs at least {node['minItems']} items")
    if "maxItems" in node and len(value) > node["maxItems"]:
        errors.append(f"{path}: allows at most {node['maxItems']} items")
    if node.get("uniqueItems"):
        for i, item in enumerate(value):
            if any(_equal(item, other) for other in value[:i]):
                errors.append(f"{path}: items must be unique; [{i}] repeats an earlier one")
                break
    if "contains" in node and not any(
        not _collect(item, node["contains"], root, path) for item in value
    ):
        errors.append(f"{path}: no item matches the 'contains' schema")


def _validate_string(value: str, node: Mapping[str, Any], path: str, errors: list[str]) -> None:
    if "minLength" in node and len(value) < node["minLength"]:
        errors.append(f"{path}: needs at least {node['minLength']} characters")
    if "maxLength" in node and len(value) > node["maxLength"]:
        errors.append(f"{path}: allows at most {node['maxLength']} characters")
    if "pattern" in node and not re.search(str(node["pattern"]), value):
        errors.append(f"{path}: does not match pattern {node['pattern']!r}")


def _validate_number(value: float, node: Mapping[str, Any], path: str, errors: list[str]) -> None:
    if "minimum" in node and value < node["minimum"]:
        errors.append(f"{path}: must be >= {node['minimum']}")
    if "maximum" in node and value > node["maximum"]:
        errors.append(f"{path}: must be <= {node['maximum']}")
    low = node.get("exclusiveMinimum")
    if isinstance(low, int | float) and not isinstance(low, bool) and value <= low:
        errors.append(f"{path}: must be > {low}")
    high = node.get("exclusiveMaximum")
    if isinstance(high, int | float) and not isinstance(high, bool) and value >= high:
        errors.append(f"{path}: must be < {high}")
    step = node.get("multipleOf")
    if step:
        ratio = value / step
        if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
            errors.append(f"{path}: must be a multiple of {step}")


def _collect(value: Any, node: Any, root: Any, path: str) -> list[str]:
    found: list[str] = []
    _validate(value, node, root, path, found)
    return found


def _validate_combinators(
    value: Any, node: Mapping[str, Any], root: Any, path: str, errors: list[str]
) -> None:
    for sub in node.get("allOf") or []:
        _validate(value, sub, root, path, errors)
    if "anyOf" in node:
        attempts = [_collect(value, sub, root, path) for sub in node["anyOf"]]
        if all(attempts):
            closest = min(attempts, key=len)
            errors.append(f"{path}: matches none of anyOf; closest: {'; '.join(closest)}")
    if "oneOf" in node:
        passing = sum(1 for sub in node["oneOf"] if not _collect(value, sub, root, path))
        if passing != 1:
            errors.append(f"{path}: must match exactly one of oneOf, matched {passing}")
    if "not" in node and not _collect(value, node["not"], root, path):
        errors.append(f"{path}: must not match the 'not' schema")
    if "if" in node:
        branch = "then" if not _collect(value, node["if"], root, path) else "else"
        if branch in node:
            _validate(value, node[branch], root, path, errors)
