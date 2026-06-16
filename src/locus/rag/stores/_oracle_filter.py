# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Mongo-style metadata filter compiler for OracleVectorStore.

Pulled out of ``oracle.py`` so the filter grammar and its tests stay
focused on one concern. Grammar mirrors the langchain-oracle
``OracleVS`` filter DSL — implemented natively, no external dep.

Leaf operators (compare against a JSON_VALUE):
    $eq, $ne, $gt, $gte, $lt, $lte

Set operators (IN / NOT IN):
    $in, $nin

Logical operators:
    $and  (list of expressions, AND-joined)
    $or   (list of expressions, OR-joined)
    $not  (single nested expression)

Implicit AND across top-level keys::

    {"category": "x", "year": {"$gt": 2020}}  # implicit AND

Generated SQL is parameterised: every operand binds through the
``params`` dict the caller passes in. No string interpolation of user
input, no SQL injection.
"""

from __future__ import annotations

from typing import Any


_LEAF_OPS: dict[str, str] = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}
_LOGICAL_OPS = frozenset({"$and", "$or", "$not"})
_SET_OPS = frozenset({"$in", "$nin"})


def stringify_filter_value(value: Any) -> str:
    """Coerce a filter operand to the string form ``JSON_VALUE`` returns.

    Oracle's ``JSON_VALUE`` returns VARCHAR2 by default, so the filter
    binds must be strings for the comparison to match what the engine
    sees. Booleans become lowercase JSON literals (``true``/``false``).
    None becomes the empty string — callers should use ``$ne`` against
    a sentinel for absence checks instead.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def compile_metadata_filter(
    filter_: dict[str, Any] | None,
    params: dict[str, Any],
    *,
    metadata_column: str,
    prefix: str = "mf",
) -> str:
    """Compile a Mongo-style filter dict into an Oracle SQL predicate.

    Returns the empty string for a falsy filter so callers can splice
    the result into a WHERE clause without checking ``None``. Otherwise
    returns a fully-parenthesised predicate suitable for AND-joining
    with other WHERE clauses.

    Args:
        filter_: The filter dict (or None).
        params: Mutable dict where bind variables get appended.
        metadata_column: Name of the JSON column (passed in so the
            compiler stays decoupled from OracleVectorConfig).
        prefix: Bind-name prefix to keep params distinct when the
            caller compiles multiple filters into one query.
    """
    if not filter_:
        return ""
    sql = _compile_node(filter_, params, prefix, metadata_column)
    return f"({sql})"


def _compile_node(
    node: Any,
    params: dict[str, Any],
    prefix: str,
    metadata_column: str,
) -> str:
    if not isinstance(node, dict):
        raise TypeError(f"metadata filter node must be a dict, got {type(node).__name__}")
    if not node:
        return "1=1"

    parts: list[str] = []
    for i, (key, value) in enumerate(node.items()):
        sub_prefix = f"{prefix}_{i}"
        if key in _LOGICAL_OPS:
            parts.append(_compile_logical(key, value, params, sub_prefix, metadata_column))
        elif key.startswith("$"):
            raise ValueError(f"Unknown top-level operator: {key!r}")
        else:
            parts.append(_compile_field(key, value, params, sub_prefix, metadata_column))
    return " AND ".join(parts) if len(parts) > 1 else parts[0]


def _compile_logical(
    op: str,
    value: Any,
    params: dict[str, Any],
    prefix: str,
    metadata_column: str,
) -> str:
    if op == "$not":
        inner = _compile_node(value, params, f"{prefix}n", metadata_column)
        return f"NOT ({inner})"
    if not isinstance(value, list):
        raise TypeError(f"{op} expects a list of expressions, got {type(value).__name__}")
    joiner = " AND " if op == "$and" else " OR "
    sub_parts = [
        f"({_compile_node(item, params, f'{prefix}_{i}', metadata_column)})"
        for i, item in enumerate(value)
    ]
    if not sub_parts:
        return "1=1"
    return joiner.join(sub_parts)


def _compile_field(
    field: str,
    value: Any,
    params: dict[str, Any],
    prefix: str,
    metadata_column: str,
) -> str:
    if not field.isidentifier():
        raise ValueError(f"Invalid metadata field name: {field!r}")
    col = f"JSON_VALUE({metadata_column}, '$.{field}')"

    if not isinstance(value, dict):
        pname = f"{prefix}_eq"
        params[pname] = stringify_filter_value(value)
        return f"{col} = :{pname}"

    parts: list[str] = []
    for i, (op, op_val) in enumerate(value.items()):
        op_prefix = f"{prefix}_{i}"
        if op in _LEAF_OPS:
            pname = f"{op_prefix}_{op.lstrip('$')}"
            params[pname] = stringify_filter_value(op_val)
            parts.append(f"{col} {_LEAF_OPS[op]} :{pname}")
        elif op in _SET_OPS:
            if not isinstance(op_val, (list, tuple)):
                raise ValueError(f"{op} expects a list, got {type(op_val).__name__}")
            if not op_val:
                parts.append("1=0" if op == "$in" else "1=1")
                continue
            placeholders = []
            for j, item in enumerate(op_val):
                pname = f"{op_prefix}_{op.lstrip('$')}_{j}"
                params[pname] = stringify_filter_value(item)
                placeholders.append(f":{pname}")
            joined = ", ".join(placeholders)
            op_sql = "IN" if op == "$in" else "NOT IN"
            parts.append(f"{col} {op_sql} ({joined})")
        else:
            raise ValueError(f"Unknown operator {op!r} on field {field!r}")
    return " AND ".join(parts) if len(parts) > 1 else parts[0]
