# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""AST-based arithmetic evaluator shared by test fixtures.

Fixtures previously used ``eval`` to evaluate LLM-emitted expressions, which
is both a latent code-injection sink and a linter noise source (S307). This
evaluator accepts constant literals and binary/unary arithmetic only.
"""

from __future__ import annotations

import ast
import operator as _op
from typing import Any


_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.USub: _op.neg,
    ast.UAdd: _op.pos,
}


def safe_math_eval(expression: str) -> float:
    """Evaluate ``expression`` as pure arithmetic.

    Raises:
        ValueError: on unsupported AST nodes.
        SyntaxError: on malformed input.
        ZeroDivisionError: on division by zero.
    """
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    return _eval(tree)
