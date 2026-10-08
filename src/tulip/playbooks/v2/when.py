# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``when`` language of a playbook branch: a small, safe condition grammar.

Ported from the registry (``tulip_registry.definitions.when``), which parses every
``when`` at publish; this copy evaluates them at run time, wherever the run's loop is (the
gateway, or a box runner). Every copy must agree on every condition, so
``tests/unit/playbooks_v2/test_when.py`` runs the registry's own test vectors here. Change
one and change the others in the same breath.

A branch is taken when its ``when`` holds against the branching step's outputs. The
grammar mirrors observai's branch evaluator (comparisons, ``AND`` / ``OR`` / ``NOT``,
parentheses, dotted paths, quoted literals) and adds the four words its authored
playbooks actually use: ``contains``, ``equals``, ``is empty`` and ``always``::

    selected_branch_ids contains tx_locking
    outputs.router.scope == 'CLUSTER-WIDE' AND outputs.router.error_count > 10
    NOT (findings is empty)
    always

Nothing here calls ``eval``. A condition is tokenized and parsed into a small tree
once, at publish time, so an author learns about a typo then rather than when a run
reaches the branch; :func:`evaluate_when` walks that tree and never raises.

Two rules worth knowing:

* On the left of any operator, and on either side of a symbolic one (``==``, ``!=``,
  ``>``, ``<``, ``>=``, ``<=``), a bare word is a **path** into the context, as in
  observai. On the right of ``contains`` and ``equals`` a bare word is a **literal**:
  ``selected_branch_ids contains tx_locking`` means the string ``"tx_locking"``.
* A path that does not resolve is ``null``. Ordering comparisons against ``null`` (or
  between values that cannot be ordered) are false, never an error.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


#: The longest condition accepted. A branch condition is one line an author reads.
MAX_WHEN_LENGTH = 500
#: The deepest nesting of parentheses and NOTs accepted.
MAX_WHEN_DEPTH = 32

_KEYWORDS = frozenset(
    {"and", "or", "not", "contains", "equals", "is", "empty", "always", "true", "false", "null"}
)

_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<string>'[^']*'|"[^"]*")
  | (?P<number>-?\d+(?:\.\d+)?(?![A-Za-z_]))
  | (?P<op>==|!=|>=|<=|>|<)
  | (?P<lparen>\()
  | (?P<rparen>\))
  | (?P<word>[A-Za-z_][A-Za-z0-9_\-]*(?:\.[A-Za-z0-9_\-]+)*)
    """,
    re.VERBOSE,
)


class WhenSyntaxError(ValueError):
    """A ``when`` condition that does not parse; the message names where."""


@dataclass(frozen=True)
class _Token:
    kind: str  # string | number | op | lparen | rparen | word | keyword
    text: str
    pos: int


# ── the tree ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Path:
    """A dotted path into the evaluation context."""

    dotted: str


@dataclass(frozen=True)
class Literal:
    """A literal value: string, number, true, false or null."""

    value: str | int | float | bool | None


Operand = Path | Literal


@dataclass(frozen=True)
class Always:
    """``always``: the branch is unconditional."""


@dataclass(frozen=True)
class Truthy:
    """A bare operand: true when its value is truthy."""

    operand: Operand


@dataclass(frozen=True)
class Compare:
    """``left OP right`` with a symbolic operator (``equals`` is ``==``)."""

    op: str
    left: Operand
    right: Operand


@dataclass(frozen=True)
class Contains:
    """``left contains right``: membership in a list, a key of a mapping, or a substring."""

    left: Operand
    right: Operand


@dataclass(frozen=True)
class Empty:
    """``operand is empty`` / ``operand is not empty``."""

    operand: Operand
    negate: bool


@dataclass(frozen=True)
class Not:
    operand: Node


@dataclass(frozen=True)
class AllOf:
    operands: tuple[Node, ...]


@dataclass(frozen=True)
class AnyOf:
    operands: tuple[Node, ...]


Node = Always | Truthy | Compare | Contains | Empty | Not | AllOf | AnyOf


# ── parsing ──────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None:
            raise WhenSyntaxError(f"unexpected {text[pos]!r} at position {pos}")
        kind = match.lastgroup or ""
        value = match.group()
        if kind != "ws":
            if kind == "word" and value.lower() in _KEYWORDS:
                kind = "keyword"
                value = value.lower()
            tokens.append(_Token(kind, value, pos))
        pos = match.end()
    return tokens


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.i = 0
        self.depth = 0

    def peek(self) -> _Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def keyword(self, *words: str) -> bool:
        tok = self.peek()
        return tok is not None and tok.kind == "keyword" and tok.text in words

    def take(self) -> _Token:
        # Every caller has peeked first, so there is a token to take.
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def fail(self, what: str) -> WhenSyntaxError:
        tok = self.peek()
        where = f"{tok.text!r} at position {tok.pos}" if tok else "the end"
        return WhenSyntaxError(f"expected {what}, found {where}")

    def parse(self) -> Node:
        if not self.tokens:
            raise WhenSyntaxError("the condition is empty")
        node = self.or_expr()
        if self.peek() is not None:
            raise self.fail("AND, OR or the end of the condition")
        return node

    def nest(self) -> None:
        self.depth += 1
        if self.depth > MAX_WHEN_DEPTH:
            raise WhenSyntaxError(f"the condition nests deeper than {MAX_WHEN_DEPTH}")

    def or_expr(self) -> Node:
        parts = [self.and_expr()]
        while self.keyword("or"):
            self.take()
            parts.append(self.and_expr())
        return parts[0] if len(parts) == 1 else AnyOf(tuple(parts))

    def and_expr(self) -> Node:
        parts = [self.not_expr()]
        while self.keyword("and"):
            self.take()
            parts.append(self.not_expr())
        return parts[0] if len(parts) == 1 else AllOf(tuple(parts))

    def not_expr(self) -> Node:
        if self.keyword("not"):
            self.take()
            self.nest()
            node = Not(self.not_expr())
            self.depth -= 1
            return node
        return self.atom()

    def atom(self) -> Node:
        tok = self.peek()
        if tok is not None and tok.kind == "lparen":
            self.take()
            self.nest()
            node = self.or_expr()
            self.depth -= 1
            closing = self.peek()
            if closing is None or closing.kind != "rparen":
                raise self.fail("')'")
            self.take()
            return node
        if self.keyword("always"):
            self.take()
            return Always()
        return self.test()

    def operand(self, *, bare_is_literal: bool) -> Operand:
        tok = self.peek()
        if tok is None:
            raise self.fail("a value")
        if tok.kind == "string":
            self.take()
            return Literal(tok.text[1:-1])
        if tok.kind == "number":
            self.take()
            return Literal(float(tok.text) if "." in tok.text else int(tok.text))
        if tok.kind == "keyword" and tok.text in ("true", "false", "null"):
            self.take()
            return Literal({"true": True, "false": False, "null": None}[tok.text])
        if tok.kind == "word":
            self.take()
            return Literal(tok.text) if bare_is_literal else Path(tok.text)
        raise self.fail("a path or a value")

    def test(self) -> Node:
        left = self.operand(bare_is_literal=False)
        tok = self.peek()
        if tok is not None and tok.kind == "op":
            self.take()
            return Compare(tok.text, left, self.operand(bare_is_literal=False))
        if self.keyword("equals"):
            self.take()
            return Compare("==", left, self.operand(bare_is_literal=True))
        if self.keyword("contains"):
            self.take()
            return Contains(left, self.operand(bare_is_literal=True))
        if self.keyword("is"):
            self.take()
            negate = False
            if self.keyword("not"):
                self.take()
                negate = True
            if not self.keyword("empty"):
                raise self.fail("'empty' after 'is'")
            self.take()
            return Empty(left, negate)
        return Truthy(left)


def parse_when(text: str) -> Node:
    """Parse a ``when`` condition into its tree, or raise :class:`WhenSyntaxError`."""
    if not isinstance(text, str):
        raise WhenSyntaxError("a condition is text")
    if len(text) > MAX_WHEN_LENGTH:
        raise WhenSyntaxError(f"the condition is longer than {MAX_WHEN_LENGTH} characters")
    return _Parser(text).parse()


def when_paths(node: Node) -> list[str]:
    """Every path a condition reads, in order of appearance (duplicates kept once)."""
    seen: list[str] = []

    def visit_operand(operand: Operand) -> None:
        if isinstance(operand, Path) and operand.dotted not in seen:
            seen.append(operand.dotted)

    def visit(n: Node) -> None:
        if isinstance(n, Truthy | Empty):
            visit_operand(n.operand)
        elif isinstance(n, Compare | Contains):
            visit_operand(n.left)
            visit_operand(n.right)
        elif isinstance(n, Not):
            visit(n.operand)
        elif isinstance(n, AllOf | AnyOf):
            for part in n.operands:
                visit(part)

    visit(node)
    return seen


# ── evaluation ───────────────────────────────────────────────────────────────


def _resolve(context: Mapping[str, Any], dotted: str) -> Any:
    current: Any = context
    for part in dotted.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, str) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _value(context: Mapping[str, Any], operand: Operand) -> Any:
    if isinstance(operand, Path):
        return _resolve(context, operand.dotted)
    return operand.value


def _compare(op: str, left: Any, right: Any) -> bool:
    if op == "==":
        return bool(left == right)
    if op == "!=":
        return bool(left != right)
    if left is None or right is None or isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        if op == ">":
            return bool(left > right)
        if op == "<":
            return bool(left < right)
        if op == ">=":
            return bool(left >= right)
        return bool(left <= right)
    except TypeError:
        return False


def _contains(container: Any, item: Any) -> bool:
    if isinstance(container, str):
        return isinstance(item, str) and item in container
    if isinstance(container, Mapping):
        return item in container
    if isinstance(container, Sequence | set | frozenset):
        return item in container
    return False


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str | Mapping | Sequence | set | frozenset):
        return len(value) == 0
    return False


def _eval(node: Node, context: Mapping[str, Any]) -> bool:
    if isinstance(node, Always):
        return True
    if isinstance(node, Truthy):
        return bool(_value(context, node.operand))
    if isinstance(node, Compare):
        return _compare(node.op, _value(context, node.left), _value(context, node.right))
    if isinstance(node, Contains):
        return _contains(_value(context, node.left), _value(context, node.right))
    if isinstance(node, Empty):
        return _is_empty(_value(context, node.operand)) != node.negate
    if isinstance(node, Not):
        return not _eval(node.operand, context)
    if isinstance(node, AllOf):
        return all(_eval(part, context) for part in node.operands)
    return any(_eval(part, context) for part in node.operands)


def evaluate_when(condition: str | Node, context: Mapping[str, Any]) -> bool:
    """Whether ``condition`` holds against ``context``. Never raises: bad input is false."""
    try:
        node = parse_when(condition) if isinstance(condition, str) else condition
        return _eval(node, context)
    except (WhenSyntaxError, RecursionError):
        return False


__all__ = [
    "MAX_WHEN_DEPTH",
    "MAX_WHEN_LENGTH",
    "Node",
    "WhenSyntaxError",
    "evaluate_when",
    "parse_when",
    "when_paths",
]
