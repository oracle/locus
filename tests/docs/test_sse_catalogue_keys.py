# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""The SSE catalogue's payload keys match what the emit sites send.

``docs/concepts/sse-events.md`` is the naming contract subscribers read. Its
payload tables list, per event, the keys the event carries. This test reads
those rows and the ``emit(...)`` / ``emit_sync(...)`` calls under
``src/locus`` and fails when a documented key is never sent or a sent key is
not documented.

Documented keys are the backticked identifiers in a row's payload cell; a
``key="value"`` token documents ``key``, and text in parentheses explains a
key (its values, when it is present) rather than naming one. Emitted keys are
the keyword arguments of a call whose first argument is an ``EV_*`` constant
or a string literal; a ``**name`` splat is resolved to the string keys of a
``name = {...}`` literal (plus ``name["k"] = ...`` assignments) in the same
function, and an event with a splat that cannot be resolved is left alone.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "locus"
CATALOGUE = ROOT / "docs" / "concepts" / "sse-events.md"

# Backticked words in payload cells that are values, not keys.
VALUE_WORDS = frozenset({"none", "null", "true", "false"})
ROW = re.compile(r"^\| `([a-z][a-z0-9_.]*)` \| (.*) \|\s*$")
PARENTHESES = re.compile(r"\([^()]*\)")
BACKTICKED = re.compile(r"`([^`]+)`")
IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")
CONSTANT = re.compile(r'^(EV_[A-Z0-9_]+) = "([a-z0-9_.]+)"', re.MULTILINE)


def documented_keys(markdown: str) -> dict[str, set[str]]:
    """Event name to documented payload keys, from the payload tables."""
    rows: dict[str, set[str]] = {}
    payload_tables = markdown.split("## Span discipline", 1)[0]
    for line in payload_tables.splitlines():
        match = ROW.match(line)
        if not match or "." not in match.group(1):
            continue
        cell = match.group(2).split(" | ")[0]
        cell = PARENTHESES.sub("", cell)
        keys = rows.setdefault(match.group(1), set())
        for token in BACKTICKED.findall(cell):
            key = token.split("=", 1)[0].strip()
            if IDENTIFIER.fullmatch(key) and key not in VALUE_WORDS:
                keys.add(key)
    return rows


class _EmitSites(ast.NodeVisitor):
    """Collect the keyword names each event is emitted with."""

    def __init__(self, constants: dict[str, str]) -> None:
        self.constants = constants
        self.keys: dict[str, set[str]] = defaultdict(set)
        self.dynamic: set[str] = set()
        self._functions: list[ast.AST] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._functions.append(node)
        self.generic_visit(node)
        self._functions.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._functions.append(node)
        self.generic_visit(node)
        self._functions.pop()

    def _splat_keys(self, name: str) -> set[str] | None:
        """The string keys of ``name``, a dict built in the enclosing function."""
        if not self._functions:
            return None
        keys: set[str] | None = None
        for node in ast.walk(self._functions[-1]):
            target = None
            value = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign):
                target, value = node.target, node.value
            if isinstance(target, ast.Name) and target.id == name:
                if isinstance(value, ast.Dict):
                    keys = {
                        k.value
                        for k in value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    }
                elif (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "dict"
                ):
                    keys = {kw.arg for kw in value.keywords if kw.arg}
                else:
                    return None
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
                and keys is not None
            ):
                keys.add(target.slice.value)
        return keys

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        callee = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if callee in ("emit", "emit_sync") and node.args:
            first = node.args[0]
            event = None
            if isinstance(first, ast.Name):
                event = self.constants.get(first.id)
            elif isinstance(first, ast.Constant) and isinstance(first.value, str):
                event = first.value
            if event is not None:
                keys = self.keys[event]
                for keyword in node.keywords:
                    if keyword.arg is not None:
                        keys.add(keyword.arg)
                        continue
                    resolved = (
                        self._splat_keys(keyword.value.id)
                        if isinstance(keyword.value, ast.Name)
                        else None
                    )
                    if resolved is None:
                        self.dynamic.add(event)
                    else:
                        keys.update(resolved)
        self.generic_visit(node)


def emitted_keys(src: Path) -> tuple[dict[str, set[str]], set[str]]:
    """Event name to emitted keys, plus the events with an unresolvable splat."""
    constants = dict(CONSTANT.findall((src / "observability" / "emit.py").read_text()))
    sites = _EmitSites(constants)
    for path in sorted(src.rglob("*.py")):
        if path.name == "emit.py":
            continue
        sites.visit(ast.parse(path.read_text(encoding="utf-8")))
    return dict(sites.keys), sites.dynamic


def differences(
    docs: dict[str, set[str]], code: dict[str, set[str]], dynamic: set[str]
) -> list[str]:
    out: list[str] = []
    for event in sorted(docs):
        if event not in code or event in dynamic:
            continue
        never_sent = sorted(docs[event] - code[event])
        not_documented = sorted(code[event] - docs[event])
        if never_sent:
            out.append(f"{event}: documented but never sent: {never_sent}")
        if not_documented:
            out.append(f"{event}: sent but not documented: {not_documented}")
    return out


def test_the_catalogue_matches_the_emit_sites() -> None:
    docs = documented_keys(CATALOGUE.read_text(encoding="utf-8"))
    code, dynamic = emitted_keys(SRC)
    assert len(docs) > 60
    assert len(code) > 50
    assert differences(docs, code, dynamic) == []


def test_payload_cells_are_read_as_keys_not_values() -> None:
    markdown = (
        "| event | payload | note |\n"
        "|---|---|---|\n"
        '| `x.started` | `pipeline_kind="sequential"`, `stage`, `mode` (`"fast"` \\| `"slow"`) | Only with `flag=True` |\n'
        "| `x.completed` | `stage`, `success`, `error` (when `success` is false) |\n"
        "\n## Span discipline\n\n| `x.started` | inside `chat` |\n"
    )
    assert documented_keys(markdown) == {
        "x.started": {"pipeline_kind", "stage", "mode"},
        "x.completed": {"stage", "success", "error"},
    }


def test_emit_sites_resolve_splats_built_in_the_function(tmp_path: Path) -> None:
    src = tmp_path / "locus"
    (src / "observability").mkdir(parents=True)
    (src / "observability" / "emit.py").write_text(
        'EV_A = "x.a"\nEV_B = "x.b"\nEV_C = "x.c"\n', encoding="utf-8"
    )
    (src / "site.py").write_text(
        "from locus.observability.emit import EV_A, EV_B, EV_C, emit, emit_sync\n"
        "\n"
        "async def one():\n"
        "    await emit(EV_A, method='m', success=True)\n"
        "\n"
        "def two(payload):\n"
        "    data: dict = {'method': 'm', 'duration_ms': 1.0}\n"
        "    if payload:\n"
        "        data['error'] = 'x'\n"
        "    emit_sync(EV_B, **data)\n"
        "    emit_sync('x.b', frames=3)\n"
        "    emit_sync(EV_C, **payload)\n",
        encoding="utf-8",
    )
    code, dynamic = emitted_keys(src)
    assert code["x.a"] == {"method", "success"}
    assert code["x.b"] == {"method", "duration_ms", "error", "frames"}
    assert dynamic == {"x.c"}
    assert differences({"x.a": {"method"}, "x.c": {"anything"}}, code, dynamic) == [
        "x.a: sent but not documented: ['success']"
    ]
