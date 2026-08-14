#!/usr/bin/env python
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Check every Python snippet in this repo's Markdown against the real SDK.

The tests prove the code works. Nothing proved the *documentation* works, and
a September-2026 audit of this repo found the cost of that gap: the README
advertised `MCPClient` in its capability matrix while the name was absent from
`tulip.integrations.__all__`, `examples/README.md` imported two hooks from a
module that has never contained them, and the quickstart's headline snippet
raised `SyntaxError` for every reader who pasted it. Each was a
copy-paste-and-it-fails bug sitting in the highest-traffic file in the project,
and each would have been caught by running this once.

Three checks, cheapest first:

1. **Syntax** — every block must compile. Not merely parse: ``ast.parse``
   accepts top-level ``await`` and only ``compile`` rejects it, which is
   exactly the hole the broken quickstart went through.
2. **Symbols** — every ``from tulip... import X`` must resolve against the
   installed SDK.
3. **Keywords** — keyword arguments passed to an imported Tulip callable must
   exist on it.

Deliberately static. It imports the SDK but never executes a snippet, so it
needs no API key, no network, and no services, which is what makes it cheap
enough to gate every PR.

Two things it is careful about, both learned from false positives:

- README prose legitimately shows fragments — ``async for event in
  agent.run(...)`` on its own to illustrate an API. A snippet with no
  top-level imports is treated as an excerpt and only reported if it is
  invalid inside a coroutine too. One that imports its own dependencies is
  presenting itself as a file you can run, so top-level ``await`` in it is a
  defect.
- ``Agent`` sets ``__signature__``, which hides the ``**kwargs`` its
  ``__init__`` really accepts, so keywords are only checked when neither the
  class nor its ``__init__`` takes ``**kwargs``.

Mark a deliberate placeholder with ``<!-- docs: skip -->`` on the line above
the fence, or ``# docs: skip`` as the block's first line.

Usage::

    python scripts/check_doc_snippets.py            # README.md + examples/
    python scripts/check_doc_snippets.py README.md
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import re
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


FENCE = re.compile(r"^([ \t]*)```(?:python|py)\s*$(.*?)^\1```[ \t]*$", re.MULTILINE | re.DOTALL)
SKIP_INLINE = "# docs: skip"
SKIP_COMMENT = "<!-- docs: skip -->"

#: Checked by default: the two files a new reader actually copies from.
DEFAULT_TARGETS = ("README.md", "examples/README.md")


@dataclass(frozen=True)
class Problem:
    path: str
    block: int
    kind: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: block {self.block}: [{self.kind}] {self.message}"


def iter_blocks(path: pathlib.Path) -> Iterator[tuple[int, str]]:
    """Yield ``(index, source)`` for each non-skipped Python block."""
    text = path.read_text(errors="replace")
    for index, match in enumerate(FENCE.finditer(text), 1):
        indent, body = match.group(1), match.group(2)
        # The HTML-comment opt-out counts only on the line immediately above
        # the fence, so one skip can never silence the following block.
        if text[: match.start()].rstrip().endswith(SKIP_COMMENT):
            continue
        if indent:
            body = "\n".join(line.removeprefix(indent) for line in body.splitlines())
        if body.lstrip().startswith(SKIP_INLINE):
            continue
        yield index, body


def _compiles(source: str) -> str | None:
    """``None`` if ``source`` compiles as a module, else the error message."""
    try:
        compile(source, "<snippet>", "exec")
    except SyntaxError as exc:
        return str(exc).split("(")[0].strip()
    return None


def _is_excerpt(tree: ast.Module, source: str) -> bool:
    """Whether the snippet is an illustrative fragment rather than a file.

    Top-level imports are the discriminator: a snippet that imports its own
    dependencies is presenting itself as something you can save and run.
    """
    if any(isinstance(node, ast.Import | ast.ImportFrom) for node in tree.body):
        return False
    return _compiles(f"async def _fragment():\n{textwrap.indent(source, '    ')}\n") is None


def _resolve(dotted: str) -> Any:
    """Resolve a dotted name to an object, or ``None``."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:cut]))
        except Exception:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj
    return None


def _takes_var_keyword(obj: Any) -> bool:
    """Whether ``obj`` swallows arbitrary keywords.

    ``Agent`` sets ``__signature__``, advertising a curated parameter list
    that omits the ``**kwargs`` its ``__init__`` really accepts — so the
    ``__init__`` has to be consulted, but only when the class defines its own.
    Inheriting ``object.__init__`` would otherwise make every call look
    unverifiable.
    """
    candidates = [obj]
    if inspect.isclass(obj) and "__init__" in vars(obj):
        candidates.append(obj.__init__)
    for candidate in candidates:
        try:
            sig = inspect.signature(candidate)
        except (ValueError, TypeError):
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return True
    return False


def _check_call(node: ast.Call, target: str, obj: Any) -> list[tuple[str, str]]:
    """Validate one call's keywords against the real callable."""
    if _takes_var_keyword(obj):
        return []
    passed = {kw.arg for kw in node.keywords if kw.arg is not None}
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return []
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return [("kwarg", f"{target}(...) has no parameter {kw!r}") for kw in sorted(passed - accepted)]


def check_block(source: str) -> list[tuple[str, str]]:
    """Every problem in one snippet, as ``(kind, message)``."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [("syntax", str(exc).split("(")[0].strip())]

    if (error := _compiles(source)) is not None and not _is_excerpt(tree, source):
        return [("syntax", error)]

    found: list[tuple[str, str]] = []
    bound: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] != "tulip":
                continue
            try:
                importlib.import_module(node.module)
            except Exception as exc:
                found.append(("import", f"{node.module}: {type(exc).__name__}"))
                continue
            for alias in node.names:
                dotted = f"{node.module}.{alias.name}"
                if _resolve(dotted) is None:
                    found.append(("symbol", f"{dotted} does not exist"))
                else:
                    bound[alias.asname or alias.name] = dotted

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            target = bound.get(node.func.id)
            if target and (obj := _resolve(target)) is not None:
                found.extend(_check_call(node, node.func.id, obj))

    return found


def check(paths: list[pathlib.Path]) -> tuple[int, list[Problem]]:
    """Check every block in every path. Returns ``(blocks_checked, problems)``."""
    problems: list[Problem] = []
    checked = 0
    for path in paths:
        if not path.exists():
            continue
        for index, source in iter_blocks(path):
            checked += 1
            problems.extend(
                Problem(str(path), index, kind, message) for kind, message in check_block(source)
            )
    return checked, problems


def main(argv: list[str]) -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    targets = (
        [pathlib.Path(a) for a in argv[1:]]
        if len(argv) > 1
        else [root / t for t in DEFAULT_TARGETS]
    )

    checked, problems = check(targets)
    print(f"checked {checked} Python snippet(s) in {len(targets)} file(s)")
    if not problems:
        print("no problems found")
        return 0

    print(f"\n{len(problems)} problem(s):\n")
    for problem in problems:
        print(f"  {problem}")
    print(
        f"\nIf a snippet names something on purpose that does not exist yet, mark it "
        f"with {SKIP_COMMENT!r} on the line above the fence, or {SKIP_INLINE!r} as "
        f"its first line."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
