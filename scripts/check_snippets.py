#!/usr/bin/env python
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Check every Python snippet in the docs against the installed SDK.

`mkdocs build --strict` proves links and mkdocstrings references resolve.
It never looks inside a fenced code block, so a snippet can name a module
that moved, a keyword argument that was renamed, or a required field that
was added, and the docs still build green. This closes that gap.

Three checks, in increasing order of what they catch:

1. **Syntax** — every ```python block must parse.
2. **Symbols** — every ``from locus... import X`` must resolve against the
   *installed* SDK.
3. **Call signatures** — keyword arguments passed to an imported Locus
   callable must exist, and required Pydantic fields must be supplied.

Two conventions decide what "a snippet must compile" means. ``README.md`` and
``examples/README.md`` show code that is pasted into a file, so a block there
must compile as a file. The pages under ``docs/`` follow the notebook
convention: a block may use top-level ``await``, ``async for`` and ``async
with`` and name tools or agents it does not define, as a notebook cell does; it
is accepted when it compiles inside a coroutine, and the symbol and keyword
checks still apply. With no arguments the checker covers all three.

Deliberately static: it imports the SDK but never executes a snippet, so
it needs no API keys, no network, and no running services. That keeps it
cheap enough to gate every PR.

Two things it is careful about, both learned from false positives:

- ``Agent`` sets ``__signature__``, which hides its ``**kwargs``. A class
  is only checked for unknown keywords when *neither* the class nor its
  ``__init__`` accepts ``**kwargs``.
- Tutorials legitimately contain names that do not exist yet (the "add a
  new event" page names ``EV_FOO_BAR`` on purpose). Put ``# docs: skip``
  as the first line of a block, or ``<!-- docs: skip -->`` on the line
  before the fence, to exempt it.

Usage::

    python scripts/check_snippets.py           # checks docs/
    python scripts/check_snippets.py docs/concepts
"""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import pathlib
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Any


FENCE = re.compile(r"^([ \t]*)```(?:python|py)\s*$(.*?)^\1```[ \t]*$", re.MULTILINE | re.DOTALL)
SKIP_INLINE = "# docs: skip"
ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = ("README.md", "examples/README.md", "docs/**/*.md")
NOTEBOOK_STYLE_DIRS = ("docs",)
SKIP_COMMENT = "<!-- docs: skip -->"


@dataclass(frozen=True)
class Problem:
    path: str
    block: int
    kind: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: block {self.block}: [{self.kind}] {self.message}"


def iter_blocks(path: pathlib.Path):
    """Yield ``(index, source)`` for each non-skipped Python block."""
    text = path.read_text(errors="replace")
    for index, match in enumerate(FENCE.finditer(text), 1):
        indent, body = match.group(1), match.group(2)
        # The HTML-comment opt-out only counts on the line immediately
        # above the fence, so one skip can never silence the next block.
        before = text[: match.start()].rstrip()
        if before.endswith(SKIP_COMMENT):
            continue
        if indent:
            body = "\n".join(line.removeprefix(indent) for line in body.splitlines())
        if body.lstrip().startswith(SKIP_INLINE):
            continue
        yield index, body


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


def _accepts_var_keyword(obj: Any) -> bool:
    """Whether ``obj`` swallows arbitrary keywords.

    A class that sets ``__signature__`` (``Agent`` does) advertises a
    curated parameter list that omits the ``**kwargs`` its ``__init__``
    really accepts, so the ``__init__`` has to be consulted too — but
    only when the class *defines its own*. Inheriting ``object.__init__``
    (every plain function has one) or Pydantic's synthesised
    ``BaseModel.__init__(**data)`` would otherwise make every call in the
    docs look unverifiable.
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


def _pydantic_fields(obj: Any) -> tuple[set[str], set[str]] | None:
    """``(all_fields, required_fields)`` if ``obj`` is a Pydantic model."""
    fields = getattr(obj, "model_fields", None)
    if not isinstance(fields, dict):
        return None
    names = set(fields)
    required = {n for n, f in fields.items() if getattr(f, "is_required", lambda: False)()}
    return names, required


def _check_call(node: ast.Call, target: str, obj: Any) -> list[tuple[str, str]]:
    """Validate one call's keywords against the real callable."""
    found: list[tuple[str, str]] = []
    passed = {kw.arg for kw in node.keywords if kw.arg is not None}
    splat = any(kw.arg is None for kw in node.keywords)

    # Checked before anything else: a class can be a Pydantic model *and*
    # define an ``__init__`` that accepts ``**kwargs`` (``Agent`` is both).
    # Its declared ``model_fields`` then describe none of what a caller
    # actually passes, so field-level checking would be pure noise.
    if _accepts_var_keyword(obj):
        return found

    model = _pydantic_fields(obj)
    if model is not None:
        names, required = model
        # A Pydantic model with extra="ignore" silently drops an unknown
        # keyword -- the snippet looks right and the object is wrong, which
        # is worse than an exception, so report it.
        for kw in sorted(passed - names):
            found.append(("kwarg", f"{target}(...) has no field {kw!r} (silently ignored)"))
        # Required-field checking only when the call is entirely keyword
        # driven: positional args and ``**spread`` both supply fields this
        # cannot see, and guessing there would produce false positives.
        if not splat and not node.args and passed:
            for miss in sorted(required - passed):
                found.append(("required", f"{target}(...) is missing required field {miss!r}"))
        return found

    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return found
    accepted = {
        n
        for n, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    for kw in sorted(passed - accepted):
        found.append(("kwarg", f"{target}(...) has no parameter {kw!r}"))
    return found


def _compiles(source: str) -> str | None:
    """``None`` if ``source`` compiles as a module, else the error message.

    ``ast.parse`` is more permissive than the compiler — it builds a tree for
    top-level ``await`` quite happily, and only ``compile()`` rejects it. That
    gap is not theoretical: it is how the quickstart shipped a headline snippet
    that raised ``SyntaxError`` for every reader who pasted it, while this
    checker ran over that file and reported no problems.
    """
    try:
        compile(source, "<snippet>", "exec")
    except SyntaxError as exc:
        return str(exc).split("(")[0].strip()
    return None


def _bound_names(tree: ast.Module) -> set[str]:
    """Every name the snippet defines for itself."""
    bound: set[str] = set(dir(builtins)) | {"__name__", "__file__", "__doc__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            args = node.args
            bound |= {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
            bound |= {a.arg for a in (args.vararg, args.kwarg) if a}
        elif isinstance(node, ast.Lambda):
            args = node.args
            bound |= {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
            bound |= {a.arg for a in (args.vararg, args.kwarg) if a}
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            # Covers assignment, for-targets, walrus and comprehension targets.
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound |= set(node.names)
    return bound


def _compiles_in_a_coroutine(source: str) -> bool:
    return _compiles(f"async def _fragment():\n{textwrap.indent(source, '    ')}\n") is None


def is_notebook_style(path: pathlib.Path) -> bool:
    """Whether ``path`` is a docs page, where the notebook convention holds."""
    try:
        parts = path.resolve().relative_to(ROOT).parts
    except ValueError:
        parts = path.parts
    return bool(parts) and parts[0] in NOTEBOOK_STYLE_DIRS


def _is_excerpt(tree: ast.Module, source: str) -> bool:
    """Whether this is an illustrative fragment rather than a runnable file.

    The docs are full of snippets that show a shape — an ``Orchestrator`` wired
    to ``issue_refund`` and ``flag_transaction``, neither of them defined
    anywhere in the block. Those were never pasteable, with or without the
    ``await``, and rewriting them into ``asyncio.run(main())`` would add
    ceremony to the one thing they exist to communicate.

    So the discriminator is whether the snippet resolves: if it loads a name it
    never binds, it is a fragment and its top-level ``await`` is a stylistic
    choice. If every name is defined, a reader *can* paste it — and then
    top-level ``await`` is a ``SyntaxError`` they will actually hit. That is
    the exact shape of the quickstart defect this check exists for.

    (An earlier version used "has top-level imports" as the test, which these
    fragments also satisfy — it flagged 60 of them.)
    """
    if _compiles(f"async def _fragment():\n{textwrap.indent(source, '    ')}\n") is not None:
        # Broken inside a coroutine too, so it is wrong either way.
        return False
    used = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return bool(used - _bound_names(tree))


def check_block(source: str, *, notebook_style: bool = False) -> list[tuple[str, str]]:
    """Every problem in one snippet, as ``(kind, message)``."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [("syntax", str(exc).split("(")[0].strip())]

    # Parse is not enough — see ``_compiles``.
    if (error := _compiles(source)) is not None and not (
        _is_excerpt(tree, source) or (notebook_style and _compiles_in_a_coroutine(source))
    ):
        return [("syntax", error)]

    found: list[tuple[str, str]] = []
    bound: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] != "locus":
                continue
            try:
                module = importlib.import_module(node.module)
            except Exception as exc:
                found.append(("import", f"{node.module}: {type(exc).__name__}"))
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                if not hasattr(module, alias.name):
                    found.append(("symbol", f"{node.module}.{alias.name} does not exist"))
                else:
                    bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "locus":
                    continue
                try:
                    importlib.import_module(alias.name)
                except Exception as exc:
                    found.append(("import", f"{alias.name}: {type(exc).__name__}"))
                else:
                    # ``import a.b`` binds ``a``, not ``a.b`` — so a later
                    # ``a.b.f(...)`` resolves through the top-level name.
                    if alias.asname:
                        bound[alias.asname] = alias.name
                    else:
                        top = alias.name.split(".")[0]
                        bound[top] = top

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func, attrs = node.func, []
        while isinstance(func, ast.Attribute):
            attrs.append(func.attr)
            func = func.value
        if not isinstance(func, ast.Name) or func.id not in bound:
            continue
        target = ".".join([bound[func.id], *reversed(attrs)])
        obj = _resolve(target)
        if obj is None or not callable(obj):
            continue
        found.extend(_check_call(node, target, obj))

    return found


def _markdown_files(targets: list[pathlib.Path]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for target in targets:
        files.extend(sorted(target.rglob("*.md")) if target.is_dir() else [target])
    return files


def check(targets: pathlib.Path | list[pathlib.Path]) -> tuple[int, list[Problem]]:
    """Check every block in every Markdown file under ``targets`` (files or directories)."""
    paths = [targets] if isinstance(targets, pathlib.Path) else targets
    problems: list[Problem] = []
    parsed = 0
    for path in _markdown_files(paths):
        notebook_style = is_notebook_style(path)
        for index, source in iter_blocks(path):
            issues = check_block(source, notebook_style=notebook_style)
            if not any(kind == "syntax" for kind, _ in issues):
                parsed += 1
            problems += [Problem(str(path), index, kind, message) for kind, message in issues]
    return parsed, problems


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        targets = [pathlib.Path(a) for a in argv[1:]]
        for target in targets:
            if not target.exists():
                print(f"no such path: {target}", file=sys.stderr)
                return 2
    else:
        targets = [path for pattern in DEFAULT_TARGETS for path in sorted(ROOT.glob(pattern))]
    parsed, problems = check(targets)
    print(f"checked {parsed} Python snippets in {len(_markdown_files(targets))} file(s)")
    if not problems:
        print("no problems found")
        return 0
    print(f"\n{len(problems)} problem(s):\n")
    for problem in problems:
        print(f"  {problem}")
    print(
        "\nIf a snippet names something on purpose that does not exist yet "
        f"(a tutorial placeholder), mark it with {SKIP_COMMENT!r} on the line "
        f"above the fence, or {SKIP_INLINE!r} as its first line."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
