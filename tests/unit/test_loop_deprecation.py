# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus.loop`` is gone, and the helper that left it keeps working.

``locus.loop`` was a second ReAct implementation parallel to the one ``Agent``
runs. ``Agent`` never used it; the single reference from the production
runtime was one private helper, which moved to :mod:`locus.tools.executor`
when the package was deprecated. The package was removed in 6.0.0, per
`DEPRECATION.md`.

The tests that matter here are the ones that would catch a removal which
*looks* done and is not: a name still importable from the old path, a
supported module still reaching for the package, and the moved helper
changing what ``idempotent=True`` does.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pathlib
import sys

import pytest


SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "locus"


# --------------------------------------------------------------------------
# The names are gone
# --------------------------------------------------------------------------


def test_the_package_no_longer_imports() -> None:
    sys.modules.pop("locus.loop", None)
    # Looked up, not imported: importing would bind ``locus.loop`` on the
    # package for the rest of the session. A checkout that imported the
    # package before the removal keeps ``src/locus/loop/__pycache__`` (git
    # never deletes it), and Python then resolves the directory as an empty
    # namespace package, whose spec has no origin. That is not the package.
    spec = importlib.util.find_spec("locus.loop")
    assert spec is None or spec.origin is None, f"locus.loop still has source: {spec.origin}"
    assert not list((SRC / "loop").rglob("*.py"))


@pytest.mark.parametrize(
    "name",
    [
        "ReActLoop",
        "ReActLoopConfig",
        "create_react_loop",
        "LoopRunner",
        "BatchRunner",
        "StreamingCollector",
        "ThinkNode",
        "ExecuteNode",
        "ReflectNode",
        "ConditionalRouter",
    ],
)
def test_the_old_names_cannot_be_imported(name: str) -> None:
    with pytest.raises(ImportError):
        exec(f"from locus.loop import {name}")  # noqa: S102 — the import statement under test


@pytest.mark.parametrize("submodule", ["nodes", "react", "router", "runner"])
def test_the_submodules_are_gone_too(submodule: str) -> None:
    with pytest.raises(ImportError):
        importlib.import_module(f"locus.loop.{submodule}")


def test_the_top_level_package_has_no_loop_attribute() -> None:
    import locus

    # With a stale ``src/locus/loop/__pycache__`` around, an earlier test's
    # failed ``from locus.loop import …`` has bound the empty namespace
    # package on ``locus``; start from the package's own lookup.
    sys.modules.pop("locus.loop", None)
    locus.__dict__.pop("loop", None)
    with pytest.raises(AttributeError):
        _ = locus.loop


def test_no_source_file_remains_on_disk() -> None:
    # Not ``.exists()``: a stale ``__pycache__`` directory may outlive the sources.
    assert not list((SRC / "loop").rglob("*.py"))


def test_the_supported_runtime_does_not_import_the_removed_package() -> None:
    """A module still importing from the old path would fail at import time;
    this names the file instead of leaving a traceback to read."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            module = getattr(node, "module", None) if isinstance(node, ast.ImportFrom) else None
            if module and module.split(".")[:2] == ["locus", "loop"]:
                offenders.append(f"{path.relative_to(SRC.parent)}:{node.lineno} imports {module}")
            if isinstance(node, ast.Import):
                offenders.extend(
                    f"{path.relative_to(SRC.parent)}:{node.lineno} imports {alias.name}"
                    for alias in node.names
                    if alias.name.split(".")[:2] == ["locus", "loop"]
                )

    assert not offenders, "\n  ".join(["locus.loop was removed, yet:", *offenders])


# --------------------------------------------------------------------------
# The helper that moved out before the removal
# --------------------------------------------------------------------------


def test_the_helper_that_moved_still_behaves() -> None:
    """``idempotent=True`` is a documented feature; moving its guts must not
    change what it does."""
    from locus.agent.result import ToolExecution
    from locus.tools.executor import find_matching_execution

    class _State:
        tool_executions = (
            ToolExecution(tool_name="book", tool_call_id="1", arguments={"day": "mon"}),
            ToolExecution(tool_name="book", tool_call_id="2", arguments={"day": "tue"}),
        )

    state = _State()

    assert find_matching_execution(state, "book", {"day": "mon"}).tool_call_id == "1"
    # Different arguments are a miss, so a legitimate re-call still runs.
    assert find_matching_execution(state, "book", {"day": "wed"}) is None
    assert find_matching_execution(state, "cancel", {"day": "mon"}) is None


def test_arguments_that_cannot_be_compared_are_a_miss_not_a_crash() -> None:
    """Dedupe is an optimisation; it must never be the thing that fails a run.

    A prior execution whose arguments will not coerce to a dict cannot be
    shown equal to anything, so the honest answer is "no match" — the tool
    runs again. Raising here would turn a cache lookup into an outage.
    """
    from locus.tools.executor import find_matching_execution

    class _Uncomparable:
        tool_name = "book"
        arguments = object()  # dict() of this raises TypeError

    class _State:
        tool_executions = (_Uncomparable(),)

    assert find_matching_execution(_State(), "book", {"day": "mon"}) is None
