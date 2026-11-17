# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus.loop`` on its way out, and the deprecation machinery finally used.

``locus.loop`` is a second ReAct implementation parallel to the one ``Agent``
runs. ``Agent`` never used it — the single reference from the production
runtime was one private helper, which has moved. Two implementations of the
same idea is worse than either alone: they drift, and a bug fixed in one stays
live in the other.

It is public API on a SemVer 2.x line, though, with sixteen exports and its own
reference page, so it gets deprecated rather than deleted. That makes this the
first exercise of ``LocusDeprecationWarning`` — the policy in `DEPRECATION.md`
was documented in two files and had zero call sites, so nothing had ever proved
a Locus deprecation reaches a consumer at all.

The tests that matter here are the ones that would catch a deprecation which
*looks* done and is not: a warning that never fires because the name is still
eagerly bound, and a removal that cannot happen because the supported runtime
still imports from the package being removed.
"""

from __future__ import annotations

import ast
import pathlib
import warnings

import pytest

from locus.core.warnings import LocusDeprecationWarning


SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "locus"


# --------------------------------------------------------------------------
# The warning actually fires
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "instead"),
    [
        ("ReActLoop", "locus.agent.Agent"),
        ("create_react_loop", "locus.agent.Agent"),
        ("LoopRunner", "Agent.arun()"),
        ("BatchRunner", "locus.evaluation.EvalRunner"),
        ("StreamingCollector", "Agent.run()"),
    ],
)
def test_using_a_deprecated_name_warns_and_says_what_to_use(name: str, instead: str) -> None:
    import locus.loop

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = getattr(locus.loop, name)

    assert len(caught) == 1
    assert issubclass(caught[0].category, LocusDeprecationWarning)
    message = str(caught[0].message)
    assert "6.0.0" in message, "a deprecation without a removal version is a rumour"
    assert instead in message


def test_a_name_with_no_clean_replacement_does_not_invent_one() -> None:
    """A vague pointer is worse than none.

    ``ThinkNode`` has no single successor — the phases are internal to
    ``Agent`` — so the warning says it is going and stops there, rather than
    sending the reader to check something that was never going to fit.
    """
    import locus.loop

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = locus.loop.ThinkNode

    assert "; use " not in str(caught[0].message)


def test_every_export_warns() -> None:
    """A partial deprecation is the worst kind: some callers never hear."""
    import locus.loop

    unwarned = []
    for name in locus.loop.__all__:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = getattr(locus.loop, name)
        if not caught:
            unwarned.append(name)

    assert not unwarned, f"these are exported but never warn: {unwarned}"


def test_the_names_still_work() -> None:
    """Deprecated is not removed. Breaking now would skip the whole policy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LocusDeprecationWarning)
        from locus.loop import BatchRunner, LoopRunner, ReActLoop

    assert all(callable(obj) for obj in (ReActLoop, LoopRunner, BatchRunner))


def test_an_unknown_attribute_still_raises() -> None:
    """``__getattr__`` must not turn every typo into a deprecation notice."""
    import locus.loop

    with pytest.raises(AttributeError, match="NoSuchThing"):
        _ = locus.loop.NoSuchThing


def test_dir_still_lists_the_module() -> None:
    """Serving names through ``__getattr__`` otherwise breaks tab-completion."""
    import locus.loop

    assert set(dir(locus.loop)) == set(locus.loop.__all__)


# --------------------------------------------------------------------------
# The removal has to be possible
# --------------------------------------------------------------------------


def test_the_supported_runtime_does_not_import_the_deprecated_package() -> None:
    """A supported runtime reaching into a package scheduled for removal is a
    removal that cannot happen.

    This is the check that keeps 6.0.0 achievable, and the one most likely to
    be quietly undone by a later change that "just needs one thing from there".
    """
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if path.is_relative_to(SRC / "loop"):
            continue
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

    assert not offenders, "\n  ".join(["locus.loop cannot be removed while:", *offenders])


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


def test_the_old_private_name_still_resolves() -> None:
    """Anything importing the private helper from its old home keeps working."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LocusDeprecationWarning)
        from locus.loop.nodes import _find_matching_execution

    from locus.tools.executor import find_matching_execution

    assert _find_matching_execution is find_matching_execution
