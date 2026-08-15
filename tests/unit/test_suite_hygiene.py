# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Guards against test code that looks like coverage and is not.

A test suite is the thing everything else is measured against, so a defect in
it is invisible by construction — nothing is checking the checker. Two such
defects were found while chasing #116, and both are the same shape: code that
reads as a test and never runs.

**Shadowed definitions.** Python keeps the last binding, so a class or function
defined twice in one module silently discards the earlier one. `test_graph.py`
had two `TestEdge` and two `TestConditionalEdge`; `test_rag_multimodal.py` had
two `TestPDFProcessor`; `test_agent_integration.py` had three byte-identical
copies of `TestHooksE2E`. Fourteen test definitions in total that pytest never
collected, sitting in the files a reviewer reads to decide whether something is
tested.

They cost nothing in coverage, as it happens — every one duplicated a test that
did run. That is luck, not design: the same mistake against a case with no twin
removes real coverage and leaves the file looking as thorough as before.

**Drifted duplicates.** The other half of #116: `SlidingWindowManager` had two
separate suites, in `test_memory.py` and `test_conversation.py`. The commit
that added `preserve_first_user` updated one of them, so the two copies
disagreed about the class's contract while both stayed green — and a failure
reported as `TestSlidingWindowManager::test_preserves_system_message` did not
say which file it came from. That ambiguity is most of why the failure was
read as flakiness.

Duplicate *names across* modules are not policed here: 83 of them exist and
most are fine, because `TestConfig` in `test_memory_backends_mysql.py` and in
`test_memory_backends_postgresql.py` are scoped by their subject. Only
same-module shadowing is unambiguously a bug, so only that is a hard failure.
"""

from __future__ import annotations

import ast
import collections
import pathlib


TESTS = pathlib.Path(__file__).resolve().parents[1]


def _definitions(tree: ast.Module) -> collections.Counter[str]:
    """Top-level names that pytest would collect, counted by how often bound."""
    names: collections.Counter[str] = collections.Counter()
    for node in tree.body:
        collected = (isinstance(node, ast.ClassDef) and node.name.startswith("Test")) or (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name.startswith("test")
        )
        if collected:
            names[node.name] += 1
    return names


def _methods(node: ast.ClassDef) -> collections.Counter[str]:
    return collections.Counter(
        child.name
        for child in node.body
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
        and child.name.startswith("test")
    )


def _test_modules() -> list[pathlib.Path]:
    return sorted(TESTS.rglob("test_*.py"))


def test_the_extractor_actually_finds_the_suite() -> None:
    """A guard that silently matches nothing passes for the wrong reason."""
    modules = _test_modules()
    assert len(modules) > 100, f"only found {len(modules)} test modules — bad glob?"


def test_no_test_is_shadowed_by_a_later_definition_in_its_own_module() -> None:
    """Redefining a name discards the first one, without a word from anybody."""
    offenders = []
    for path in _test_modules():
        counts = _definitions(ast.parse(path.read_text()))
        for name, count in counts.items():
            if count > 1:
                offenders.append(f"{path.relative_to(TESTS.parent)}: {name} defined {count}x")

    assert not offenders, (
        "these definitions are dead — Python keeps the last binding, so the "
        "earlier ones never run:\n  " + "\n  ".join(offenders)
    )


def test_no_test_method_is_shadowed_within_its_class() -> None:
    """The same trap one level down, and harder to spot by eye."""
    offenders = []
    for path in _test_modules():
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            for name, count in _methods(node).items():
                if count > 1:
                    offenders.append(
                        f"{path.relative_to(TESTS.parent)}: {node.name}.{name} defined {count}x"
                    )

    assert not offenders, "shadowed test methods never run:\n  " + "\n  ".join(offenders)


def test_sliding_window_has_exactly_one_suite() -> None:
    """The #116 regression itself.

    Two suites for one class drifted apart once already: the commit that added
    ``preserve_first_user`` taught only one of them the new contract, and a
    failure reported by class name did not say which file it came from.
    """
    homes = [
        path.relative_to(TESTS.parent)
        for path in _test_modules()
        if any(
            isinstance(node, ast.ClassDef) and node.name == "TestSlidingWindowManager"
            for node in ast.parse(path.read_text()).body
        )
    ]

    assert len(homes) == 1, f"TestSlidingWindowManager should have one home, found {homes}"
