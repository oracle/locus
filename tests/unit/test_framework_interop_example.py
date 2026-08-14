# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The framework-interop example, which backs the project's headline claim.

The homepage says you can "govern the agents you already run in LangChain,
CrewAI, or the OpenAI Agents SDK". `examples/notebook_88_framework_interop.py`
is the only runnable proof of that, so it gets tests — and they cover both
halves of what the file has to do.

Most CI runs will not have LangChain installed, so the test that matters there
is the *skip* path: an example that tracebacks when an optional dependency is
missing reads as a broken project rather than as one install away, and that is
how a reader decides the claim was hollow. It must exit 0 and say what to
install.

Where the bridges are present, the second test runs the real thing and reads
the ledger: the ungated agent pays $4,000,000, the gated one pays nothing, and
nothing about the agent changed in between.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest


EXAMPLE = (
    pathlib.Path(__file__).resolve().parents[2] / "examples" / "notebook_88_framework_interop.py"
)


def _run() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_the_example_exists_where_the_docs_point() -> None:
    assert EXAMPLE.is_file()


def test_it_exits_cleanly_either_way() -> None:
    """Whether or not the bridges are installed, this must never traceback."""
    result = _run()

    assert result.returncode == 0, result.stderr[-2000:]
    assert "Traceback" not in result.stderr


def test_a_missing_bridge_is_reported_rather_than_raised() -> None:
    try:
        import langchain_core  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("the bridges are installed; the skip path cannot be observed here")

    out = _run().stdout
    assert "skipped" in out
    assert "pip install" in out
    assert "tulip-frameworks" in out


def test_the_gate_holds_a_four_million_dollar_refund() -> None:
    """The claim itself, run for real. Needs the optional bridges."""
    pytest.importorskip("langchain_core", reason="pip install tulip-frameworks[langchain]")
    pytest.importorskip("langgraph", reason="pip install tulip-frameworks[langgraph]")
    pytest.importorskip("tulip_frameworks", reason="pip install tulip-frameworks")

    result = _run()
    assert result.returncode == 0, result.stderr[-2000:]

    # Run 1 — the agent as it exists today. The money moves.
    assert "4,000,000.00 paid out" in result.stdout
    # Run 2 — one tool wrapped, nothing else touched. It does not.
    assert "0.00 paid out" in result.stdout
    assert "blast radius 4000 exceeds the maximum 1" in result.stdout
    assert "chain intact: ✓" in result.stdout
