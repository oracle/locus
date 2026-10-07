# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A workspace read is not repeatable: the agent loop must run it again after a write.

The loop answers a repeated call to an ``idempotent`` tool from the run's earlier
result without running it. Live on dev (functional F23) a subagent read a file,
edited it, read it again and was shown the file as it was before its own edit.
"""

from __future__ import annotations

from pathlib import Path

from tulip import Agent
from tulip.harness import LocalBackend, build_harness
from tulip.harness.ledger import ReadLedger
from tulip.harness.tools import FACTORIES
from tulip.harness.tools.common import HarnessConfig, HarnessContext
from tulip.testing import ScriptedModel, text, tool_call


def test_no_harness_tool_is_idempotent(tmp_path: Path) -> None:
    context = HarnessContext(
        backend=LocalBackend(root=str(tmp_path)),
        config=HarnessConfig(),
        ledger=ReadLedger(require_read=True),
    )
    cached = [
        name for name, make in FACTORIES.items() if getattr(make(context), "idempotent", False)
    ]
    assert cached == []


def test_a_read_after_an_edit_shows_the_edit(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("parent was here\n")
    harness = build_harness(LocalBackend(root=str(tmp_path)), tools=["read", "edit"])
    model = ScriptedModel(
        [
            tool_call("read", call_id="r1", path="notes.txt"),
            tool_call(
                "edit", call_id="e1", path="notes.txt", old="parent was here", new="child was here"
            ),
            tool_call("read", call_id="r2", path="notes.txt"),
            text("done"),
        ]
    )
    agent = Agent(model=model, tools=harness.tools, max_iterations=6)
    result = agent.run_sync("go")

    reads = [e for e in result.state.tool_executions if e.tool_name == "read"]
    assert len(reads) == 2
    assert reads[1].idempotent_cache_hit is False
    assert "child was here" in str(reads[1].result)
    assert (tmp_path / "notes.txt").read_text() == "child was here\n"
