# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tiered compaction and tool-result offloading on a live OCI model.

Two runs on ``oci:openai.gpt-4.1`` through the ``oci:`` factory (``OCI_PROFILE``,
``OCI_REGION`` and ``OCI_COMPARTMENT_ID`` from the environment; skipped when
the profile is not set):

* An agent with a small configured window reads a dozen large records
  through a tool. The whole log is several times the window, so without
  compaction the run would overflow it; with ``TieredCompaction`` it
  completes, every compaction tier reports tokens going down, and the final
  answer still carries the code word planted in the first record, which
  only the compacted log can have kept. That record is also the one large
  enough for ``ToolResultOffload`` to store, so the code word reaches the
  answer through the preview, the summary or a ``fetch_tool_result`` call.
* A coding-harness session runs a command whose output is far too long to
  show; the harness stores it whole, the model answers from what it can see
  or fetch, and the run completes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from locus.agent import Agent
from locus.core.events import CompactionEvent, TerminateEvent
from locus.harness import HarnessConfig, LocalBackend, build_harness
from locus.memory import (
    ClearToolResults,
    Relevance,
    SlidingWindow,
    Summarize,
    TieredCompaction,
    ToolResultOffload,
)
from locus.tools.decorator import tool
from locus.tools.result_storage import ToolResultStore


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("OCI_PROFILE"), reason="OCI not configured (OCI_PROFILE)"),
]

MODEL_ID = "openai.gpt-4.1"
CODE_WORD = "AMBER-FALCON"
RECORDS = 12
WINDOW = 12_000


def _model() -> Any:
    from locus.models import get_model

    kwargs: dict[str, Any] = {
        "profile": os.environ["OCI_PROFILE"],
        "region": os.getenv("OCI_REGION", "us-chicago-1"),
        "max_tokens": 1_024,
    }
    compartment = os.getenv("OCI_COMPARTMENT") or os.getenv("OCI_COMPARTMENT_ID")
    if compartment:
        kwargs["compartment_id"] = compartment
    return get_model(f"oci:{MODEL_ID}", **kwargs)


def _record(n: int) -> str:
    head = f"RECORD {n} of {RECORDS}\n"
    if n == 1:
        head += f"code word: {CODE_WORD}\n"
    # Record 1 alone is over the offload threshold; the rest stay inline.
    rows = 200 if n == 1 else 60
    body = "".join(f"row {n}-{i:03d}: " + "measurement " * 6 + "\n" for i in range(rows))
    tail = "end of record; call next_record for the next one.\n"
    if n == RECORDS:
        tail = "end of record; THIS WAS THE LAST RECORD.\n"
    return head + body + tail


async def test_a_run_that_overflows_a_small_window_completes_and_keeps_the_fact() -> None:
    served: list[int] = []

    @tool
    def next_record() -> str:
        """Return the next record of the log; says when it was the last one."""
        n = len(served) + 1
        if n > RECORDS:
            return "No records left: you have read them all."
        served.append(n)
        return _record(n)

    agent = Agent(
        model=_model(),
        tools=[next_record],
        system_prompt=(
            "You are a careful analyst. Use the tools; do not guess. When the task is "
            "done, answer in one short paragraph."
        ),
        conversation_manager=TieredCompaction(
            [
                ClearToolResults(0.5, keep_tokens=1_500),
                SlidingWindow(0.7, keep_turns=6),
                Summarize(0.8),
                Relevance(0.95, keep_turns=4),
            ]
        ),
        tool_result_offload=ToolResultOffload(threshold_chars=8_000, preview_chars=600),
        context_window=WINDOW,
        reflexion=False,
        grounding=False,
        max_iterations=40,
    )

    events = [
        e
        async for e in agent.run(
            "Read every record of the log, one next_record call at a time, until the "
            "tool says it was the last record. Then report (1) how many records there "
            "were and (2) the code word stated in record 1, exactly as written. If the "
            "code word is no longer in front of you, use fetch_tool_result on record 1's "
            "handle to read it again rather than guessing."
        )
    ]

    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert terminate.reason == "complete", terminate
    assert len(served) == RECORDS, "every record was read"
    answer = terminate.final_message or ""
    assert CODE_WORD in answer, answer
    assert str(RECORDS) in answer, answer

    whole_log = sum(len(_record(n)) for n in range(1, RECORDS + 1)) // 4
    assert whole_log > WINDOW, "without compaction the log alone overflows the window"
    compactions = [e for e in events if isinstance(e, CompactionEvent)]
    assert compactions, "the window was reached and compacted"
    assert len(_record(1)) > 8_000, "record 1 was offloaded"
    assert all(len(_record(n)) < 8_000 for n in range(2, RECORDS + 1)), "the rest stayed inline"
    assert all(e.tier is not None for e in compactions)
    assert all(e.tokens_after < e.tokens_before for e in compactions)
    assert all(e.tokens_after < WINDOW for e in compactions)
    assert not any(e.exhausted for e in compactions)


async def test_the_coding_harness_completes_with_a_large_command_output(tmp_path: Path) -> None:
    saved: dict[str, str] = {}
    store = ToolResultStore(
        save=saved.__setitem__, load=saved.get, threshold_chars=1_000, preview_chars=200
    )
    harness = build_harness(
        LocalBackend(root=str(tmp_path)),
        tools=["bash", "fetch_tool_result"],
        config=HarnessConfig(result_store=store, output_chars=2_000, output_head_chars=600),
    )
    agent = Agent(
        model=_model(),
        tools=harness.tools,
        system_prompt=(
            "You are a coding agent in a shell workspace. Use bash for commands.\n"
            + harness.prompt_fragment
        ),
        conversation_manager=TieredCompaction(),
        context_window=WINDOW,
        reflexion=False,
        grounding=False,
        max_iterations=12,
    )

    events = [
        e
        async for e in agent.run(
            "Run exactly this command once: seq 1 20000 | awk '{print \"line\", $1}'. "
            "Then tell me how many lines it printed and what the last line was. "
            "If the output shown to you was cut, use fetch_tool_result with the key "
            "from the note to read the part you need."
        )
    ]

    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert terminate.reason == "complete", terminate
    assert saved, "the whole output was stored under a key"
    assert any(text.rstrip().endswith("line 20000") for text in saved.values())
    answer = terminate.final_message or ""
    assert "20000" in answer.replace(",", ""), answer
