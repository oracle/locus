# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tool-result offloading: large results leave the conversation by handle.

The offload is exercised on its own (threshold, handle, preview, paging),
through the agent loop (the message the model reads carries the handle, the
``fetch_tool_result`` tool reads the text back), across a checkpoint round
trip, through compaction (a cleared stub keeps the handle) and in the coding
harness (``bash`` output stored whole is readable with ``fetch_tool_result``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from locus.agent import Agent
from locus.core.messages import Message, Role, ToolCall, ToolResult
from locus.harness import HarnessConfig, LocalBackend, build_harness
from locus.harness.toolset import DEFAULT_TOOLS
from locus.memory.backends import MemoryCheckpointer
from locus.memory.compaction import CLEARED_OUTPUT_KEY, CompactionTracker, ContextCompactor
from locus.memory.offload import OFFLOAD_MARKER, ToolResultOffload, offload_handle, page_of
from locus.testing import FunctionModel, text, tool_call
from locus.tools.decorator import tool
from locus.tools.result_storage import ToolResultStore


BIG = "".join(f"line {i:05d} of the big output\n" for i in range(1_000))  # ~30k chars


def _result(content: str, *, call_id: str = "c1", name: str = "read_file") -> ToolResult:
    return ToolResult(tool_call_id=call_id, name=name, content=content)


# --- the offload on its own ---------------------------------------------------


def test_results_under_the_threshold_are_untouched() -> None:
    offload = ToolResultOffload(threshold_chars=100)
    small = _result("x" * 100)

    assert offload.offload(small, run_id="r", iteration=1) is small
    assert not offload.should_offload("x" * 100)
    assert offload.should_offload("x" * 101)
    assert offload.preview_chars == 100, "the preview never exceeds the threshold"


def test_a_large_result_becomes_a_handle_and_a_preview() -> None:
    saved: dict[str, str] = {}
    store = ToolResultStore(save=saved.__setitem__, load=saved.get)
    offload = ToolResultOffload(store, threshold_chars=1_000, preview_chars=200)

    out = offload.offload(_result(BIG, call_id="call_7"), run_id="run-1", iteration=3)

    assert out.content is not None
    assert out.content.startswith(OFFLOAD_MARKER)
    handle = offload_handle(out)
    assert handle is not None
    assert saved[handle] == BIG
    assert "call_7" in handle, "the handle names the call, so parallel calls never collide"
    assert "fetch_tool_result" in out.content
    assert BIG[:200] in out.content
    assert len(out.content) < 600
    # Offloading an offloaded result is a no-op.
    assert offload.offload(out, run_id="run-1", iteration=3) is out
    assert offload_handle(Message.tool(out)) == handle
    assert offload_handle("plain text") is None
    assert offload_handle(None) is None


def test_fetch_pages_through_the_stored_text() -> None:
    offload = ToolResultOffload(threshold_chars=1_000, page_chars=10_000)
    out = offload.offload(_result(BIG), run_id="r", iteration=1)
    handle = offload_handle(out)
    assert handle is not None

    first = offload.fetch(handle)
    assert first.startswith(f"[fetch_tool_result handle={handle} offset=0 next_offset=10000 of ")
    assert first.split("\n", 1)[1] == BIG[:10_000]
    last = offload.fetch(handle, offset=25_000)
    assert "next_offset=end" in last
    assert last.split("\n", 1)[1] == BIG[25_000:]
    assert offload.fetch(handle, offset=10**9).split("\n", 1)[1] == ""
    with pytest.raises(KeyError):
        offload.fetch("locus:result:nothing")
    assert (
        page_of("abc", "h", offset=1, limit=1)
        == "[fetch_tool_result handle=h offset=1 next_offset=2 of 3 characters]\nb"
    )


async def test_the_fetch_tool_reads_back_and_names_an_unknown_handle() -> None:
    offload = ToolResultOffload(threshold_chars=1_000)
    out = offload.offload(_result(BIG), run_id="r", iteration=1)
    handle = offload_handle(out)
    fetch = offload.as_tool()

    assert fetch.name == "fetch_tool_result"
    page = await fetch.execute(handle=handle, offset=0, limit=500)
    assert BIG[:500] in str(page)
    missing = await fetch.execute(handle="locus:result:gone")
    assert "No stored result" in str(missing)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold_chars": 0},
        {"preview_chars": -1},
        {"page_chars": 10},
    ],
)
def test_invalid_settings_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ToolResultOffload(**kwargs)


# --- through the agent loop ---------------------------------------------------


@tool
def big_tool() -> str:
    """Return a large output."""
    return BIG


def _agent(offload: Any, handler: Any, **kwargs: Any) -> Agent:
    return Agent(
        model=FunctionModel(handler),
        tools=[big_tool],
        tool_result_offload=offload,
        reflexion=False,
        grounding=False,
        max_iterations=6,
        **kwargs,
    )


def test_the_agent_offloads_and_registers_the_fetch_tool() -> None:
    offload = ToolResultOffload(threshold_chars=1_000, preview_chars=100)
    fetched: list[str] = []

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        names = {t["function"]["name"] if "function" in t else t["name"] for t in tools}
        assert "fetch_tool_result" in names
        results = [m for m in messages if m.role == Role.TOOL]
        if not results:
            return tool_call("big_tool", call_id="b1")
        if len(results) == 1:
            handle = offload_handle(results[0])
            assert handle is not None
            return tool_call("fetch_tool_result", call_id="f1", handle=handle, offset=28_000)
        fetched.append(results[-1].content or "")
        return text("done")

    result = _agent(offload, handler).run_sync("read it")

    assert result.success
    tool_messages = [m for m in result.state.messages if m.role == Role.TOOL]
    assert tool_messages[0].content is not None
    assert tool_messages[0].content.startswith(OFFLOAD_MARKER)
    assert len(tool_messages[0].content) < 700, "the model saw a handle and a preview"
    assert fetched
    assert BIG[28_000:] in fetched[0]
    assert "line 00999" in fetched[0]
    assert not tool_messages[1].content.startswith(OFFLOAD_MARKER), (  # type: ignore[union-attr]
        "a fetched page is never offloaded again, whatever its size"
    )


def test_a_plain_true_turns_offloading_on_with_the_defaults() -> None:
    agent = _agent(True, lambda m, t: text("ok"))
    assert isinstance(agent.config.tool_result_offload, ToolResultOffload)
    assert agent.config.tool_result_offload.threshold_chars == 16_000
    assert "fetch_tool_result" in agent._tool_registry
    off = _agent(False, lambda m, t: text("ok"))
    assert off.config.tool_result_offload is None
    assert "fetch_tool_result" not in off._tool_registry


async def test_the_handle_survives_a_checkpoint_round_trip() -> None:
    offload = ToolResultOffload(threshold_chars=1_000, preview_chars=100)
    checkpointer = MemoryCheckpointer()

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        if any(m.role == Role.TOOL for m in messages):
            return text("done")
        return tool_call("big_tool", call_id="b1")

    agent = _agent(offload, handler, checkpointer=checkpointer)
    await agent.arun("read it", thread_id="t-1")

    loaded = await checkpointer.load("t-1")
    assert loaded is not None
    stored = next(m for m in loaded.messages if m.role == Role.TOOL)
    handle = offload_handle(stored)
    assert handle is not None, "the handle travelled in the message text"
    assert offload.load(handle) == BIG
    assert BIG[:100] in (stored.content or "")


# --- through compaction -------------------------------------------------------


async def test_a_cleared_stub_keeps_the_handle() -> None:
    offload = ToolResultOffload(threshold_chars=1_000, preview_chars=900)
    offloaded = offload.offload(_result(BIG, call_id="c0"), run_id="r", iteration=0)
    handle = offload_handle(offloaded)
    messages = [Message.system("SYSTEM"), Message.user("TASK")]
    for i in range(6):
        messages.append(
            Message.assistant(
                content=f"step {i}",
                tool_calls=[ToolCall(id=f"c{i}", name="read_file", arguments={"path": f"f{i}"})],
            )
        )
        messages.append(
            Message.tool(offloaded)
            if i == 0
            else Message(
                role=Role.TOOL, content="o" * 6_000, tool_call_id=f"c{i}", name="read_file"
            )
        )
    compactor = ContextCompactor(context_length=10_000)

    outcome = await compactor.compact(messages, iteration=6, tracker=CompactionTracker())

    assert outcome is not None
    assert outcome.stage == "prune"
    stub = next(m for m in outcome.messages if m.tool_call_id == "c0")
    assert stub.metadata.get(CLEARED_OUTPUT_KEY)
    assert handle is not None
    assert handle in (stub.content or "")
    assert "fetch_tool_result" in (stub.content or "")
    assert offload_handle(stub) is None, "a stub is not itself an offloaded result"


# --- in the coding harness ----------------------------------------------------


async def test_the_harness_fetches_a_stored_command_output_back(tmp_path: Path) -> None:
    saved: dict[str, str] = {}
    store = ToolResultStore(
        save=saved.__setitem__, load=saved.get, threshold_chars=100, preview_chars=50
    )
    harness = build_harness(
        LocalBackend(root=str(tmp_path)),
        config=HarnessConfig(result_store=store, output_chars=300, output_head_chars=100),
    )

    assert [t.name for t in harness.tools][-1] == "fetch_tool_result"
    assert "fetch_tool_result" in harness.prompt_fragment
    shown = str(await harness.tool("bash").execute(command="seq 1 2000"))
    assert "fetch_tool_result(key, offset) reads it back" in shown
    (key,) = saved
    assert f"key={key}" in shown

    page = str(await harness.tool("fetch_tool_result").execute(key=key, offset=0, limit=1_000))
    assert page.startswith(f"[fetch_tool_result handle={key} offset=0 next_offset=1000")
    assert page.split("\n", 1)[1] == saved[key][:1_000]
    tail = str(await harness.tool("fetch_tool_result").execute(key=key, offset=8_800))
    assert tail.rstrip().endswith("2000")
    missing = str(await harness.tool("fetch_tool_result").execute(key="locus:result:nope"))
    assert "No stored output" in missing


def test_the_harness_leaves_the_fetch_tool_out_without_a_store(tmp_path: Path) -> None:
    harness = build_harness(LocalBackend(root=str(tmp_path)))
    assert [t.name for t in harness.tools] == list(DEFAULT_TOOLS)
    assert "fetch_tool_result" not in DEFAULT_TOOLS
    explicit = build_harness(LocalBackend(root=str(tmp_path)), tools=["fetch_tool_result"])
    assert [t.name for t in explicit.tools] == ["fetch_tool_result"]
