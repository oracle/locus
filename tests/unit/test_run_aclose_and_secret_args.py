# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``Agent.run()`` closes promptly and hook-injected secrets stay ephemeral.

* ``aclose()`` on the generator ``Agent.run()`` returns must close the run it
  drives right away (its ``finally`` runs before ``aclose()`` returns), not
  whenever the garbage collector gets to the inner generator.
* ``Agent.run`` is typed as an ``AsyncGenerator`` so ``contextlib.aclosing``
  accepts it without a cast.
* ``BeforeToolCallEvent.secret_arguments`` reach the tool call but never the
  checkpoint, the conversation or the event stream.
"""

from __future__ import annotations

import contextlib
import gc
import typing
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from tulip.agent import Agent
from tulip.core.events import ToolStartEvent
from tulip.hooks.provider import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookPriority,
    HookProvider,
)
from tulip.memory.backends.file import FileCheckpointer
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


CONFIRM = "confirm-9d2e71-SECRET"


# ---------------------------------------------------------------------------
# 3. aclose() closes the inner run
# ---------------------------------------------------------------------------


async def test_aclose_on_run_closes_the_inner_run_immediately() -> None:
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    agent = Agent(
        model=ScriptedModel([tool_call("ping", call_id="p1"), text("done")]),
        tools=[ping],
        reflexion=False,
        grounding=False,
    )
    gc.disable()  # prove it is not the collector doing the cleanup
    try:
        stream = agent.run("go")
        async for event in stream:
            if isinstance(event, ToolStartEvent):
                break
        assert agent._active_runs, "the run should be in flight"
        await stream.aclose()
        # The inner run's ``finally`` (which unregisters the run) has already
        # executed — no event-loop turn, no GC needed.
        assert agent._active_runs == {}
    finally:
        gc.enable()


async def test_aclosing_closes_the_run_on_early_exit() -> None:
    agent = Agent(model=ScriptedModel([text("hi")]), reflexion=False, grounding=False)
    async with contextlib.aclosing(agent.run("go")) as stream:
        async for _event in stream:
            break
    assert agent._active_runs == {}


# ---------------------------------------------------------------------------
# 7. Agent.run is an AsyncGenerator
# ---------------------------------------------------------------------------


def test_run_is_typed_as_an_async_generator() -> None:
    for method in (Agent.run, Agent.resume):
        hint = typing.get_type_hints(method)["return"]
        assert typing.get_origin(hint) is AsyncGenerator, method


# ---------------------------------------------------------------------------
# 8. Hook-injected secret arguments are ephemeral
# ---------------------------------------------------------------------------


class _ConfirmTokenHook(HookProvider):
    def __init__(self) -> None:
        self.after_arguments: list[dict[str, Any]] = []

    @property
    def priority(self) -> int:
        return HookPriority.BUSINESS_DEFAULT

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        if event.tool_name == "book":
            event.secret_arguments = {"confirm_token": CONFIRM}

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        self.after_arguments.append(dict(event.arguments))


async def test_secret_arguments_reach_the_tool_but_are_never_persisted(tmp_path: Path) -> None:
    received: list[dict[str, Any]] = []

    @tool
    def book(hotel: str, confirm_token: str = "") -> str:
        """Book a hotel."""
        received.append({"hotel": hotel, "confirm_token": confirm_token})
        return f"booked {hotel}"

    hook = _ConfirmTokenHook()
    checkpointer = FileCheckpointer(tmp_path / "cp")
    agent = Agent(
        model=ScriptedModel([tool_call("book", call_id="b1", hotel="Aman"), text("booked")]),
        tools=[book],
        hooks=[hook],
        checkpointer=checkpointer,
        reflexion=False,
        grounding=False,
    )
    events = [e async for e in agent.run("book it", thread_id="t-secret")]

    # The tool ran with the secret merged in.
    assert received == [{"hotel": "Aman", "confirm_token": CONFIRM}]

    # Nothing at rest carries it.
    saved = b"".join(p.read_bytes() for p in (tmp_path / "cp").rglob("*") if p.is_file())
    assert saved
    assert CONFIRM.encode() not in saved
    state = await checkpointer.load("t-secret")
    assert state is not None
    [execution] = state.tool_executions
    assert execution.arguments == {"hotel": "Aman"}

    # Nor does the event stream, nor observers of the persisted arguments.
    for event in events:
        assert CONFIRM not in event.model_dump_json()
    assert hook.after_arguments == [{"hotel": "Aman"}]


def test_secret_arguments_are_redacted_from_the_event_repr() -> None:
    event = BeforeToolCallEvent("book", "b1", {"hotel": "Aman"})
    event.secret_arguments = {"confirm_token": CONFIRM}
    shown = repr(event)
    assert CONFIRM not in shown
    assert "confirm_token" in shown
    assert "Aman" in shown
