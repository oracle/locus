# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``a2a.task.processing`` is emitted each time a task turns ``working``."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from locus.a2a import A2AClient, A2AServer, Message, TextPart
from locus.agent import Agent
from locus.core.messages import Role
from locus.observability import run_context
from locus.observability.event_bus import get_event_bus
from locus.testing import FunctionModel, text, tool_call
from locus.tools.decorator import tool


@tool
def ask_region(question: str) -> str:
    """Pause the run for a person: the interrupt marker any tool may return."""
    return json.dumps({"__interrupt__": True, "question": question, "options": None})


def _agent() -> Agent:
    def handler(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        answers = [m.content for m in messages if m.role == Role.TOOL]
        if answers:
            return text(f"done {answers[-1]}")
        return tool_call("ask_region", question="Which region?")

    return Agent(model=FunctionModel(handler), tools=[ask_region], reflexion=False, grounding=False)


def _plain_agent() -> Agent:
    return Agent(
        model=FunctionModel(lambda messages, tools: text("hello")),
        reflexion=False,
        grounding=False,
    )


def _message(content: str, *, task_id: str | None = None) -> Message:
    return Message(
        role="user",
        parts=[TextPart(text=content)],
        messageId=f"m-{content[:8]}",
        taskId=task_id,
    )


class _Observed:
    """Run ``body`` under a run id and collect the ``a2a.*`` events a subscriber saw."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def run(self, server: A2AServer, body: Any) -> Any:
        bus = get_event_bus()
        async with run_context() as rid:

            async def collect() -> None:
                async for event in bus.subscribe(rid):
                    if event.event_type.startswith("a2a."):
                        self.events.append((event.event_type, dict(event.data)))

            collector = asyncio.create_task(collect())
            await asyncio.sleep(0)
            transport = httpx.ASGITransport(app=server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
                client = A2AClient("http://testserver", http_client=http)
                result = await body(client)
            await bus.close_stream(rid)
            await collector
        return result

    def named(self, name: str) -> list[dict[str, Any]]:
        return [data for event, data in self.events if event == name]


async def test_processing_sits_between_received_and_completed() -> None:
    server = A2AServer(agent=_plain_agent(), allow_unauthenticated=True)
    seen = _Observed()

    async def body(client: A2AClient) -> Any:
        return await client.send_message(_message("hi"))

    task = await seen.run(server, body)

    names = [
        n
        for n, _ in seen.events
        if n in {"a2a.task.received", "a2a.task.processing", "a2a.task.completed"}
    ]
    assert names == ["a2a.task.received", "a2a.task.processing", "a2a.task.completed"]
    (processing,) = seen.named("a2a.task.processing")
    assert processing["task_id"] == task.id
    assert processing["context_id"] == task.contextId
    assert processing["resumed"] is False


async def test_a_resumed_task_emits_processing_again_with_resumed_true() -> None:
    server = A2AServer(agent=_agent(), allow_unauthenticated=True)
    seen = _Observed()

    async def body(client: A2AClient) -> Any:
        paused = await client.send_message(_message("deploy"))
        return paused, await client.send_message(_message("eu-frankfurt-1", task_id=paused.id))

    paused, resumed = await seen.run(server, body)

    assert resumed.id == paused.id
    processing = seen.named("a2a.task.processing")
    assert [p["resumed"] for p in processing] == [False, True]
    assert {p["task_id"] for p in processing} == {paused.id}


async def test_a_failed_task_still_emits_processing_once() -> None:
    def boom(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        raise RuntimeError("boom")

    agent = Agent(model=FunctionModel(boom), reflexion=False, grounding=False)
    server = A2AServer(agent=agent, allow_unauthenticated=True)
    seen = _Observed()

    async def body(client: A2AClient) -> Any:
        try:
            return await client.send_message(_message("hi"))
        except Exception as exc:  # noqa: BLE001 — either a failed task or a protocol error
            return exc

    await seen.run(server, body)

    assert len(seen.named("a2a.task.processing")) == 1
