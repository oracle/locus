# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MCP wiring on the agent.

MCP support was ~90% built and stranded: ``MCPClient`` was not exported,
there was no way to hand an agent an MCP server, and ``to_tulip_tools()``
called ``asyncio.get_event_loop().run_until_complete(...)`` from a sync
method — which raises inside any running loop, and a running loop is the
only way to reach it, since ``await client.connect()`` comes first.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.integrations import MCPClient
from tulip.testing import ScriptedModel, text, tool_call


SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_docs",
        "description": "Search the documentation.",
        "inputSchema": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    },
    {
        "name": "get_page",
        "description": "Fetch one page.",
        "inputSchema": {
            "type": "object",
            "properties": {"page_id": {"type": "string"}},
            "required": ["page_id"],
        },
    },
]


class _FakeServer(MCPClient):
    """An MCP server that answers from memory.

    Subclassed rather than monkeypatched: ``MCPClient`` is a Pydantic model,
    so assigning over a method raises.
    """

    async def connect(self) -> None:
        return None

    async def list_tools(self) -> list[dict[str, Any]]:
        return SCHEMAS

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return f"{name}:{sorted(arguments.items())}"


class _DeadServer(_FakeServer):
    async def list_tools(self) -> list[dict[str, Any]]:
        raise ConnectionError("server is down")


def _server(cls: type[_FakeServer] = _FakeServer) -> _FakeServer:
    return cls(base_url="http://mcp.test/mcp")


# --------------------------------------------------------------------------
# to_tulip_tools — the sync/async bug
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_tulip_tools_works_inside_a_running_loop() -> None:
    """The old implementation raised RuntimeError here."""
    tools = _server().to_tulip_tools(SCHEMAS)
    assert [t.name for t in tools] == ["search_docs", "get_page"]


@pytest.mark.asyncio
async def test_each_tool_calls_its_own_name() -> None:
    """Late binding: a bare closure would give every tool the last name."""
    search, page = _server().to_tulip_tools(SCHEMAS)
    assert await search.fn(q="hi") == "search_docs:[('q', 'hi')]"
    assert await page.fn(page_id="42") == "get_page:[('page_id', '42')]"


@pytest.mark.asyncio
async def test_tool_schema_survives_the_conversion() -> None:
    search = _server().to_tulip_tools(SCHEMAS)[0]
    schema = search.to_openai_schema()["function"]
    assert schema["description"] == "Search the documentation."
    assert schema["parameters"]["required"] == ["q"]


# --------------------------------------------------------------------------
# AgentConfig.mcp_servers
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tools_are_attached_on_the_first_run() -> None:
    agent = Agent(model=ScriptedModel(["ok"], repeat_last=True), mcp_servers=[_server()])
    assert agent.tools.list_tools() == []

    await agent.arun("hello")

    assert agent.tools.list_tools() == ["search_docs", "get_page"]


@pytest.mark.asyncio
async def test_the_agent_can_call_an_mcp_tool() -> None:
    model = ScriptedModel(
        [tool_call("search_docs", q="admission gate"), text("Found it.")],
        repeat_last=True,
    )
    agent = Agent(model=model, mcp_servers=[_server()])

    result = await agent.arun("search for the admission gate")

    assert [t.tool_name for t in result.tool_executions] == ["search_docs"]
    assert "admission gate" in result.tool_executions[0].result
    assert result.text == "Found it."


@pytest.mark.asyncio
async def test_attachment_happens_once() -> None:
    """A second run must not re-list or duplicate the tools."""
    agent = Agent(model=ScriptedModel(["ok"], repeat_last=True), mcp_servers=[_server()])
    await agent.arun("one")
    await agent.arun("two")
    assert agent.tools.list_tools() == ["search_docs", "get_page"]


@pytest.mark.asyncio
async def test_mcp_tools_are_offered_to_the_model() -> None:
    """Attaching is pointless if the model never sees them."""
    model = ScriptedModel(["ok"], repeat_last=True)
    await Agent(model=model, mcp_servers=[_server()]).arun("hi")
    assert model.offered_tools[0] == ["search_docs", "get_page"]


@pytest.mark.asyncio
async def test_an_unreachable_server_does_not_kill_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An optional tool source being down should not refuse the answer."""
    model = ScriptedModel(["answered anyway"], repeat_last=True)
    agent = Agent(model=model, mcp_servers=[_DeadServer(base_url="http://x/mcp")])

    with caplog.at_level(logging.WARNING):
        result = await agent.arun("hello")

    assert result.text == "answered anyway"
    assert agent.tools.list_tools() == []
    assert any("unavailable" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_healthy_server_still_attaches_when_another_is_down() -> None:
    agent = Agent(
        model=ScriptedModel(["ok"], repeat_last=True),
        mcp_servers=[_DeadServer(base_url="http://down/mcp"), _server()],
    )
    await agent.arun("hi")
    assert agent.tools.list_tools() == ["search_docs", "get_page"]


@pytest.mark.asyncio
async def test_no_mcp_servers_is_a_no_op() -> None:
    agent = Agent(model=ScriptedModel(["ok"], repeat_last=True))
    await agent.arun("hi")
    assert agent.tools.list_tools() == []


@pytest.mark.asyncio
async def test_mcp_tools_coexist_with_local_tools() -> None:
    from tulip.tools.decorator import tool

    @tool
    def local_lookup(order_id: str) -> str:
        """Look up an order."""
        return "local"

    agent = Agent(
        model=ScriptedModel(["ok"], repeat_last=True),
        tools=[local_lookup],
        mcp_servers=[_server()],
    )
    await agent.arun("hi")
    assert set(agent.tools.list_tools()) == {"local_lookup", "search_docs", "get_page"}
