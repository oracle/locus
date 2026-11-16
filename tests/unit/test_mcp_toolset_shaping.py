# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``include`` / ``exclude`` / ``prefix`` on the MCP toolset.

Precedence, as documented on the concept page: ``allowed_tools`` and
``include`` narrow (both must match when given), ``exclude`` removes and wins
over ``include``, ``tool_filter`` decides last, and ``prefix`` renames after
all matching — the patterns always see the server's own names.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from tests._mcp_deps import require_server_deps


require_server_deps()

from locus.agent import Agent  # noqa: E402
from locus.core.events import ToolCompleteEvent  # noqa: E402
from locus.integrations.fastmcp import MCPClient, MCPToolNotAllowedError  # noqa: E402
from locus.testing import ScriptedModel, text, tool_call  # noqa: E402
from tests.unit.test_mcp_fidelity import _LoopbackServer  # noqa: E402


SCHEMAS = [
    {"name": "search_hotels", "description": "", "inputSchema": {"type": "object"}},
    {"name": "search_flights", "description": "", "inputSchema": {"type": "object"}},
    {"name": "book_hotel", "description": "", "inputSchema": {"type": "object"}},
    {"name": "admin_reset", "description": "", "inputSchema": {"type": "object"}},
]


def _names(client: MCPClient) -> list[str]:
    return [t.name for t in client.to_locus_tools(SCHEMAS)]


def test_include_takes_names_and_globs() -> None:
    assert _names(MCPClient(base_url="https://x.example", include=["search_*"])) == [
        "search_hotels",
        "search_flights",
    ]
    assert _names(MCPClient(base_url="https://x.example", include=["book_hotel"])) == ["book_hotel"]


def test_exclude_wins_over_include() -> None:
    client = MCPClient(
        base_url="https://x.example", include=["search_*", "book_*"], exclude=["*_flights"]
    )
    assert _names(client) == ["search_hotels", "book_hotel"]
    # Excluding something include also names: exclude decides.
    client = MCPClient(base_url="https://x.example", include=["admin_reset"], exclude=["admin_*"])
    assert _names(client) == []


def test_allowed_tools_and_include_both_narrow() -> None:
    client = MCPClient(
        base_url="https://x.example",
        allowed_tools=["search_hotels", "admin_reset"],
        include=["search_*"],
    )
    assert _names(client) == ["search_hotels"]


def test_tool_filter_runs_last_on_the_survivors() -> None:
    seen: list[str] = []

    def no_admin(schema: dict[str, Any]) -> bool:
        seen.append(schema["name"])
        return not schema["name"].startswith("admin")

    client = MCPClient(base_url="https://x.example", exclude=["*_flights"], tool_filter=no_admin)
    assert _names(client) == ["search_hotels", "book_hotel"]
    assert "search_flights" not in seen  # already excluded; the predicate never saw it


def test_prefix_renames_after_matching_and_is_not_part_of_the_patterns() -> None:
    client = MCPClient(
        base_url="https://x.example", prefix="travel_", include=["search_*"], exclude=["travel_*"]
    )
    # ``exclude=["travel_*"]`` matches nothing: patterns see server names.
    assert _names(client) == ["travel_search_hotels", "travel_search_flights"]
    # Calls through the prefixed name resolve to the server's name.
    assert client._server_name("travel_search_hotels") == "search_hotels"
    assert client._server_name("search_hotels") == "search_hotels"


# ---------------------------------------------------------------------------
# Two servers mounted side by side, against a real server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server() -> Iterator[_LoopbackServer]:
    srv = _LoopbackServer().start()
    yield srv
    srv.stop()


async def test_two_servers_mount_without_a_name_clash(server: _LoopbackServer) -> None:
    # The same server twice stands in for two servers that both publish a
    # ``search_hotels``; prefixes keep the agent's tool names distinct.
    billing = MCPClient(
        base_url=server.url, verify_url=False, prefix="billing_", include=["search_hotels"]
    )
    travel = MCPClient(
        base_url=server.url, verify_url=False, prefix="travel_", include=["search_hotels"]
    )
    model = ScriptedModel(
        [
            tool_call("billing_search_hotels", call_id="b1", city="Rome"),
            tool_call("travel_search_hotels", call_id="t1", city="Oslo"),
            text("ok"),
        ]
    )
    agent = Agent(model=model, mcp_servers=[billing, travel], reflexion=False, grounding=False)
    events = [event async for event in agent.run("go")]
    done = {e.tool_call_id: e for e in events if isinstance(e, ToolCompleteEvent)}
    assert done["b1"].tool_name == "billing_search_hotels"
    assert done["t1"].tool_name == "travel_search_hotels"
    assert "Grand Rome" in (done["b1"].result or "")
    assert "Grand Oslo" in (done["t1"].result or "")
    tools = {t.name for t in agent._tool_registry.tools.values()}
    assert {"billing_search_hotels", "travel_search_hotels"} <= tools
    assert "search_hotels" not in tools
    await billing.close()
    await travel.close()


async def test_list_tools_honours_include_exclude_and_calls_are_blocked(
    server: _LoopbackServer,
) -> None:
    client = MCPClient(
        base_url=server.url,
        verify_url=False,
        include=["search_hotels", "explicit_*", "whoami"],
        exclude=["explicit_error"],
        prefix="fid.",
    )
    await client.connect()
    try:
        listed = sorted(t["name"] for t in await client.list_tools())
        assert listed == ["explicit_both", "search_hotels", "whoami"]  # server names
        tools = sorted(t.name for t in client.to_locus_tools(await client.list_tools()))
        assert tools == ["fid.explicit_both", "fid.search_hotels", "fid.whoami"]
        # Through the prefixed name or the server's, the call lands.
        assert "Grand Rome" in await client.call_tool("fid.search_hotels", {"city": "Rome"})
        assert "Grand Oslo" in await client.call_tool("search_hotels", {"city": "Oslo"})
        with pytest.raises(MCPToolNotAllowedError):
            await client.call_tool("explicit_error", {})
        with pytest.raises(MCPToolNotAllowedError):
            await client.call_tool("fid.photo", {})
    finally:
        await client.close()
