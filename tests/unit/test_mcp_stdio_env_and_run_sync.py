# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A stdio MCP server and the process that spawns it.

The server inherits this process's environment (the ``mcp`` SDK alone would
start it with ``PATH``, ``HOME``, ``USER`` and little else), ``env=`` adds to
it, a server that dies during the handshake is reported with its stderr, and
``Agent.run_sync`` refuses — rather than hangs on — MCP tools whose session
is bound to the running loop.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from tests._mcp_deps import require_server_deps


require_server_deps()
pytest.importorskip("fastmcp")

from locus.agent import Agent  # noqa: E402
from locus.core.messages import Role  # noqa: E402
from locus.integrations.fastmcp import MCPClient, MCPConnectionError  # noqa: E402
from locus.testing import FunctionModel, text, tool_call  # noqa: E402


_ENV_SERVER = '''
import os

from fastmcp import FastMCP

mcp = FastMCP("envtest")


@mcp.tool()
def env(name: str) -> str:
    """The value of an environment variable, or <unset>."""
    return os.environ.get(name, "<unset>")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


mcp.run(show_banner=False)
'''

_DYING_SERVER = """
import sys

sys.stderr.write("envtest: OCI_PROFILE is not set, cannot build the model\\n")
sys.exit(3)
"""


@pytest.fixture
def env_server(tmp_path: Path) -> list[str]:
    script = tmp_path / "env_server.py"
    script.write_text(textwrap.dedent(_ENV_SERVER))
    return [sys.executable, str(script)]


async def test_a_stdio_server_inherits_the_environment_and_takes_env(
    env_server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCUS_TEST_INHERITED", "from-parent")
    client = MCPClient(
        server_command=env_server, env={"LOCUS_TEST_EXTRA": "from-env"}, connect_timeout=60
    )
    await client.connect()
    try:
        assert await client.call_tool("env", {"name": "LOCUS_TEST_INHERITED"}) == "from-parent"
        assert await client.call_tool("env", {"name": "LOCUS_TEST_EXTRA"}) == "from-env"
    finally:
        await client.close()


async def test_inherit_env_false_starts_from_the_sdk_environment(
    env_server: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCUS_TEST_INHERITED", "from-parent")
    client = MCPClient(
        server_command=env_server,
        inherit_env=False,
        env={"LOCUS_TEST_EXTRA": "from-env"},
        connect_timeout=60,
    )
    await client.connect()
    try:
        assert await client.call_tool("env", {"name": "LOCUS_TEST_INHERITED"}) == "<unset>"
        assert await client.call_tool("env", {"name": "LOCUS_TEST_EXTRA"}) == "from-env"
    finally:
        await client.close()


async def test_a_server_that_dies_during_setup_is_reported_with_its_stderr(
    tmp_path: Path,
) -> None:
    script = tmp_path / "dying_server.py"
    script.write_text(textwrap.dedent(_DYING_SERVER))
    client = MCPClient(server_command=[sys.executable, str(script)], connect_timeout=60)
    with pytest.raises(MCPConnectionError, match="OCI_PROFILE is not set") as info:
        await client.connect()
    assert "failed during setup" in str(info.value)
    assert info.value.__cause__ is not None


def _adder(messages, tools):  # type: ignore[no-untyped-def]
    answers = [m for m in messages if m.role == Role.TOOL]
    if answers:
        return text(f"sum: {answers[-1].content}")
    return tool_call("add", a=2, b=3)


async def test_run_sync_refuses_mcp_tools_bound_to_the_running_loop(
    env_server: list[str],
) -> None:
    """The MCP page's pattern: connect in ``async def main()``, hand the tools
    to an agent. ``run_sync`` there would drive the run on a worker thread's
    loop, which cannot reach the session, so it says so; ``arun`` works."""
    client = MCPClient(server_command=env_server, connect_timeout=60)
    await client.connect()
    try:
        tools = client.to_locus_tools(await client.list_tools())
        agent = Agent(model=FunctionModel(_adder), tools=tools, reflexion=False, grounding=False)
        with pytest.raises(RuntimeError, match=r"await agent\.arun") as info:
            agent.run_sync("add 2 and 3")
        assert client.label in str(info.value)

        # The same for a pre-connected client passed as mcp_servers=.
        via_servers = Agent(
            model=FunctionModel(_adder), mcp_servers=[client], reflexion=False, grounding=False
        )
        with pytest.raises(RuntimeError, match="running event loop"):
            via_servers.run_sync("add 2 and 3")

        result = await agent.arun("add 2 and 3")
        assert result.message == "sum: 5"
    finally:
        await client.close()
    # Once the session is gone nothing is bound to the loop any more.
    assert client.bound_loop() is None
