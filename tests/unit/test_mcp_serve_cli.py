# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus mcp serve``: an agent or a tool module as an MCP server over stdio.

The smoke test spawns the real command in a child process and talks to it
with ``MCPClient`` — the same path a desktop client takes.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from tests._mcp_deps import require_server_deps


require_server_deps()
pytest.importorskip("fastmcp")

from locus.cli import build_parser, main  # noqa: E402
from locus.integrations.fastmcp import MCPClient, MCPElicitation, create_tool_server  # noqa: E402
from locus.integrations.mcp_serve import MCPServeError, build_server, load_target  # noqa: E402


_TOOLS_MODULE = '''
from locus.tools.decorator import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First.
        b: Second.
    """
    return a + b


@tool
def shout(text: str) -> str:
    """Upper-case the text.

    Args:
        text: What to shout.
    """
    return text.upper()


not_a_tool = 42
'''

_AGENT_MODULE = '''
import json

from locus.agent import Agent
from locus.core.messages import Role
from locus.testing import FunctionModel, text, tool_call
from locus.tools.decorator import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First.
        b: Second.
    """
    return a + b


@tool
def ask_user(question: str) -> str:
    """Ask the person a question and pause until they answer.

    Args:
        question: What to ask.
    """
    return json.dumps({"__interrupt__": True, "question": question, "options": None})


def _model(messages, tools):
    answers = [m for m in messages if m.role == Role.TOOL]
    if not answers:
        return tool_call("ask_user", call_id="q1", question="Which region?")
    return text(f"deployed to {answers[-1].content}")


def make_agent():
    return Agent(
        model=FunctionModel(_model), tools=[add, ask_user], reflexion=False, grounding=False
    )


agent = make_agent()
tools = [add]
'''


@pytest.fixture
def tools_file(tmp_path: Path) -> Path:
    path = tmp_path / "billing_tools.py"
    path.write_text(textwrap.dedent(_TOOLS_MODULE))
    return path


@pytest.fixture
def agent_file(tmp_path: Path) -> Path:
    path = tmp_path / "deploy_app.py"
    path.write_text(textwrap.dedent(_AGENT_MODULE))
    return path


# ---------------------------------------------------------------------------
# The parser and the target loader
# ---------------------------------------------------------------------------


def test_the_parser_knows_mcp_serve() -> None:
    args = build_parser().parse_args(
        ["mcp", "serve", "app.py:agent", "--transport", "http", "--port", "9000", "--name", "x"]
    )
    assert (args.command, args.action, args.target) == ("mcp", "serve", "app.py:agent")
    assert (args.transport, args.host, args.port, args.path, args.name) == (
        "http",
        "127.0.0.1",
        9000,
        "/mcp",
        "x",
    )
    defaults = build_parser().parse_args(["mcp", "serve", "tools"])
    assert defaults.transport == "stdio"


def test_load_target_finds_tools_agents_and_factories(tools_file: Path, agent_file: Path) -> None:
    agent, tools, name = load_target(str(tools_file))
    assert agent is None
    assert [t.name for t in tools] == ["add", "shout"]
    assert name == "billing_tools"

    agent, tools, _ = load_target(f"{agent_file}:agent")
    assert agent is not None
    assert tools == []

    agent, tools, _ = load_target(f"{agent_file}:make_agent")
    assert agent is not None

    agent, tools, name = load_target(f"{agent_file}:tools")
    assert agent is None
    assert [t.name for t in tools] == ["add"]
    assert name == "tools"

    agent, tools, name = load_target(f"{agent_file}:add")
    assert [t.name for t in tools] == ["add"]


def test_load_target_explains_what_is_wrong(tmp_path: Path, tools_file: Path) -> None:
    with pytest.raises(MCPServeError, match="no such file"):
        load_target(str(tmp_path / "missing.py"))
    with pytest.raises(MCPServeError, match="has no attribute"):
        load_target(f"{tools_file}:nope")
    with pytest.raises(MCPServeError, match="not an Agent"):
        load_target(f"{tools_file}:not_a_tool")
    empty = tmp_path / "empty_mod.py"
    empty.write_text("x = 1\n")
    with pytest.raises(MCPServeError, match="defines no tools"):
        load_target(str(empty))
    with pytest.raises(MCPServeError, match="cannot import"):
        load_target("no_such_module_locus_mcp")


def test_a_bad_target_is_a_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["mcp", "serve", str(tmp_path / "missing.py")]) == 2
    assert "no such file" in capsys.readouterr().err


async def test_build_server_publishes_the_tools(tools_file: Path, agent_file: Path) -> None:
    from fastmcp import Client

    async with Client(build_server(str(tools_file))) as client:
        names = sorted(t.name for t in await client.list_tools())
        assert names == ["add", "shout"]
        result = await client.call_tool("add", {"a": "2", "b": "3"})
        assert result.content[0].text == "5"

    async with Client(build_server(f"{agent_file}:agent", name="deploy")) as client:
        names = sorted(t.name for t in await client.list_tools())
        assert {"add", "run_agent", "run_agent_stream"} <= set(names)


async def test_create_tool_server_takes_plain_tools() -> None:
    from fastmcp import Client

    from locus.tools.decorator import tool

    @tool
    def ping() -> str:
        """Pong."""
        return "pong"

    async with Client(create_tool_server([ping], name="pingers")) as client:
        result = await client.call_tool("ping", {})
        assert result.content[0].text == "pong"


# ---------------------------------------------------------------------------
# Smoke over stdio: the real command in a child process
# ---------------------------------------------------------------------------


def _serve_command(target: str) -> list[str]:
    return [sys.executable, "-m", "locus.cli", "mcp", "serve", target]


async def test_serve_a_tool_module_over_stdio(tools_file: Path) -> None:
    client = MCPClient(server_command=_serve_command(str(tools_file)), connect_timeout=60)
    await client.connect()
    try:
        names = sorted(t["name"] for t in await client.list_tools())
        assert names == ["add", "shout"]
        assert await client.call_tool("shout", {"text": "hi"}) == "HI"
        assert await client.call_tool("add", {"a": "2", "b": "40"}) == "42"
    finally:
        await client.close()


async def test_serve_an_agent_over_stdio_relays_its_questions(agent_file: Path) -> None:
    asked: list[MCPElicitation] = []

    def answer(question: MCPElicitation) -> Any:
        asked.append(question)
        return "eu-frankfurt-1"

    client = MCPClient(
        server_command=_serve_command(f"{agent_file}:agent"),
        connect_timeout=60,
        elicitation_handler=answer,
    )
    await client.connect()
    try:
        names = {t["name"] for t in await client.list_tools()}
        assert {"add", "ask_user", "run_agent", "run_agent_stream"} <= names
        assert await client.call_tool("add", {"a": "1", "b": "2"}) == "3"
        # The served agent asks ``ask_user``; the server relays it to us as
        # an elicitation and resumes the agent with our answer.
        reply = await client.call_tool("run_agent_stream", {"prompt": "deploy"})
    finally:
        await client.close()
    assert reply == "deployed to eu-frankfurt-1"
    assert asked[0].message == "Which region?"


# ---------------------------------------------------------------------------
# serve(): the transports, without binding anything
# ---------------------------------------------------------------------------


class _FakeServer:
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> None:
        self.runs.append(kwargs)


def test_serve_picks_the_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    from locus.integrations import mcp_serve

    fake = _FakeServer()
    built: list[tuple[str, str | None]] = []

    def build(spec: str, *, name: str | None = None, version: str = "1.0.0") -> _FakeServer:
        built.append((spec, name))
        return fake

    monkeypatch.setattr(mcp_serve, "build_server", build)
    mcp_serve.serve("app.py:agent", name="support")
    mcp_serve.serve("tools", transport="http", host="0.0.0.0", port=9000, path="/tools")  # noqa: S104
    assert built == [("app.py:agent", "support"), ("tools", None)]
    assert fake.runs == [
        {"transport": "stdio", "show_banner": False},
        {
            "transport": "http",
            "host": "0.0.0.0",  # noqa: S104
            "port": 9000,
            "path": "/tools",
            "show_banner": False,
        },
    ]


def test_serve_names_the_missing_server_package(monkeypatch: pytest.MonkeyPatch) -> None:
    from locus.integrations import mcp_serve

    monkeypatch.setitem(sys.modules, "fastmcp", None)
    with pytest.raises(MCPServeError, match="pip install fastmcp"):
        mcp_serve.serve("app.py:agent")


def test_the_cli_reports_a_serve_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from locus.integrations import mcp_serve

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise ImportError("fastmcp is not installed")

    monkeypatch.setattr(mcp_serve, "serve", refuse)
    assert main(["mcp", "serve", "app.py:agent"]) == 2
    assert "fastmcp is not installed" in capsys.readouterr().err


def test_load_target_when_a_file_has_no_loader(
    tools_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *a, **k: None)
    with pytest.raises(MCPServeError, match="cannot load"):
        load_target(str(tools_file))


def test_load_target_imports_a_module_by_name_from_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "locus_mcp_serve_cwd_tools"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_TOOLS_MODULE))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(tmp_path)])
    try:
        agent, tools, default_name = load_target(name)
    finally:
        sys.modules.pop(name, None)
    assert agent is None
    assert [t.name for t in tools] == ["add", "shout"]
    assert default_name == name
