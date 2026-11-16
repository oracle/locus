# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus mcp serve``: an agent or a tool module as an MCP server, in one command.

::

    locus mcp serve app.py:agent                       # stdio, for desktop clients
    locus mcp serve billing_tools --transport http     # every @tool in the module
    locus mcp serve app.py:make_agent --port 9000 --path /mcp

The target is ``MODULE:ATTR`` or ``FILE.py:ATTR`` naming an ``Agent``, a
zero-argument function returning one, a ``Tool``, or a list of tools; or a bare
module / file whose module-level ``Tool`` objects are all served. An agent is
served through :class:`~locus.integrations.LocusMCPServer` (its tools plus
``run_agent`` / ``run_agent_stream``); tools go up as themselves.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from locus.tools.decorator import Tool


if TYPE_CHECKING:
    from fastmcp import FastMCP

Transport = Literal["stdio", "http"]


class MCPServeError(Exception):
    """The target could not be served; the message says why."""


def _import_module(target: str) -> Any:
    """Import ``target`` — a dotted module name or a ``.py`` path."""
    if target.endswith(".py") or os.sep in target:
        path = Path(target)
        if not path.is_file():
            raise MCPServeError(f"no such file: {target}")
        name = f"_locus_mcp_serve_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise MCPServeError(f"cannot load {target}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        return importlib.import_module(target)
    except ImportError as exc:
        raise MCPServeError(f"cannot import {target}: {exc}") from exc


def _module_tools(module: Any) -> list[Tool]:
    seen: set[int] = set()
    tools: list[Tool] = []
    for value in vars(module).values():
        if isinstance(value, Tool) and id(value) not in seen:
            seen.add(id(value))
            tools.append(value)
    return tools


def load_target(spec: str) -> tuple[Any | None, list[Tool], str]:
    """What ``spec`` names: ``(agent, tools, default_name)``.

    Exactly one of ``agent`` and ``tools`` is populated.
    """
    from locus.agent import Agent

    target, sep, attr = spec.rpartition(":")
    if not sep or not target or not attr:
        module = _import_module(spec)
        tools = _module_tools(module)
        if not tools:
            raise MCPServeError(f"{spec} defines no tools (module-level Tool objects)")
        name = Path(spec).stem if spec.endswith(".py") else spec.rsplit(".", 1)[-1]
        return None, tools, name

    module = _import_module(target)
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise MCPServeError(f"{target} has no attribute {attr!r}") from exc
    if not isinstance(obj, Agent | Tool) and callable(obj):
        obj = obj()
    if isinstance(obj, Agent):
        return obj, [], getattr(obj.config, "name", None) or attr
    if isinstance(obj, Tool):
        return None, [obj], attr
    if isinstance(obj, list | tuple) and obj and all(isinstance(t, Tool) for t in obj):
        return None, list(obj), attr
    raise MCPServeError(f"{spec} is not an Agent, a Tool, a list of tools, or a factory for one")


def build_server(spec: str, *, name: str | None = None, version: str = "1.0.0") -> FastMCP:
    """The FastMCP server for ``spec``, not yet running."""
    from locus.integrations.fastmcp import LocusMCPServer, create_tool_server

    agent, tools, default_name = load_target(spec)
    if agent is not None:
        return LocusMCPServer(agent=agent, name=name or default_name, version=version).build()
    return create_tool_server(tools, name=name or default_name, version=version)


def serve(
    spec: str,
    *,
    transport: Transport = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp",
    name: str | None = None,
) -> None:
    """Serve ``spec`` over stdio or Streamable HTTP until interrupted."""
    try:
        import fastmcp  # noqa: F401
    except ImportError as exc:
        raise MCPServeError("serving over MCP needs fastmcp: pip install fastmcp") from exc
    server = build_server(spec, name=name)
    kwargs: dict[str, Any] = {"show_banner": False}
    if transport == "stdio":
        server.run(transport="stdio", **kwargs)
        return
    server.run(transport="http", host=host, port=port, path=path, **kwargs)


__all__ = ["MCPServeError", "Transport", "build_server", "load_target", "serve"]
