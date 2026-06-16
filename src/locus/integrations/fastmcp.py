# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""MCP integration for Locus.

Provides both:
1. Server: Expose Locus agents as MCP servers (via fastMCP)
2. Client: Connect to external MCP servers (via mcp SDK)

Works with any MCP-compliant server or client.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from locus.tools.decorator import Tool, tool


if TYPE_CHECKING:
    from fastmcp import FastMCP

    from locus.agent.agent import Agent


# =============================================================================
# Schema utilities - Convert JSON Schema to Pydantic models
# =============================================================================


class _ToolArgsBase(BaseModel):
    """Base class for dynamically generated tool argument models."""

    model_config = ConfigDict(extra="forbid")


def _json_schema_type_to_python(prop: dict[str, Any]) -> type[Any]:
    """Translate a JSON schema fragment into a Python type."""
    schema_type = prop.get("type")

    # Handle nullable types
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        schema_type = non_null[0] if non_null else None

    if schema_type == "array":
        items_schema = prop.get("items")
        if items_schema and isinstance(items_schema, dict):
            item_type = _json_schema_type_to_python(items_schema)
            return list[item_type]  # type: ignore[valid-type]
        return list[Any]

    if schema_type == "object":
        return dict[str, Any]

    mapping: dict[str | None, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    # ``Any`` is a typing-special-form, not a runtime type. Pydantic
    # accepts it as a field type at runtime; mypy 1.13 (pre-commit) is
    # stricter than the local dev mypy here, so the ignore is keyed
    # only on the strict path.
    return mapping.get(schema_type, Any)  # type: ignore[arg-type, unused-ignore]


def build_args_model(tool_name: str, schema: dict[str, Any] | None) -> type[BaseModel] | None:
    """Convert a JSON schema dict into a Pydantic BaseModel.

    This is essential for fastMCP which requires proper function signatures,
    not **kwargs.
    """
    if not isinstance(schema, dict):
        return None

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None

    required = set(schema.get("required", []))
    fields: dict[str, tuple[type[Any], Any]] = {}

    for field_name, prop in properties.items():
        if not isinstance(prop, dict):
            continue

        py_type = _json_schema_type_to_python(prop)
        default = prop.get("default")
        description = prop.get("description")

        if field_name in required and default is None:
            field_info = Field(..., description=description)
        else:
            field_default = default if default is not None else None
            field_info = Field(field_default, description=description)

        fields[field_name] = (py_type, field_info)

    if not fields:
        return None

    model_name = f"MCPTool_{tool_name.replace('-', '_').replace(' ', '_')}_Args"
    return create_model(model_name, __base__=_ToolArgsBase, **fields)  # type: ignore[call-overload,no-any-return]


# =============================================================================
# Tool conversion utilities
# =============================================================================


def mcp_tool_to_locus(
    name: str,
    description: str,
    func: Callable[..., Any],
    parameters: dict[str, Any] | None = None,
) -> Tool:
    """
    Convert an MCP-style tool to a Locus Tool.

    When ``parameters`` is provided, the JSON Schema is used **as-is** to
    construct the Tool. This preserves the source tool's flat-field
    schema end-to-end so the LLM sees the original argument shape
    (e.g. ``{tenant_id, regex, limit}``) instead of the generic
    ``{kwargs: …}`` shape that the ``@tool`` decorator would otherwise
    derive from the wrapper's ``**kwargs`` signature.

    When ``parameters`` is omitted, falls back to the decorator-derived
    schema (parameter-less tool) for backward compatibility.

    Args:
        name: Tool name
        description: Tool description
        func: The async function to call
        parameters: JSON Schema for parameters

    Returns:
        Locus Tool instance
    """

    async def _invoke(**kwargs: Any) -> str:
        result = await func(**kwargs)
        if isinstance(result, str):
            return result
        return json.dumps(result)

    if parameters is not None:
        # Direct construction: keep the source MCP server's
        # inputSchema as the Tool's parameters dict. The Tool's
        # execute path forwards ``**kwargs`` to ``_invoke`` which
        # forwards them to the original ``func``, so the LLM's tool
        # call args land flat at the server.
        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            fn=_invoke,
            idempotent=False,
        )

    # Fallback: derive the schema from the wrapper signature.
    # No-args tools work; tools that need typed args should pass
    # ``parameters=`` explicitly.
    @tool(name=name, description=description)
    async def wrapper(**kwargs: Any) -> str:
        return await _invoke(**kwargs)

    return wrapper


def locus_tool_to_mcp(locus_tool: Tool) -> dict[str, Any]:
    """
    Convert a Locus Tool to MCP tool schema.

    Args:
        locus_tool: Locus Tool instance

    Returns:
        MCP-compatible tool definition
    """
    return {
        "name": locus_tool.name,
        "description": locus_tool.description or "",
        "inputSchema": locus_tool.parameters or {"type": "object", "properties": {}},
    }


# =============================================================================
# MCP Server (uses fastMCP)
# =============================================================================


_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _create_tool_wrapper(tool_obj: Tool) -> Callable:
    """Create a wrapper function for a Locus tool that fastMCP can use.

    FastMCP introspects the wrapper's signature to build its JSON Schema,
    so we cannot just hand it a bare ``**kwargs`` function. Historically we
    built source text and ran ``exec(compile(...))`` with interpolated tool
    and parameter names. Even with tight identifier validation, that path
    carried standing RCE risk (CWE-94) against a compromised or hostile MCP
    manifest, and tripped bandit S102.

    This implementation instead:

      1. Validates tool and parameter names against a strict identifier
         allow-list (defence in depth; a future refactor might drop the
         check otherwise).
      2. Builds a plain async closure over ``tool_obj.execute``.
      3. Attaches a synthetic ``inspect.Signature`` so fastMCP sees the
         declared parameters without any source code being evaluated.

    No ``exec`` / ``compile`` on attacker-influenced strings.
    """
    params = tool_obj.parameters or {"type": "object", "properties": {}}
    properties = params.get("properties", {})
    required = set(params.get("required", []))

    # Validate the tool name even for the no-args path so call sites see a
    # consistent error for malformed manifests.
    safe_func_name = tool_obj.name.replace("-", "_")
    if not _SAFE_IDENTIFIER_RE.match(safe_func_name):
        raise ValueError(f"Unsafe tool name: {tool_obj.name!r}")

    # Validate every parameter name up front so partial schemas fail loudly
    # rather than at first call.
    param_names = list(properties.keys())
    for name in param_names:
        if not _SAFE_IDENTIFIER_RE.match(name):
            raise ValueError(f"Unsafe parameter name: {name!r}")

    async def _invoke(**kwargs: Any) -> str:
        # Drop None placeholders that we used to model optional params.
        real_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        result = await tool_obj.execute(**real_kwargs)
        if isinstance(result, str):
            return result
        return json.dumps(result)

    if not param_names:
        # fastMCP is fine with a zero-arg callable here — no signature
        # synthesis needed.
        async def no_args_wrapper() -> str:
            return await _invoke()

        no_args_wrapper.__name__ = tool_obj.name
        no_args_wrapper.__doc__ = tool_obj.description
        return no_args_wrapper

    # Build a signature that fastMCP / inspect can walk. Required params are
    # positional-or-keyword without a default; optional params get a None
    # default so fastMCP records them as optional.
    sig_params: list[inspect.Parameter] = []
    for name in param_names:
        if name in required:
            sig_params.append(
                inspect.Parameter(
                    name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=str,
                )
            )
        else:
            sig_params.append(
                inspect.Parameter(
                    name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=str,
                )
            )
    synthetic_sig = inspect.Signature(
        parameters=sig_params,
        return_annotation=str,
    )

    async def wrapper(**kwargs: Any) -> str:
        # Filter to declared parameters so stray kwargs from fastMCP routing
        # never reach the user tool.
        accepted = {k: v for k, v in kwargs.items() if k in param_names}
        return await _invoke(**accepted)

    wrapper.__name__ = tool_obj.name
    wrapper.__doc__ = tool_obj.description
    wrapper.__signature__ = synthetic_sig  # type: ignore[attr-defined]
    # Pydantic's TypeAdapter consults ``typing.get_type_hints`` (i.e.
    # ``__annotations__``) rather than the synthetic signature, so we
    # populate annotations too. Without this, fastMCP's schema-generation
    # path raises ``KeyError`` for the declared parameter names.
    annotations: dict[str, Any] = dict.fromkeys(param_names, str)
    annotations["return"] = str
    wrapper.__annotations__ = annotations
    return wrapper


class LocusMCPServer(BaseModel):
    """
    Exposes a Locus Agent as an MCP server.

    This allows Locus agents to be used by any MCP-compatible client.

    Example:
        >>> from locus import Agent
        >>> from locus.integrations import LocusMCPServer
        >>>
        >>> agent = Agent(model=model, tools=[...])
        >>> server = LocusMCPServer(agent=agent, name="my-agent")
        >>> server.run()  # Starts MCP server
    """

    agent: Any = Field(..., description="Locus Agent instance")
    name: str = Field(default="locus-agent", description="Server name")
    version: str = Field(default="1.0.0", description="Server version")

    _mcp: FastMCP | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _create_mcp(self) -> FastMCP:
        """Create FastMCP server instance."""
        from fastmcp import FastMCP

        mcp = FastMCP(self.name)

        # Register agent's tools as MCP tools
        if hasattr(self.agent, "_tool_registry"):
            self.agent._initialize()
            for tool_obj in self.agent._tool_registry.tools.values():
                wrapper = _create_tool_wrapper(tool_obj)
                mcp.tool()(wrapper)

        # Register the main "run" tool that invokes the agent
        agent = self.agent

        @mcp.tool()
        async def run_agent(prompt: str) -> str:
            """Run the Locus agent with a prompt and return the response."""
            result = agent.run_sync(prompt)
            return str(result.message)

        # Register a streaming version
        @mcp.tool()
        async def run_agent_stream(prompt: str) -> str:
            """Run the agent with streaming, returning final result."""
            events = []
            async for event in agent.run(prompt):
                events.append(event)
            # Return the final message from the last event
            for event in reversed(events):
                if hasattr(event, "final_message") and event.final_message:
                    return str(event.final_message)
            return "Agent completed without response"

        return mcp

    def run(self, transport: Literal["stdio", "http", "sse", "streamable-http"] = "stdio") -> None:
        """
        Run the MCP server.

        Args:
            transport: Transport type ("stdio", "http", "sse", or "streamable-http").
        """
        if self._mcp is None:
            self._mcp = self._create_mcp()

        self._mcp.run(transport=transport)

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a single MCP request (for testing)."""
        if self._mcp is None:
            self._mcp = self._create_mcp()

        # Process based on method
        method = request.get("method", "")

        if method == "tools/list":
            tools = []
            if hasattr(self.agent, "_tool_registry"):
                self.agent._initialize()
                for tool_obj in self.agent._tool_registry.tools.values():
                    tools.append(locus_tool_to_mcp(tool_obj))
            return {"tools": tools}

        if method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            if tool_name == "run_agent":
                result = self.agent.run_sync(arguments.get("prompt", ""))
                return {"content": [{"type": "text", "text": result.message}]}

            # Find and execute the tool
            if hasattr(self.agent, "_tool_registry"):
                self.agent._initialize()
                tool_obj = self.agent._tool_registry.get(tool_name)
                if tool_obj:
                    result = await tool_obj.execute(**arguments)
                    text = result if isinstance(result, str) else json.dumps(result)
                    return {"content": [{"type": "text", "text": text}]}

            return {"error": {"code": -32602, "message": f"Unknown tool: {tool_name}"}}

        return {"error": {"code": -32601, "message": f"Unknown method: {method}"}}


def create_mcp_server(
    agent: Agent,
    name: str = "locus-agent",
    version: str = "1.0.0",
) -> LocusMCPServer:
    """
    Create an MCP server from a Locus Agent.

    Args:
        agent: Locus Agent instance
        name: Server name
        version: Server version

    Returns:
        LocusMCPServer instance

    Example:
        >>> server = create_mcp_server(agent, name="my-assistant")
        >>> server.run()
    """
    return LocusMCPServer(agent=agent, name=name, version=version)


# =============================================================================
# MCP Client (uses mcp SDK for full compatibility)
# =============================================================================


class MCPClient(BaseModel):
    """
    Client for connecting to external MCP servers.

    Uses the official MCP SDK for full protocol compatibility.
    Supports both stdio and HTTP transports.

    Example:
        >>> # Connect to stdio MCP server
        >>> client = MCPClient(server_command=["python", "mcp_server.py"])
        >>> await client.connect()
        >>> tools = await client.list_tools()
        >>> result = await client.call_tool("search", {"query": "hello"})
        >>> await client.close()

        >>> # Connect to HTTP MCP server
        >>> client = MCPClient(base_url="https://mcp.example.com")
        >>> await client.connect()
        >>> ...
    """

    # Stdio transport
    server_command: list[str] | None = Field(
        default=None, description="Command to start stdio MCP server"
    )

    # HTTP transport
    base_url: str | None = Field(default=None, description="URL for HTTP MCP server")
    access_token: str | None = Field(default=None, description="Bearer token for auth")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    verify_url: bool = Field(
        default=True,
        description=(
            "Run the SSRF pre-flight guard on base_url before connecting. "
            "Set to False for in-cluster / loopback MCP servers that "
            "resolve to private addresses. Cloud metadata endpoints are "
            "blocked regardless of this flag."
        ),
    )
    allow_private_url: bool = Field(
        default=False,
        description=(
            "When verify_url=True, permit base_url to resolve to a "
            "private / loopback / link-local address. Cloud metadata "
            "endpoints are blocked regardless."
        ),
    )
    verify_packages: bool = Field(
        default=True,
        description=(
            "For stdio MCP servers launched via npx/uvx/pipx/bunx/pnpx, "
            "consult the OSV malware database before spawning and refuse "
            "to launch any package with MAL-* advisories. Fails open on "
            "network errors. Set LOCUS_MCP_SKIP_OSV=1 to disable globally."
        ),
    )

    _session: Any = None
    _client_context: Any = None
    _connected: bool = False
    _process: Any = None

    model_config = {"arbitrary_types_allowed": True}

    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self._connected:
            return

        if self.base_url:
            await self._connect_http()
        elif self.server_command:
            await self._connect_stdio()
        else:
            raise ValueError("Must provide either base_url or server_command")

        self._connected = True

    async def _connect_http(self) -> None:
        """Connect via HTTP/SSE transport."""
        if self.base_url is None:
            msg = "_connect_http called without base_url"
            raise RuntimeError(msg)
        try:
            from mcp.client.session import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as e:
            raise ImportError(
                "mcp package required for HTTP transport. Install with: pip install mcp"
            ) from e

        # Pre-flight SSRF guard. Rejecting a model-supplied base_url that
        # resolves to a cloud-metadata endpoint or a private network is
        # the one check we can do cheaply before any bytes go on the wire.
        # Redirect-based bypass is a known limitation — see url_safety.py.
        if self.verify_url and self.base_url:
            from locus.tools.url_safety import validate_url

            validate_url(self.base_url, allow_private=self.allow_private_url)

        # Set up auth if token provided
        auth = None
        if self.access_token:
            import httpx

            class BearerAuth(httpx.Auth):
                def __init__(self, token: str):
                    self.token = token

                def auth_flow(  # type: ignore[no-untyped-def]
                    self, request
                ):  # httpx.Auth.auth_flow signature varies across SDK versions
                    request.headers["Authorization"] = f"Bearer {self.token}"
                    yield request

            auth = BearerAuth(self.access_token)

        # Propagate verify_ssl to the underlying httpx client so the config
        # field is actually enforced. Without this, a caller who disables
        # or enables TLS verification sees no effect — the default was used.
        import httpx

        verify_ssl = self.verify_ssl

        def _httpx_factory(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                headers=headers,
                timeout=timeout if timeout is not None else httpx.Timeout(30.0),
                auth=auth,
                verify=verify_ssl,
                follow_redirects=True,
            )

        self._client_context = streamablehttp_client(
            self.base_url,
            auth=auth,
            httpx_client_factory=_httpx_factory,
        )
        read_stream, write_stream, _ = await self._client_context.__aenter__()

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport."""
        try:
            from mcp.client.session import ClientSession
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            raise ImportError(
                "mcp package required for stdio transport. Install with: pip install mcp"
            ) from e

        from mcp import StdioServerParameters

        assert self.server_command is not None  # guarded by connect()
        cmd = self.server_command[0]
        cmd_args = self.server_command[1:] if len(self.server_command) > 1 else []

        # OSV malware pre-check for supply-chain launchers (npx/uvx/…).
        # Fails open on any lookup issue; see locus.integrations.osv for
        # the full behaviour contract.
        if self.verify_packages:
            from locus.core.errors import ValidationError
            from locus.integrations.osv import check_package_for_malware

            reason = check_package_for_malware(cmd, cmd_args)
            if reason:
                raise ValidationError(f"MCP launch blocked: {reason}")

        server_params = StdioServerParameters(
            command=cmd,
            args=cmd_args,
        )

        self._client_context = stdio_client(server_params)
        read_stream, write_stream = await self._client_context.__aenter__()

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from the MCP server."""
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")

        result = await self._session.list_tools()

        # Convert MCP Tool objects to dicts
        tools = []
        for mcp_tool in result.tools:
            tools.append(
                {
                    "name": mcp_tool.name,
                    "description": mcp_tool.description or "",
                    "inputSchema": mcp_tool.inputSchema if hasattr(mcp_tool, "inputSchema") else {},
                }
            )
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server."""
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")

        result = await self._session.call_tool(name=name, arguments=arguments)

        # Extract text from result content
        if hasattr(result, "content"):
            texts = []
            for item in result.content:
                if hasattr(item, "text"):
                    texts.append(item.text)
            return "\n".join(texts) if texts else str(result)

        return str(result)

    async def close(self) -> None:
        """Close the connection."""
        self._connected = False

        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass
            self._session = None

        if self._client_context:
            try:
                await self._client_context.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass
            self._client_context = None

    async def __aenter__(self) -> MCPClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    def to_locus_tools(self, tools: list[dict[str, Any]]) -> list[Tool]:
        """Convert MCP tools to Locus tools."""
        locus_tools = []
        for mcp_tool in tools:
            # Create a closure to capture the tool name
            tool_name = mcp_tool["name"]

            async def make_func(name: str = tool_name) -> Callable:
                async def func(**kwargs: Any) -> str:
                    return await self.call_tool(name, kwargs)

                return func

            locus_tool = mcp_tool_to_locus(
                name=mcp_tool["name"],
                description=mcp_tool.get("description", ""),
                func=asyncio.get_event_loop().run_until_complete(make_func()),
                parameters=mcp_tool.get("inputSchema"),
            )
            locus_tools.append(locus_tool)

        return locus_tools
