# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""MCP integration for Tulip.

Provides both:
1. Server: Expose Tulip agents as MCP servers (via fastMCP)
2. Client: Connect to external MCP servers (via mcp SDK)

Works with any MCP-compliant server or client.

The client keeps what an MCP server returns, not only its text: see
:class:`MCPToolResult` (``structuredContent``, ``isError``, images and
resources), per-request identity (:class:`MCPRequestContext`,
``MCPClient.headers_provider``), tool allowlisting
(``MCPClient.allowed_tools`` / ``tool_filter``), progress notifications
(surfaced as :class:`~tulip.core.events.ToolProgressEvent`) and
:class:`MCPConnectionError` for a server that is down or drops mid-call.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Hashable, Mapping
from contextlib import AsyncExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model

from tulip.core.media import encode_image
from tulip.tools.context import current_tool_context, progress_reporter
from tulip.tools.decorator import Tool, tool
from tulip.tools.output import ToolOutput


if TYPE_CHECKING:
    from fastmcp import FastMCP

    from tulip.agent.agent import Agent


logger = logging.getLogger(__name__)

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


def mcp_tool_to_tulip(
    name: str,
    description: str,
    func: Callable[..., Any],
    parameters: dict[str, Any] | None = None,
    *,
    output_schema: dict[str, Any] | None = None,
    emits_progress: bool = False,
) -> Tool:
    """
    Convert an MCP-style tool to a Tulip Tool.

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
        output_schema: The tool's declared ``outputSchema``, kept on
            :attr:`Tool.output_schema`.
        emits_progress: Stream progress the tool reports as live
            ``ToolProgressEvent`` s (see :attr:`Tool.emits_progress`).

    Returns:
        Tulip Tool instance
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
            output_schema=output_schema if isinstance(output_schema, dict) else None,
            emits_progress=emits_progress,
        )

    # Fallback: derive the schema from the wrapper signature.
    # No-args tools work; tools that need typed args should pass
    # ``parameters=`` explicitly.
    @tool(name=name, description=description)
    async def wrapper(**kwargs: Any) -> str:
        return await _invoke(**kwargs)

    wrapper.emits_progress = emits_progress
    wrapper.output_schema = output_schema if isinstance(output_schema, dict) else None
    return wrapper


def tulip_tool_to_mcp(tulip_tool: Tool) -> dict[str, Any]:
    """
    Convert a Tulip Tool to MCP tool schema.

    Args:
        tulip_tool: Tulip Tool instance

    Returns:
        MCP-compatible tool definition
    """
    return {
        "name": tulip_tool.name,
        "description": tulip_tool.description or "",
        "inputSchema": tulip_tool.parameters or {"type": "object", "properties": {}},
    }


# =============================================================================
# MCP Server (uses fastMCP)
# =============================================================================


_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _create_tool_wrapper(tool_obj: Tool) -> Callable:
    """Create a wrapper function for a Tulip tool that fastMCP can use.

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


class TulipMCPServer(BaseModel):
    """
    Exposes a Tulip Agent as an MCP server.

    This allows Tulip agents to be used by any MCP-compatible client.

    Example:
        >>> from tulip import Agent
        >>> from tulip.integrations import TulipMCPServer
        >>>
        >>> agent = Agent(model=model, tools=[...])
        >>> server = TulipMCPServer(agent=agent, name="my-agent")
        >>> server.run()  # Starts MCP server
    """

    agent: Any = Field(..., description="Tulip Agent instance")
    name: str = Field(default="tulip-agent", description="Server name")
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
            """Run the Tulip agent with a prompt and return the response."""
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
                    tools.append(tulip_tool_to_mcp(tool_obj))
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
    name: str = "tulip-agent",
    version: str = "1.0.0",
) -> TulipMCPServer:
    """
    Create an MCP server from a Tulip Agent.

    Args:
        agent: Tulip Agent instance
        name: Server name
        version: Server version

    Returns:
        TulipMCPServer instance

    Example:
        >>> server = create_mcp_server(agent, name="my-assistant")
        >>> server.run()
    """
    return TulipMCPServer(agent=agent, name=name, version=version)


# =============================================================================
# MCP Client (uses mcp SDK for full compatibility)
# =============================================================================


class MCPConnectionError(ConnectionError):
    """The MCP server could not be reached, or the connection dropped.

    Raised instead of the transport's own exceptions — including the
    ``CancelledError`` an anyio cancel scope leaks when a streamable-HTTP
    connection dies — so a dead server surfaces as an ordinary tool error
    and never as a cancellation of the agent run that happened to call it.
    """


class MCPToolNotAllowedError(PermissionError):
    """A call named a tool the client's ``allowed_tools`` / ``tool_filter`` excludes."""


@dataclass(frozen=True)
class MCPRequestContext:
    """What a :attr:`MCPClient.headers_provider` knows about a request.

    Attributes:
        metadata: The agent run's ``metadata`` (``agent.run(..., metadata=)``)
            — the natural place for the end user's identity or token.
        run_id: The agent run id, when the request serves a tool call.
        tool_name: The MCP tool being called; ``None`` for ``tools/list``.
        tool_call_id: The model's tool-call id, when serving a tool call.
    """

    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    run_id: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None


#: ``(MCPRequestContext) -> headers`` — sync or async; ``None``/``{}`` adds none.
HeadersProvider = Callable[
    [MCPRequestContext],
    Mapping[str, str] | None | Awaitable[Mapping[str, str] | None],
]

#: ``(MCPRequestContext, resolved per-run headers) -> key`` — which
#: per-identity MCP session serves a request. ``None`` falls back to keying by
#: the full headers.
SessionKey = Callable[[MCPRequestContext, Mapping[str, str]], Hashable | None]


@dataclass(frozen=True)
class MCPToolResult:
    """Everything an MCP ``tools/call`` returned, not just its text.

    Attributes:
        text: The model-facing text: text blocks joined by newlines, embedded
            text resources inlined, images embedded as Tulip image segments
            (see :mod:`tulip.core.media`) and other blocks as short
            ``[audio: …]`` / ``[resource: …]`` markers so nothing vanishes
            silently. Falls back to the JSON of ``structured_content`` when
            the server sent no content blocks.
        is_error: The server's ``isError`` flag — the tool ran and failed.
        structured_content: The server's ``structuredContent``, if any.
        content: Every content block as a JSON-mode dict in MCP's shape.
        meta: The result's ``_meta``, if any.
    """

    text: str
    is_error: bool = False
    structured_content: dict[str, Any] | None = None
    content: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] | None = None

    def to_tool_output(self) -> ToolOutput:
        """This result as a :class:`~tulip.tools.output.ToolOutput`.

        Text blocks are already in the string; the rest travel as
        ``content_blocks``.
        """
        blocks = [block for block in self.content if block.get("type") != "text"]
        return ToolOutput(
            self.text,
            structured_content=self.structured_content,
            content_blocks=blocks or None,
            is_error=self.is_error,
        )


_MISSING = object()


def _mcp_field(obj: Any, camel: str, snake: str, *, default: Any = None) -> Any:
    """Read a field off an ``mcp`` SDK model on either major version.

    ``tulip[mcp]`` accepts ``mcp>=1.0``. mcp 1.x models expose the wire names
    as attributes (``isError``, ``structuredContent``, ``inputSchema``,
    ``mimeType``); mcp 2.x renamed them to snake_case (``is_error``, ...) and
    keeps the camelCase spelling only as a validation/serialisation alias, so
    ``getattr(result, "isError")`` silently misses on 2.x.
    """
    value = getattr(obj, camel, _MISSING)
    if value is _MISSING:
        value = getattr(obj, snake, default)
    return value


def _dump_block(item: Any) -> dict[str, Any] | None:
    """A content block as a JSON-mode dict, or None for a non-model value."""
    dump = getattr(item, "model_dump", None)
    if not callable(dump):
        return None
    try:
        dumped = dump(mode="json", by_alias=True, exclude_none=True)
    except Exception:  # noqa: BLE001 — a malformed block must not fail the call
        return None
    return dumped if isinstance(dumped, dict) else None


def _image_segment(data: Any, mime_type: str) -> str:
    """An MCP image block as a Tulip image segment, or a marker if unusable."""
    if isinstance(data, str) and data:
        try:
            return encode_image(base64.b64decode(data, validate=True), mime_type)
        except (binascii.Error, ValueError):
            pass
    return f"[image: {mime_type}]"


def _block_text(item: Any, *, embed_images: bool) -> str | None:
    """The model-facing text for one content block."""
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text
    kind = getattr(item, "type", None)
    mime = _mcp_field(item, "mimeType", "mime_type")
    mime = mime if isinstance(mime, str) and mime else None
    if kind == "image":
        if embed_images:
            return _image_segment(getattr(item, "data", None), mime or "image/png")
        return f"[image: {mime or 'image'}]"
    if kind == "audio":
        return f"[audio: {mime or 'audio'}]"
    if kind == "resource":
        resource = getattr(item, "resource", None)
        inner = getattr(resource, "text", None)
        if isinstance(inner, str):
            return inner
        inner_mime = _mcp_field(resource, "mimeType", "mime_type") or "binary"
        return f"[resource: {getattr(resource, 'uri', '')} ({inner_mime})]"
    if kind == "resource_link":
        return f"[resource link: {getattr(item, 'name', '')} {getattr(item, 'uri', '')}]"
    return None


def _convert_call_result(raw: Any, *, embed_images: bool = True) -> MCPToolResult:
    """Translate an MCP ``CallToolResult`` into an :class:`MCPToolResult`."""
    items = getattr(raw, "content", None)
    if items is None:
        return MCPToolResult(text=str(raw))

    parts: list[str] = []
    blocks: list[dict[str, Any]] = []
    for item in items or []:
        block = _dump_block(item)
        if block is not None:
            blocks.append(block)
        rendered = _block_text(item, embed_images=embed_images)
        if rendered is not None:
            parts.append(rendered)

    structured = _mcp_field(raw, "structuredContent", "structured_content")
    structured = structured if isinstance(structured, dict) else None
    is_error = _mcp_field(raw, "isError", "is_error", default=False) is True
    meta = getattr(raw, "meta", None)
    meta = meta if isinstance(meta, dict) else None

    if parts:
        text = "\n".join(parts)
    elif structured is not None:
        # The spec asks servers to mirror structured content as text; not all
        # do, and a model shown an empty result reads it as "nothing found".
        text = json.dumps(structured, default=str)
    else:
        text = ""
    return MCPToolResult(
        text=text,
        is_error=is_error,
        structured_content=structured,
        content=blocks,
        meta=meta,
    )


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """Whether ``fn`` takes keyword ``name`` — older mcp releases lack some."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _being_cancelled() -> bool:
    """Whether the *current task* has a pending cancellation request.

    Distinguishes a genuine cancellation of the agent run (propagate it) from
    a ``CancelledError`` an anyio cancel scope leaks out of a dying transport
    (convert it into :class:`MCPConnectionError`).
    """
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


#: ``McpError`` code the SDK uses when the connection closed under a request.
_MCP_CONNECTION_CLOSED = -32000

#: Exception type names that only arise when the transport itself is gone —
#: anyio memory streams closed under the session, httpx transport failures.
_CONNECTION_LOSS_TYPES = frozenset(
    {
        "BrokenResourceError",
        "ClosedResourceError",
        "EndOfStream",
        "ConnectError",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
        "ConnectTimeout",
    }
)


def _is_connection_loss(exc: BaseException) -> bool:
    """Whether ``exc`` means the connection is gone (not a tool/protocol error)."""
    exc = _unwrap(exc)
    if type(exc).__name__ in _CONNECTION_LOSS_TYPES or isinstance(exc, ConnectionError):
        return True
    error = getattr(exc, "error", None)
    return getattr(error, "code", None) == _MCP_CONNECTION_CLOSED


def _unwrap(exc: BaseException) -> BaseException:
    """The single leaf of a one-member exception group, else ``exc``."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


#: Run metadata for requests made outside a tool call (``tools/list`` while
#: attaching), so a :class:`HeadersProvider` sees the same metadata there.
_request_metadata: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "tulip_mcp_request_metadata", default=None
)


class _SessionHandle:
    """One MCP session, owned by a dedicated task.

    The transport (``streamablehttp_client`` / ``stdio_client``) and
    ``ClientSession`` are anyio context managers with task groups inside.
    Entering them in the caller's task — as this client used to — binds their
    cancel scopes to that task: a transport failure then cancels *the agent
    run*, and closing from any other task (or at GC) fails with "Attempted to
    exit cancel scope in a different task". Here one runner task enters them,
    parks until asked to stop, and exits them itself; callers use the session
    from their own tasks, which the SDK supports.
    """

    def __init__(self, label: str, headers: dict[str, str] | None = None) -> None:
        self.label = label
        self.session: Any = None
        self.transport: Any = None
        self.task: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None
        #: Per-identity headers, applied to every HTTP request this session
        #: sends. Mutable: a session keyed by principal picks up a rotated
        #: token on its next request instead of opening a new session.
        self.headers: dict[str, str] = dict(headers or {})
        #: ``time.monotonic()`` of the last request start/finish.
        self.last_used: float = time.monotonic()
        #: Requests currently using this session (idle eviction skips it).
        self.in_flight: int = 0

    @property
    def alive(self) -> bool:
        """Whether the runner — and so the connection — is still up."""
        return self.task is not None and not self.task.done()

    async def start(
        self,
        transport: Any,
        session_factory: Callable[[Any, Any], Any],
        connect_timeout: float | None,
    ) -> Any:
        """Open the connection and return the initialized session."""
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[Any] = loop.create_future()
        self._stop = asyncio.Event()
        self.transport = transport
        self.task = loop.create_task(
            self._run(transport, session_factory, ready), name=f"mcp-session:{self.label}"
        )
        try:
            self.session = await asyncio.wait_for(asyncio.shield(ready), connect_timeout)
        except TimeoutError as exc:
            await self._abort(ready)
            msg = f"timed out after {connect_timeout}s connecting to MCP server {self.label}"
            raise MCPConnectionError(msg) from exc
        except BaseException:
            await self._abort(ready)
            raise
        return self.session

    async def _abort(self, ready: asyncio.Future[Any]) -> None:
        if not ready.done():
            ready.cancel()
        await self.close(grace=1.0)

    async def _run(
        self,
        transport: Any,
        session_factory: Callable[[Any, Any], Any],
        ready: asyncio.Future[Any],
    ) -> None:
        try:
            async with AsyncExitStack() as stack:
                streams = await stack.enter_async_context(transport)
                session = await stack.enter_async_context(session_factory(streams[0], streams[1]))
                await session.initialize()
                if not ready.done():
                    ready.set_result(session)
                assert self._stop is not None
                await self._stop.wait()
        except asyncio.CancelledError:
            # Our own close()/loop shutdown, or the transport's task group
            # cancelling its host after a connection failure. Either way the
            # session is finished; callers learn it from ``alive``.
            if not ready.done():
                ready.set_exception(
                    MCPConnectionError(
                        f"MCP server {self.label} is unreachable or closed the "
                        "connection during setup"
                    )
                )
            raise
        except Exception as exc:  # noqa: BLE001 — every transport failure ends the session
            leaf = _unwrap(exc)
            if not ready.done():
                ready.set_exception(leaf)
            else:
                logger.warning(
                    "MCP session %s ended: %s: %s", self.label, type(leaf).__name__, leaf
                )
        finally:
            if not ready.done():
                ready.set_exception(
                    MCPConnectionError(
                        f"MCP server {self.label} closed the connection during setup"
                    )
                )

    async def close(self, grace: float = 5.0) -> None:
        """Ask the runner to exit its contexts; cancel it if it will not
        within ``grace`` seconds."""
        task = self.task
        if task is None or task.done():
            return
        if self._stop is not None:
            self._stop.set()
        done, _ = await asyncio.wait({task}, timeout=grace)
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def _await_while_alive(
    handle: _SessionHandle | None,
    awaitable: Awaitable[Any],
    *,
    liveness_interval: float | None = None,
) -> Any:
    """Await ``awaitable``, failing fast if the connection is gone.

    Two ways a request can otherwise wait out its whole read timeout — by
    default, forever — for a response that will never come:

    * the session's runner dies (the transport's task group failed);
    * the server dies *while streaming this request's response*. The SDK's
      streamable-HTTP transport logs the broken stream and returns, leaving
      the request pending with no error.

    The first is watched directly. For the second, a request pending longer
    than ``liveness_interval`` pings the server each interval; a ping that
    fails means the connection is gone. A ping that merely times out (a busy
    server) is not treated as a failure.
    """
    if handle is None or handle.task is None:
        return await awaitable
    call = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait(
                {call, handle.task},
                timeout=liveness_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if call in done:
                return call.result()
            if handle.task in done:
                break
            ping = getattr(handle.session, "send_ping", None)
            if ping is None:
                continue
            try:
                await asyncio.wait_for(ping(), timeout=liveness_interval)
            except TimeoutError:
                continue
            except Exception:  # noqa: BLE001 — any failure to ping is a lost connection
                if call.done():
                    return call.result()
                break
    except BaseException:
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)
        raise
    call.cancel()
    await asyncio.gather(call, return_exceptions=True)
    msg = f"MCP server {handle.label} closed the connection"
    raise MCPConnectionError(msg)


class _TypedConnectErrors:
    """Turn a failure to reach the server while OPENING a session into
    :class:`MCPConnectionError` — the typed error a failure mid-request
    already is — so an agent run gets a tool error it can report, not the
    transport's own exception (``httpx.ConnectError`` and friends)."""

    def __init__(self, label: str) -> None:
        self.label = label

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        if exc is None or isinstance(exc, MCPConnectionError) or not isinstance(exc, Exception):
            return
        if _is_connection_loss(exc):
            msg = f"MCP server {self.label} is unreachable: {type(_unwrap(exc)).__name__}: {exc}"
            raise MCPConnectionError(msg) from exc


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

    **Rich results.** :meth:`call_tool` returns the text, as it always has.
    :meth:`call_tool_result` returns an :class:`MCPToolResult` with the
    server's ``structuredContent``, ``isError`` and every content block; the
    tools :meth:`to_tulip_tools` builds use it, so an agent's
    ``ToolCompleteEvent`` carries ``structured_content`` and ``error``
    while the model still reads the text.

    **One client, many users.** Headers are resolved per request from, in
    order: :attr:`headers`, :attr:`access_token`, the run metadata key
    :attr:`metadata_headers_key` (``agent.run(..., metadata={"mcp_headers":
    {"Authorization": "Bearer <user token>"}})``) and :attr:`headers_provider`.
    Requests whose per-run headers differ go over separate MCP sessions,
    because an MCP session is stateful and must never be shared between
    identities. :attr:`session_key` keys them by principal instead (a rotated
    token then reuses the principal's session, which sends the new token);
    :attr:`session_idle_ttl` closes idle ones and :attr:`max_sessions` caps
    them (least recently used idle session closed first).
    :attr:`session_stats` counts opens and closes.

    Per-run headers from run metadata are *ephemeral*: the agent carries them
    on the run's in-memory context and never writes them into its state, so a
    bearer token never reaches a checkpoint, an event or a result.

    **Resilience.** Each session runs in a task of its own, so a server that
    dies takes down only its session: the call in flight fails with
    :class:`MCPConnectionError` (a tool error to the agent, never a
    cancellation of the run) and the next call reconnects.
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
            "network errors. Set TULIP_MCP_SKIP_OSV=1 to disable globally."
        ),
    )

    name: str | None = Field(
        default=None, description="Label for logs and errors; defaults to the URL/command."
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom HTTP headers sent on every request (HTTP transport).",
    )
    headers_provider: HeadersProvider | None = Field(
        default=None,
        description=(
            "Called per request with an MCPRequestContext (run metadata, tool "
            "name, call id); returns extra headers — e.g. the end user's "
            "bearer token. Sync or async. HTTP transport only."
        ),
    )
    metadata_headers_key: str | None = Field(
        default="mcp_headers",
        description=(
            "Run-metadata key holding per-run headers: "
            "agent.run(..., metadata={'mcp_headers': {...}}). The agent treats "
            "this key as ephemeral: never persisted in state, checkpoints or "
            "events. None disables."
        ),
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Only these tool names are listed, attached and callable.",
    )
    tool_filter: Callable[[dict[str, Any]], bool] | None = Field(
        default=None,
        description="Predicate over a tool schema dict; False hides and blocks the tool.",
    )
    connect_timeout: float | None = Field(
        default=30.0, description="Seconds to wait for connect + initialize; None waits forever."
    )
    call_timeout: float | None = Field(
        default=None, description="Read timeout (seconds) for one tools/call; None = SDK default."
    )
    max_sessions: int = Field(
        default=16,
        ge=1,
        description=(
            "Per-identity sessions kept open; opening one more closes the "
            "least recently used idle one."
        ),
    )
    session_key: SessionKey | None = Field(
        default=None,
        description=(
            "(MCPRequestContext, headers) -> key choosing the per-identity "
            "session for a request, e.g. the verified principal of a JWT. "
            "Requests with the same key share one session, which always sends "
            "the latest headers (a rotated token does not open a new session). "
            "Must identify the principal: two users must never map to one key. "
            "None (default) or a None result keys by the full headers."
        ),
    )
    session_idle_ttl: float | None = Field(
        default=300.0,
        gt=0,
        description=(
            "Close a per-identity session unused for this many seconds. "
            "None keeps sessions until evicted by max_sessions or close()."
        ),
    )
    reconnect_interval: float = Field(
        default=30.0,
        ge=0,
        description=(
            "An agent retries attaching this server's tools at most this often "
            "(seconds) after a failed attach."
        ),
    )
    liveness_interval: float | None = Field(
        default=15.0,
        gt=0,
        description=(
            "While a request has been pending this long, ping the server every "
            "interval; a failed ping fails the request with MCPConnectionError "
            "instead of leaving it waiting on a server that died mid-response. "
            "None disables."
        ),
    )
    embed_images: bool = Field(
        default=True,
        description=(
            "Embed image content in the tool result as Tulip image segments, "
            "so multimodal adapters show the model the image."
        ),
    )

    _session: Any = None
    _client_context: Any = None
    _connected: bool = False
    _process: Any = None
    _runner: _SessionHandle | None = None
    _ever_connected: bool = False
    _identity_sessions: OrderedDict[Hashable, _SessionHandle] = PrivateAttr(
        default_factory=OrderedDict
    )
    _identity_locks: dict[Hashable, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _sweeper: asyncio.Task[None] | None = PrivateAttr(default=None)
    _opened_sessions: int = PrivateAttr(default=0)
    _closed_sessions: int = PrivateAttr(default=0)
    _schemas: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def label(self) -> str:
        """How this server is named in logs and errors."""
        if self.name:
            return self.name
        if self.base_url:
            return self.base_url
        if self.server_command:
            return self.server_command[0]
        return type(self).__name__

    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self._runner is not None and not self._runner.alive:
            await self._discard(self._runner)  # the last connection died
        if self._connected:
            return

        if self.base_url:
            await self._connect_http()
        elif self.server_command:
            await self._connect_stdio()
        else:
            raise ValueError("Must provide either base_url or server_command")

        self._connected = True
        self._ever_connected = True

    async def _connect_http(
        self,
        extra_headers: Mapping[str, str] | None = None,
        *,
        install: bool = True,
    ) -> _SessionHandle:
        """Connect via HTTP/SSE transport.

        Args:
            extra_headers: Per-identity headers layered over :attr:`headers`.
            install: Make this the default session (``connect()``); False
                opens a per-identity session and leaves the default alone.
        """
        if self.base_url is None:
            msg = "_connect_http called without base_url"
            raise RuntimeError(msg)
        try:
            from mcp.client import streamable_http as _streamable_http
            from mcp.client.session import ClientSession
        except ImportError as e:
            raise ImportError(
                "mcp package required for HTTP transport. Install with: pip install mcp"
            ) from e

        # Pre-flight SSRF guard. Rejecting a model-supplied base_url that
        # resolves to a cloud-metadata endpoint or a private network is
        # the one check we can do cheaply before any bytes go on the wire.
        # Redirect-based bypass is a known limitation — see url_safety.py.
        if self.verify_url and self.base_url:
            from tulip.tools.url_safety import validate_url

            validate_url(self.base_url, allow_private=self.allow_private_url)

        request_headers = {**self.headers, **(extra_headers or {})}
        has_authorization = any(k.lower() == "authorization" for k in request_headers)
        handle = _SessionHandle(self.label, dict(extra_headers or {}))
        static_headers = dict(self.headers)
        opened_with = set(extra_headers or {})

        # Set up auth if token provided. A per-request Authorization header
        # (headers / headers_provider / run metadata) wins over the static
        # token — BearerAuth would otherwise overwrite it on the wire.
        auth = None
        if self.access_token and not has_authorization:
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

        async def _current_headers(request: httpx.Request) -> None:
            # The session's CURRENT per-identity headers win over whatever
            # the transport stamped at open time: a rotated token is sent as
            # soon as it is known, never the one the session opened with.
            live = handle.headers
            for name in opened_with - set(live):
                if name in static_headers:
                    request.headers[name] = static_headers[name]
                elif name in request.headers:
                    del request.headers[name]
            for name, value in live.items():
                request.headers[name] = value

        def _httpx_factory(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                headers={**request_headers, **(headers or {})},
                timeout=timeout if timeout is not None else httpx.Timeout(30.0),
                auth=auth,
                verify=verify_ssl,
                follow_redirects=True,
                event_hooks={"request": [_current_headers]} if extra_headers else None,
            )

        # mcp renamed ``streamablehttp_client`` → ``streamable_http_client``
        # and changed its signature (a ready ``http_client`` replaces the
        # ``auth``/``httpx_client_factory`` pair). Branch on what the installed
        # version offers so both the pinned floor and the latest work; either
        # way the client carries our auth, TLS-verify and redirect settings.
        _legacy_client = getattr(_streamable_http, "streamablehttp_client", None)
        if _legacy_client is not None:
            legacy_kwargs: dict[str, Any] = {
                "auth": auth,
                "httpx_client_factory": _httpx_factory,
            }
            if request_headers:
                legacy_kwargs["headers"] = request_headers
            client_context = _legacy_client(self.base_url, **legacy_kwargs)
        else:
            # ``getattr`` on purpose: the renamed API is typed against the
            # httpx fork mcp vendors (httpx2), which duck-types with the
            # client our factory builds — a static call here would pin us to
            # whichever fork the installed mcp declares.
            _new_client = getattr(_streamable_http, "streamable_http_client")  # noqa: B009
            client_context = _new_client(
                self.base_url,
                http_client=_httpx_factory(auth=auth),
            )

        await handle.start(client_context, ClientSession, self.connect_timeout)
        if install:
            self._install(handle)
        return handle

    async def _connect_stdio(self) -> _SessionHandle:
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
        # Fails open on any lookup issue; see tulip.integrations.osv for
        # the full behaviour contract.
        if self.verify_packages:
            from tulip.core.errors import ValidationError
            from tulip.integrations.osv import check_package_for_malware

            reason = check_package_for_malware(cmd, cmd_args)
            if reason:
                raise ValidationError(f"MCP launch blocked: {reason}")

        server_params = StdioServerParameters(
            command=cmd,
            args=cmd_args,
        )

        handle = _SessionHandle(self.label)
        await handle.start(stdio_client(server_params), ClientSession, self.connect_timeout)
        self._install(handle)
        return handle

    def _install(self, handle: _SessionHandle) -> None:
        """Make ``handle`` the default session."""
        self._runner = handle
        self._session = handle.session
        self._client_context = handle.transport

    async def _discard(self, handle: _SessionHandle | None) -> None:
        """Forget a dead session so the next request reconnects."""
        if handle is None or handle is self._runner:
            self._runner = None
            self._session = None
            self._client_context = None
            self._connected = False
        else:
            for key, candidate in list(self._identity_sessions.items()):
                if candidate is handle:
                    await self._close_identity(key, "connection lost", grace=1.0)
                    return
        if handle is not None:
            await handle.close(grace=1.0)

    async def close(self) -> None:
        """Close the connection — the default session and every per-identity one."""
        self._connected = False
        self._ever_connected = False

        sweeper, self._sweeper = self._sweeper, None
        if sweeper is not None and not sweeper.done() and sweeper is not asyncio.current_task():
            sweeper.cancel()
            await asyncio.gather(sweeper, return_exceptions=True)
        for key in list(self._identity_sessions):
            await self._close_identity(key, "client closed")

        runner = self._runner
        if runner is not None:
            self._runner = None
            await runner.close()
            self._session = None
            self._client_context = None
            return

        # No runner: state was installed directly (tests, subclasses). Exit
        # the contexts in place, as before.
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

    # ------------------------------------------------------------------
    # Per-request identity
    # ------------------------------------------------------------------

    def _request_context(self, tool_name: str | None) -> MCPRequestContext:
        """The context a headers provider sees for a request made now."""
        ctx = current_tool_context()
        if ctx is not None and tool_name is not None:
            # Per-run headers travel as the run's ephemeral metadata (never
            # persisted); providers see them merged back into ``metadata``.
            ephemeral = getattr(ctx, "ephemeral_metadata", None)
            return MCPRequestContext(
                metadata=(
                    {**ctx.invocation_metadata, **ephemeral}
                    if ephemeral
                    else ctx.invocation_metadata
                ),
                run_id=ctx.run_id or None,
                tool_name=tool_name,
                tool_call_id=ctx.tool_call_id,
            )
        return MCPRequestContext(metadata=_request_metadata.get() or {}, tool_name=tool_name)

    async def _dynamic_headers(self, rctx: MCPRequestContext) -> dict[str, str]:
        """Per-request headers from run metadata and the headers provider."""
        if not self.base_url:
            return {}
        resolved: dict[str, str] = {}
        if self.metadata_headers_key:
            from_metadata = rctx.metadata.get(self.metadata_headers_key)
            if isinstance(from_metadata, Mapping):
                resolved.update({str(k): str(v) for k, v in from_metadata.items()})
        if self.headers_provider is not None:
            provided = self.headers_provider(rctx)
            if inspect.isawaitable(provided):
                provided = await provided
            if provided:
                resolved.update({str(k): str(v) for k, v in provided.items()})
        return resolved

    async def _session_for(self, rctx: MCPRequestContext) -> tuple[Any, _SessionHandle | None]:
        """The session to use for a request — per identity when headers vary.

        A server that cannot be reached raises :class:`MCPConnectionError`,
        like a connection lost mid-request, never the transport's own error.
        """
        dynamic = await self._dynamic_headers(rctx)
        if dynamic:
            return await self._identity_session(dynamic, rctx)

        if self._runner is not None and not self._runner.alive:
            await self._discard(self._runner)
        if self._session is None:
            if not self._ever_connected:
                raise RuntimeError("Not connected. Call connect() first.")
            # The server went away since the last call; try it again.
            async with self._typed_connect_errors():
                await self.connect()
        return self._session, self._runner

    def _typed_connect_errors(self) -> _TypedConnectErrors:
        return _TypedConnectErrors(self.label)

    async def _session_key(self, rctx: MCPRequestContext, dynamic: dict[str, str]) -> Hashable:
        if self.session_key is not None:
            custom: Any = self.session_key(rctx, dynamic)
            if inspect.isawaitable(custom):
                custom = await custom
            if custom is not None:
                return ("key", custom)
        return ("headers", tuple(sorted(dynamic.items())))

    async def _identity_session(
        self, dynamic: dict[str, str], rctx: MCPRequestContext | None = None
    ) -> tuple[Any, _SessionHandle]:
        key = await self._session_key(rctx or MCPRequestContext(), dynamic)
        await self._evict_idle()
        lock = self._identity_locks.setdefault(key, asyncio.Lock())
        async with lock:
            handle = self._identity_sessions.get(key)
            if handle is not None and handle.alive:
                if handle.headers != dynamic:
                    # Same session key, new headers (a rotated token): the
                    # session sends these from its next request on.
                    handle.headers.clear()
                    handle.headers.update(dynamic)
                handle.last_used = time.monotonic()
                self._identity_sessions.move_to_end(key)
                return handle.session, handle
            if handle is not None:
                await self._close_identity(key, "connection lost", grace=1.0)
            async with self._typed_connect_errors():
                handle = await self._connect_http(dynamic, install=False)
            self._identity_locks.setdefault(key, lock)
            self._identity_sessions[key] = handle
            self._opened_sessions += 1
            logger.info(
                "MCP session opened on %s (live=%d, opened=%d, closed=%d)",
                self.label,
                len(self._identity_sessions),
                self._opened_sessions,
                self._closed_sessions,
            )
            await self._enforce_max_sessions(keep=key)
            self._ensure_sweeper()
            return handle.session, handle

    async def _close_identity(self, key: Hashable, reason: str, *, grace: float = 5.0) -> None:
        """Forget and close one per-identity session, counting it."""
        handle = self._identity_sessions.pop(key, None)
        lock = self._identity_locks.get(key)
        if lock is not None and not lock.locked():
            del self._identity_locks[key]
        if handle is None:
            return
        self._closed_sessions += 1
        logger.info(
            "MCP session closed on %s: %s (live=%d, opened=%d, closed=%d)",
            self.label,
            reason,
            len(self._identity_sessions),
            self._opened_sessions,
            self._closed_sessions,
        )
        await handle.close(grace=grace)

    async def _enforce_max_sessions(self, *, keep: Hashable) -> None:
        """Close least-recently-used sessions beyond :attr:`max_sessions`,
        preferring ones with no request in flight."""
        while len(self._identity_sessions) > self.max_sessions:
            candidates = [k for k in self._identity_sessions if k != keep]
            idle = [k for k in candidates if self._identity_sessions[k].in_flight == 0]
            victim = (idle or candidates)[0]
            await self._close_identity(victim, "max_sessions reached")

    async def _evict_idle(self) -> None:
        """Close per-identity sessions idle longer than :attr:`session_idle_ttl`."""
        ttl = self.session_idle_ttl
        if ttl is None:
            return
        now = time.monotonic()
        for key, handle in list(self._identity_sessions.items()):
            if handle.in_flight == 0 and now - handle.last_used >= ttl:
                await self._close_identity(key, f"idle for {ttl:g}s")
            elif not handle.alive and handle.in_flight == 0:
                await self._close_identity(key, "connection lost", grace=1.0)

    def _ensure_sweeper(self) -> None:
        """Run the idle sweep in the background while sessions are open, so
        an idle session is closed even when no further request arrives."""
        if self.session_idle_ttl is None:
            return
        sweeper = self._sweeper
        loop = asyncio.get_running_loop()
        if sweeper is not None and not sweeper.done() and sweeper.get_loop() is loop:
            return
        self._sweeper = loop.create_task(self._sweep(), name=f"mcp-session-sweeper:{self.label}")

    async def _sweep(self) -> None:
        while self._identity_sessions and self.session_idle_ttl is not None:
            await asyncio.sleep(max(self.session_idle_ttl / 2, 0.05))
            await self._evict_idle()

    @property
    def session_stats(self) -> dict[str, int]:
        """Per-identity session accounting: ``opened`` and ``closed`` since
        construction, and ``live`` now. Plain ints, ready for a metrics gauge."""
        return {
            "opened": self._opened_sessions,
            "closed": self._closed_sessions,
            "live": len(self._identity_sessions),
        }

    async def _guarded(
        self, handle: _SessionHandle | None, awaitable: Awaitable[Any], what: str
    ) -> Any:
        """Run one request; a dropped connection becomes :class:`MCPConnectionError`."""
        if handle is not None:
            handle.in_flight += 1
        try:
            return await _await_while_alive(
                handle, awaitable, liveness_interval=self.liveness_interval
            )
        except asyncio.CancelledError:
            if _being_cancelled():
                raise  # the caller really is being cancelled
            await self._discard(handle)
            msg = f"MCP connection to {self.label} was lost during {what}"
            raise MCPConnectionError(msg) from None
        except MCPConnectionError:
            await self._discard(handle)
            raise
        except Exception as exc:
            if _is_connection_loss(exc) or (handle is not None and not handle.alive):
                await self._discard(handle)
                msg = f"MCP connection to {self.label} was lost during {what}: {exc}"
                raise MCPConnectionError(msg) from exc
            raise
        finally:
            if handle is not None:
                handle.in_flight -= 1
                handle.last_used = time.monotonic()

    # ------------------------------------------------------------------
    # Tool filtering
    # ------------------------------------------------------------------

    def _is_allowed(self, schema: Mapping[str, Any]) -> bool:
        name = schema.get("name")
        if self.allowed_tools is not None and name not in self.allowed_tools:
            return False
        return self.tool_filter is None or bool(self.tool_filter(dict(schema)))

    def _ensure_allowed(self, name: str) -> None:
        if self.allowed_tools is None and self.tool_filter is None:
            return
        if not self._is_allowed(self._schemas.get(name, {"name": name})):
            msg = f"MCP tool {name!r} is not allowed on {self.label}"
            raise MCPToolNotAllowedError(msg)

    # ------------------------------------------------------------------
    # Protocol calls
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from the MCP server.

        Tools excluded by :attr:`allowed_tools` / :attr:`tool_filter` are
        left out. Each dict carries ``name``, ``description`` and
        ``inputSchema``, plus ``outputSchema``, ``title`` and
        ``annotations`` when the server declares them.
        """
        session, handle = await self._session_for(self._request_context(None))
        result = await self._guarded(handle, session.list_tools(), "tools/list")

        # Convert MCP Tool objects to dicts
        tools = []
        for mcp_tool in result.tools:
            schema: dict[str, Any] = {
                "name": mcp_tool.name,
                "description": mcp_tool.description or "",
                "inputSchema": _mcp_field(mcp_tool, "inputSchema", "input_schema", default={}),
            }
            output_schema = _mcp_field(mcp_tool, "outputSchema", "output_schema")
            if isinstance(output_schema, dict):
                schema["outputSchema"] = output_schema
            title = getattr(mcp_tool, "title", None)
            if isinstance(title, str) and title:
                schema["title"] = title
            annotations = _dump_block(getattr(mcp_tool, "annotations", None))
            if annotations:
                schema["annotations"] = annotations
            if not self._is_allowed(schema):
                continue
            self._schemas[schema["name"]] = schema
            tools.append(schema)
        return tools

    async def call_tool_result(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        progress_callback: Callable[[float, float | None, str | None], Any] | None = None,
    ) -> MCPToolResult:
        """Call a tool and return everything the server sent back.

        Args:
            name: Tool name.
            arguments: Tool arguments.
            progress_callback: Called ``(progress, total, message)`` for each
                progress notification (may be sync or async). Defaults to the
                running agent's progress stream when called from a tool.

        Raises:
            MCPToolNotAllowedError: ``name`` is excluded by the allowlist/filter.
            MCPConnectionError: The server could not be reached or dropped the
                connection mid-call (the next call reconnects).
        """
        self._ensure_allowed(name)
        session, handle = await self._session_for(self._request_context(name))

        call = session.call_tool
        kwargs: dict[str, Any] = {}
        if self.call_timeout is not None and _accepts_kwarg(call, "read_timeout_seconds"):
            kwargs["read_timeout_seconds"] = timedelta(seconds=self.call_timeout)

        on_progress: Callable[[float, float | None, str | None], Any] | None = progress_callback
        if on_progress is None:
            on_progress = progress_reporter()
        if on_progress is not None and _accepts_kwarg(call, "progress_callback"):
            report = on_progress

            async def _progress(
                progress: float, total: float | None, message: str | None = None
            ) -> None:
                try:
                    outcome = report(progress, total, message)
                    if inspect.isawaitable(outcome):
                        await outcome
                except Exception:  # noqa: BLE001 — a bad observer must not fail the call
                    logger.debug("MCP progress callback failed", exc_info=True)

            kwargs["progress_callback"] = _progress

        raw = await self._guarded(
            handle, call(name=name, arguments=arguments, **kwargs), f"tools/call {name}"
        )
        return _convert_call_result(raw, embed_images=self.embed_images)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server and return its text.

        Kept for compatibility; :meth:`call_tool_result` also returns
        ``structuredContent``, ``isError`` and non-text content.
        """
        return (await self.call_tool_result(name, arguments)).text

    # ------------------------------------------------------------------
    # Agent integration
    # ------------------------------------------------------------------

    async def load_tools(self, metadata: Mapping[str, Any] | None = None) -> list[Tool]:
        """Connect if needed, list the tools and convert them — what an agent
        does on attach.

        Args:
            metadata: The run metadata, visible to per-run headers
                (:attr:`metadata_headers_key`, :attr:`headers_provider`) for
                the ``tools/list`` request.
        """
        token = _request_metadata.set(dict(metadata or {}))
        try:
            if not self._connected:
                dynamic = await self._dynamic_headers(self._request_context(None))
                if not dynamic:
                    # Per-identity requests open their own sessions; the
                    # shared default one is only needed without them.
                    await self.connect()
            schemas = await self.list_tools()
            return self.to_tulip_tools(schemas)
        finally:
            _request_metadata.reset(token)

    def to_tulip_tools(self, tools: list[dict[str, Any]]) -> list[Tool]:
        """Convert MCP tool schemas into Tulip tools bound to this client.

        Each tool returns a :class:`~tulip.tools.output.ToolOutput`: the
        model reads the text; ``structuredContent``, ``isError`` and non-text
        blocks reach the agent's ``ToolCompleteEvent``. Progress
        notifications stream as ``ToolProgressEvent`` s.

        Args:
            tools: Schemas as returned by :meth:`list_tools`.

        Returns:
            One :class:`~tulip.tools.decorator.Tool` per allowed schema, each
            calling back through this client.
        """
        tulip_tools = []
        for mcp_tool in tools:
            if not self._is_allowed(mcp_tool):
                continue

            def make_func(name: str = mcp_tool["name"]) -> Callable[..., Any]:
                """Build the coroutine that calls one MCP tool.

                The default argument binds ``name`` per iteration; a bare
                closure over the loop variable would give every tool the last
                name in the list.

                Deliberately a plain ``def``. This was an ``async def``
                unwrapped with
                ``asyncio.get_event_loop().run_until_complete(...)``, which
                raises ``RuntimeError: This event loop is already running``
                inside any running loop — and a running loop is the only way
                to reach here, since ``await client.connect()`` comes first.
                Building a closure never needed the event loop at all.
                """

                async def func(**kwargs: Any) -> str:
                    # A subclass (or patch) that overrides ``call_tool`` keeps
                    # its routing; everyone else gets the rich result.
                    if getattr(type(self), "call_tool", None) is not _BASE_CALL_TOOL:
                        return await self.call_tool(name, kwargs)
                    result = await self.call_tool_result(name, kwargs)
                    return result.to_tool_output()

                func.__name__ = name
                return func

            tulip_tool = mcp_tool_to_tulip(
                name=mcp_tool["name"],
                description=mcp_tool.get("description", ""),
                func=make_func(),
                parameters=mcp_tool.get("inputSchema"),
                output_schema=mcp_tool.get("outputSchema"),
                emits_progress=True,
            )
            tulip_tools.append(tulip_tool)

        return tulip_tools


_BASE_CALL_TOOL = MCPClient.call_tool
