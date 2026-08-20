# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tool decorator for Tulip - 100% Pydantic."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from pydantic import BaseModel

from tulip.observability.emit import (
    EV_TOOL_SANDBOX_COMPLETED,
    EV_TOOL_SANDBOX_STARTED,
    emit,
    emit_sync,
)
from tulip.tools.context import ToolContext
from tulip.tools.sandbox import (
    SandboxExecutionError,
    SandboxSpec,
    ToolSandbox,
    failure_message,
    normalize_sandbox,
    provider_label,
    run_tool_sandboxed,
    validate_sandboxable,
)
from tulip.tools.schema import generate_schema


if TYPE_CHECKING:
    from collections.abc import Iterable


P = ParamSpec("P")
R = TypeVar("R")


class Tool(BaseModel):
    """
    A tool that can be called by agents.

    Created via the @tool decorator.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    idempotent: bool = False
    """When True, the ReAct loop deduplicates calls: if the model emits the
    same (tool_name, arguments) combination that has already been executed
    earlier in the current agent run, the prior result is reused and the
    tool function is not invoked again. Use for tools that either have
    side-effects you don't want duplicated (bookings, transfers, writes) or
    whose output is stable across the run (config/date lookups)."""

    labels: frozenset[str] = frozenset()
    """Policy-matching labels this tool declares (e.g. ``{"code-exec"}``).
    Matched against :class:`~tulip.security.policy.ControlPolicy` label sets
    such as ``require_sandbox_for`` by governance hooks like
    :class:`~tulip.tools.sandbox.SandboxEnforcerHook`."""

    sandbox: SandboxSpec | None = None
    """When set, :meth:`execute` ships the function's *source* into the
    configured sandbox and runs it there — the host process never executes
    the body. The function must be self-contained: synchronous, imports
    inside its body, JSON-serializable arguments and return value, and no
    ``ctx``/``context`` parameter. See :mod:`tulip.tools.sandbox`."""

    model_config = {"arbitrary_types_allowed": True}

    @property
    def func(self) -> Callable[..., Any]:
        """Alias for :attr:`fn`. Some samples and downstream code reach
        for ``.func`` (the LangChain/LangGraph idiom); keep both names
        pointed at the same underlying callable so users don't have to
        write ``getattr(t, 'fn', None) or getattr(t, 'func', t)``."""
        return self.fn

    async def execute(self, ctx: ToolContext | None = None, **kwargs: Any) -> Any:
        """
        Execute the tool with given arguments.

        Args:
            ctx: Optional tool context (injected if function accepts it)
            **kwargs: Tool arguments

        Returns:
            Tool result
        """
        if self.sandbox is not None:
            return await self._execute_sandboxed(self.sandbox, kwargs)

        # Check if function accepts context
        sig = inspect.signature(self.fn)
        accepts_ctx = any(name in ("ctx", "context") for name in sig.parameters)

        if accepts_ctx and ctx is not None:
            # Find the context parameter name
            ctx_param = next(name for name in sig.parameters if name in ("ctx", "context"))
            kwargs[ctx_param] = ctx

        # Bind before calling, so a caller mistake is distinguishable from a
        # bug inside the tool. Both raise TypeError, but only one is worth
        # telling a model about — and "search() missing 1 required positional
        # argument: 'title'" is a Python signature error, not an instruction it
        # can act on. Naming the tool and the missing parameters is.
        try:
            sig.bind(**kwargs)
        except TypeError as exc:
            missing = [
                name
                for name, param in sig.parameters.items()
                if param.default is inspect.Parameter.empty
                and param.kind
                not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                and name not in kwargs
                and name not in ("ctx", "context")
            ]
            detail = f"missing required argument(s): {', '.join(missing)}" if missing else str(exc)
            raise TypeError(
                f"{self.name} was called with the wrong arguments — {detail}. "
                f"Call {self.name} again with every required argument filled in; "
                f"if a value is unknown, ask for it rather than guessing."
            ) from exc

        # Execute function
        if asyncio.iscoroutinefunction(self.fn):
            result = await self.fn(**kwargs)
        else:
            # Run sync function in thread pool. Propagate the current
            # contextvars context so observability emits (run_id) and
            # any other contextvar-driven instrumentation see the same
            # state inside the worker thread.
            import contextvars  # noqa: PLC0415

            loop = asyncio.get_event_loop()
            ctxvars_snapshot = contextvars.copy_context()
            result = await loop.run_in_executor(
                None,
                lambda: ctxvars_snapshot.run(self.fn, **kwargs),
            )

        return self._format_result(result)

    async def _execute_sandboxed(self, spec: SandboxSpec, kwargs: dict[str, Any]) -> str:
        """Run the tool's source in the configured sandbox (worker thread —
        providers are blocking) and format the recovered value."""
        await emit(
            EV_TOOL_SANDBOX_STARTED,
            tool=self.name,
            provider=provider_label(spec),
            timeout=spec.timeout,
        )
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_tool_sandboxed(spec, self.fn, kwargs),
        )
        await emit(
            EV_TOOL_SANDBOX_COMPLETED,
            tool=self.name,
            ok=result.ok,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            duration_ms=result.duration_ms,
        )
        if not result.ok:
            raise SandboxExecutionError(failure_message(self.name, result), result=result)
        return self._format_result(result.value)

    def _format_result(self, result: Any) -> str:
        """Format tool result as string for LLM."""
        if result is None:
            return "Success (no output)"

        if isinstance(result, str):
            return result

        if isinstance(result, BaseModel):
            return result.model_dump_json()

        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, default=str)

        return str(result)

    def to_openai_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Direct invocation of the tool.

        A sandboxed tool is sandboxed here too — direct calls are not a
        bypass. The recovered (JSON round-tripped) value is returned raw,
        mirroring a plain function call.
        """
        if self.sandbox is None:
            return self.fn(*args, **kwargs)
        bound = inspect.signature(self.fn).bind(*args, **kwargs)
        spec = self.sandbox
        emit_sync(
            EV_TOOL_SANDBOX_STARTED,
            tool=self.name,
            provider=provider_label(spec),
            timeout=spec.timeout,
        )
        result = run_tool_sandboxed(spec, self.fn, dict(bound.arguments))
        emit_sync(
            EV_TOOL_SANDBOX_COMPLETED,
            tool=self.name,
            ok=result.ok,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            duration_ms=result.duration_ms,
        )
        if not result.ok:
            raise SandboxExecutionError(failure_message(self.name, result), result=result)
        return result.value


@overload
def tool(fn: Callable[P, R]) -> Tool: ...


@overload
def tool(
    fn: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    idempotent: bool = False,
    labels: Iterable[str] | None = None,
    sandbox: SandboxSpec | ToolSandbox | str | bool | None = None,
) -> Callable[[Callable[P, R]], Tool]: ...


def tool(
    fn: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    idempotent: bool = False,
    labels: Iterable[str] | None = None,
    sandbox: SandboxSpec | ToolSandbox | str | bool | None = None,
) -> Tool | Callable[[Callable[P, R]], Tool]:
    """
    Decorator to create a tool from a function.

    Usage:
        @tool
        def search(query: str) -> str:
            '''Search the knowledge base.'''
            return "results..."

        @tool(name="custom_name", description="Custom description")
        def my_tool(x: int) -> int:
            return x * 2

        @tool(idempotent=True)
        def open_incident(alert_id: str, severity: str) -> dict:
            '''Open an incident — safe to mark idempotent because repeated
            calls for the same alert would open duplicate incidents,
            which we never want.'''
            ...

        @tool(sandbox=True, labels={"code-exec"})
        def run_snippet(source: str) -> dict:
            '''Evaluate a snippet — runs in an isolated box, never in the
            host process. Self-contained: imports live inside the body.'''
            import ast
            return {"parsed": ast.dump(ast.parse(source))}

    Args:
        fn: The function to wrap
        name: Override tool name (defaults to function name)
        description: Override description (defaults to docstring)
        idempotent: If True, the ReAct loop deduplicates calls with
            matching (name, arguments) within a single agent run. Prevents
            duplicate side-effects when a model re-issues a tool call it
            has already made this turn.
        labels: Policy-matching labels the tool declares; matched against
            :class:`~tulip.security.policy.ControlPolicy` label sets (e.g.
            ``require_sandbox_for``).
        sandbox: Run the tool in an isolated box instead of the host
            process. ``True`` uses the default provider (``$TULIP_SANDBOX``
            or the built-in subprocess box); a name, provider object, or
            :class:`~tulip.tools.sandbox.SandboxSpec` selects/configures
            one. The function must be synchronous and self-contained; this
            is validated at decoration time. See :mod:`tulip.tools.sandbox`.

    Returns:
        Tool instance
    """

    def decorator(func: Callable[P, R]) -> Tool:
        spec = normalize_sandbox(sandbox)
        if spec is not None:
            validate_sandboxable(func)

        # Generate schema
        schema = generate_schema(func, description)
        func_schema = schema["function"]

        return Tool(
            name=name or func_schema["name"],
            description=func_schema["description"],
            parameters=func_schema["parameters"],
            fn=func,
            idempotent=idempotent,
            labels=frozenset(labels or ()),
            sandbox=spec,
        )

    if fn is not None:
        # Called without arguments: @tool
        return decorator(fn)

    # Called with arguments: @tool(name="...")
    return decorator
