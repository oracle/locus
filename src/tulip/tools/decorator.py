# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Tool decorator for Tulip - 100% Pydantic."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, overload

from pydantic import BaseModel

from tulip.tools.context import ToolContext
from tulip.tools.schema import generate_schema


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
        # Check if function accepts context
        sig = inspect.signature(self.fn)
        accepts_ctx = any(name in ("ctx", "context") for name in sig.parameters)

        if accepts_ctx and ctx is not None:
            # Find the context parameter name
            ctx_param = next(name for name in sig.parameters if name in ("ctx", "context"))
            kwargs[ctx_param] = ctx

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
        """Direct invocation of the tool."""
        return self.fn(*args, **kwargs)


@overload
def tool(fn: Callable[P, R]) -> Tool: ...


@overload
def tool(
    fn: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    idempotent: bool = False,
) -> Callable[[Callable[P, R]], Tool]: ...


def tool(
    fn: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    idempotent: bool = False,
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
        def book_flight(flight_id: str, customer_id: str) -> dict:
            '''Book a flight — safe to mark idempotent because repeated
            calls with the same flight/customer would create duplicate
            bookings, which we never want.'''
            ...

    Args:
        fn: The function to wrap
        name: Override tool name (defaults to function name)
        description: Override description (defaults to docstring)
        idempotent: If True, the ReAct loop deduplicates calls with
            matching (name, arguments) within a single agent run. Prevents
            duplicate side-effects when a model re-issues a tool call it
            has already made this turn.

    Returns:
        Tool instance
    """

    def decorator(func: Callable[P, R]) -> Tool:
        # Generate schema
        schema = generate_schema(func, description)
        func_schema = schema["function"]

        return Tool(
            name=name or func_schema["name"],
            description=func_schema["description"],
            parameters=func_schema["parameters"],
            fn=func,
            idempotent=idempotent,
        )

    if fn is not None:
        # Called without arguments: @tool
        return decorator(fn)

    # Called with arguments: @tool(name="...")
    return decorator
