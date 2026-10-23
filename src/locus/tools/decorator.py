# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tool decorator for Locus - 100% Pydantic."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, get_type_hints, overload

from pydantic import BaseModel, TypeAdapter, ValidationError

from locus.observability.emit import (
    EV_TOOL_SANDBOX_COMPLETED,
    EV_TOOL_SANDBOX_STARTED,
    emit,
    emit_sync,
)
from locus.tools.context import ToolContext
from locus.tools.sandbox import (
    SandboxExecutionError,
    SandboxSpec,
    ToolSandbox,
    failure_message,
    normalize_sandbox,
    provider_label,
    run_tool_sandboxed,
    validate_sandboxable,
)
from locus.tools.schema import generate_schema


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
    Matched against :class:`~locus.control.policy.ControlPolicy` label sets
    such as ``require_sandbox_for`` by governance hooks like
    :class:`~locus.tools.sandbox.SandboxEnforcerHook`."""

    sandbox: SandboxSpec | None = None
    """When set, :meth:`execute` ships the function's *source* into the
    configured sandbox and runs it there — the host process never executes
    the body. The function must be self-contained: synchronous, imports
    inside its body, JSON-serializable arguments and return value, and no
    ``ctx``/``context`` parameter. See :mod:`locus.tools.sandbox`."""

    deferred: bool = False
    """When True, the tool's schema is NOT sent to the model until it is
    activated — the model reaches it through the ``tool_search`` builtin
    (auto-registered when any deferred tool exists). Deferral is a context
    economy: only visibility changes. The tool is registered, gated, and
    policy-matched exactly as an eager tool; activating it adds its schema
    to the next model call, nothing more (#177)."""

    native: dict[str, dict[str, Any]] = {}
    """Provider-native definitions of this tool, keyed by adapter
    (``"anthropic"``, ``"openai_responses"``). An adapter that finds its key
    sends that definition instead of the function schema; the others keep
    using :attr:`parameters`. Keys starting with ``_`` are directives to the
    adapter (a beta header, a request option) and are never sent as part of
    the tool. See :mod:`locus.tools.computer`."""

    emits_progress: bool = False
    """When True, the agent runtime streams progress this tool reports
    (:func:`locus.tools.context.report_progress`, or an MCP server's
    ``notifications/progress``) as live
    :class:`~locus.core.events.ToolProgressEvent` s while the call runs.
    Tools created from an MCP server set it."""

    output_schema: dict[str, Any] | None = None
    """JSON Schema of the structured content the tool returns, when it
    declares one (an MCP tool's ``outputSchema``). Informational: it is not
    sent to the model and does not change how the result is handled."""

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
        sig = inspect.signature(self.fn)
        kwargs = self._coerce_arguments(sig, kwargs)

        if self.sandbox is not None:
            return await self._execute_sandboxed(self.sandbox, kwargs)

        # Check if function accepts context
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

    def _coerce_arguments(self, sig: inspect.Signature, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Coerce model-provided arguments to the declared parameter types.

        Models do not reliably honour the JSON types in a tool schema:
        Llama on OCI sends ``{"a": "17", "b": "25"}`` for two ``int``
        parameters, and ``add("17", "25")`` is ``"1725"``. Each argument
        is validated against the parameter's annotation with pydantic in
        lax mode (``"17"`` → ``17``, ``"2.5"`` → ``2.5``, ``"true"`` →
        ``True``, a dict → the declared ``BaseModel``); a string that fails
        is parsed as JSON once (``"[1, 2]"`` → ``[1, 2]``) and validated
        again; a number handed to a ``str`` parameter is stringified. An
        argument that still does not validate is passed through unchanged
        so a permissive tool keeps working as before. Unannotated
        parameters, ``Any`` and the ``ctx``/``context`` slot are left
        alone. The input dict is not mutated.
        """
        try:
            hints = get_type_hints(self.fn)
        except Exception:  # noqa: BLE001 - unresolved forward refs: nothing to coerce against
            return kwargs
        out = dict(kwargs)
        for name, value in kwargs.items():
            if name in ("ctx", "context") or name not in sig.parameters:
                continue
            hint = hints.get(name)
            if hint is None or hint is Any:
                continue
            if hint is str:
                if isinstance(value, bool | int | float) and not isinstance(value, str):
                    out[name] = str(value)
                continue
            try:
                adapter: TypeAdapter[Any] = TypeAdapter(hint)
            except Exception:  # noqa: BLE001 - exotic annotation pydantic can't model
                continue
            try:
                out[name] = adapter.validate_python(value)
                continue
            except ValidationError:
                pass
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    continue
                try:
                    out[name] = adapter.validate_python(parsed)
                except ValidationError:
                    pass
        return out

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
        """Format tool result as string for LLM.

        A :class:`~locus.tools.output.ToolOutput` is a ``str`` and passes
        through untouched, so its structured content reaches the executor.
        """
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
        """Get OpenAI-compatible tool schema.

        A tool with :attr:`native` definitions carries them under a top-level
        ``native`` key for the adapters that understand it.
        """
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
        if self.native:
            schema["native"] = self.native
        return schema

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
    deferred: bool = False,
    emits_progress: bool = False,
) -> Callable[[Callable[P, R]], Tool]: ...


def tool(
    fn: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    idempotent: bool = False,
    labels: Iterable[str] | None = None,
    sandbox: SandboxSpec | ToolSandbox | str | bool | None = None,
    deferred: bool = False,
    emits_progress: bool = False,
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
            :class:`~locus.control.policy.ControlPolicy` label sets (e.g.
            ``require_sandbox_for``).
        sandbox: Run the tool in an isolated box instead of the host
            process. ``True`` uses the default provider (``$LOCUS_SANDBOX``
            or the built-in subprocess box); a name, provider object, or
            :class:`~locus.tools.sandbox.SandboxSpec` selects/configures
            one. The function must be synchronous and self-contained; this
            is validated at decoration time. See :mod:`locus.tools.sandbox`.
        emits_progress: Stream the progress this tool reports with
            :func:`locus.tools.context.report_progress` as live
            :class:`~locus.core.events.ToolProgressEvent` s.

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
            deferred=deferred,
            emits_progress=emits_progress,
        )

    if fn is not None:
        # Called without arguments: @tool
        return decorator(fn)

    # Called with arguments: @tool(name="...")
    return decorator
