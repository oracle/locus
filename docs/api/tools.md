# Tools

## Decorator

The primary entry point — wrap a Python function and you get a `Tool`
the agent can call. Parameters are introspected from the type hints
and the JSON Schema is generated automatically.

::: tulip.tools.decorator.tool
::: tulip.tools.decorator.Tool

## Tool context

Inject per-call context (the agent's state, custom metadata, the
hook orchestrator) into a tool by declaring a `ToolContext` parameter.

::: tulip.tools.context.ToolContext

## Registry

The agent's compiled tool collection. Built once during
`Agent.__init__`; mutating `config.tools` directly afterwards has no
effect (use `Agent.add_tool` / `add_tools` instead).

::: tulip.tools.registry.ToolRegistry

## Executors

The strategy the agent uses to run a batch of tool calls. The default
is `ConcurrentExecutor` (parallel up to `max_concurrency`);
`SequentialExecutor` runs them one at a time for tools that share
non-thread-safe state.

::: tulip.tools.executor.ToolExecutor
::: tulip.tools.executor.ConcurrentExecutor
::: tulip.tools.executor.SequentialExecutor

## Schema generation

JSON Schema generation from Python type hints / Pydantic models —
used by the `@tool` decorator but also callable directly when you
need a schema for an external system (e.g. an MCP server).

::: tulip.tools.schema.generate_schema
::: tulip.tools.schema.pydantic_to_json_schema

## Built-in tools

::: tulip.tools.builtins.get_today_date
