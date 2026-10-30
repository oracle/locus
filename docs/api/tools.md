# Tools

Define the functions an agent can call — the `@tool` decorator,
execution strategies, and JSON Schema generation.

For the concepts, start with [Tools](../concepts/tools.md).

## Decorator

The primary entry point — wrap a Python function and you get a `Tool`
the agent can call. Parameters are introspected from the type hints
and the JSON Schema is generated automatically.

::: locus.tools.decorator.tool
::: locus.tools.decorator.Tool

## Tool context

Inject per-call context (the agent's state, the run and call
identifiers, invocation metadata and per-tool config) into a tool by
declaring a `ToolContext` parameter.

::: locus.tools.context.ToolContext

## Registry

The agent's compiled tool collection. Built once during
`Agent.__init__`; mutating `config.tools` directly afterwards has no
effect (use `Agent.add_tool` / `add_tools` instead).

::: locus.tools.registry.ToolRegistry

## Executors

The strategy the agent uses to run a batch of tool calls. The default
is `ConcurrentExecutor` (parallel up to `max_concurrency`);
`SequentialExecutor` runs them one at a time for tools that share
non-thread-safe state.

::: locus.tools.executor.ToolExecutor
::: locus.tools.executor.ConcurrentExecutor
::: locus.tools.executor.SequentialExecutor

## Schema generation

JSON Schema generation from Python type hints / Pydantic models —
used by the `@tool` decorator but also callable directly when you
need a schema for an external system (e.g. an MCP server).

::: locus.tools.schema.generate_schema
::: locus.tools.schema.pydantic_to_json_schema

## Built-in tools

::: locus.tools.builtins.get_today_date

### OCI tools

Agents that drive Oracle Cloud Infrastructure: `use_oci` runs a read-only
OCI Python SDK operation and `describe_oci` lists the SDK's services, client
classes and operations. Both need the `oci` extra. See the
[OCI tools notebook](../notebooks/notebook_36_oci_tools.md).

::: locus.tools.oci.use_oci
::: locus.tools.oci.describe_oci
