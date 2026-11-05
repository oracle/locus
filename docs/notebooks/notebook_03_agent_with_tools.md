# Agent with Tools

Plain Python functions, decorated with `@tool`, become tools the agent
can call. The model decides when to use them; the SDK runs them and feeds
the result back into the next model call. This is what turns a language
model into an agent.

The scenario is a deployment-readiness check. The tools give the agent a
build-registry lookup (image tag, digest and signature state) and a
service record (DNS and replica health), and the system prompt tells it
to fetch the facts before giving a go or no-go. The data is fictional:
`example.com` hostnames, placeholder digests and made-up service names.

What this notebook covers:

- Turning a Python function into a tool with `@tool`.
- Passing tools to `Agent(tools=[...])`.
- Watching `ToolStartEvent` and `ToolCompleteEvent` in the stream (the
  [Events](../concepts/events.md) page describes every event type).
- Tools with optional arguments, default values, and structured return
  types.

Prerequisite: the [Basic agent](notebook_02_basic_agent.md) notebook.

Every lookup the agent makes here, image digest or health record, follows
the same path: `@tool` turns the function into a `Tool` with a schema
once, when the module is imported, and on each call the model's `ToolCall`
reaches the executor, which runs the function and returns the result to
the model as its next observation:

{{ locus_diagram tool-pipeline }}

## Key lines

`@tool` turns a plain function into a tool; the docstring is the
description the model reads:

{{ locus_source examples/notebook_03_agent_with_tools.py::lookup_image_digest }}

The agent takes its tools in `Agent(tools=[...])` and the model decides
when to call them; `result.metrics.tool_calls` counts the calls the run
made:

{{ locus_source examples/notebook_03_agent_with_tools.py::example_agent_with_tools }}

The streaming form surfaces each lookup as a `ToolStartEvent` /
`ToolCompleteEvent` pair, in the order the agent made them. The loop
below picks them out by `event_type` (`tool_start` and `tool_complete`)
and reads the final answer off the `terminate` event:

{{ locus_source examples/notebook_03_agent_with_tools.py::example_tool_events }}

## Run it

```bash
python examples/notebook_03_agent_with_tools.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Under the mock model each agent run makes exactly one tool call, to the
first tool in its `tools=[...]` list, and then answers with a fixed
sentence keyed on the prompt's wording, so the go/no-go verdicts in the
output below are scripted rather than derived from the lookups. With a
live provider the model chooses which tools to call and reasons from
their results.

## Output

Running it offline — no credentials, bundled mock model — prints the tools answering direct calls, the single lookup the mock model makes in each agent run (`Tool calls made: 1`; one `tool_start` / `tool_complete` pair in Part 4) followed by a fixed verdict keyed on the prompt's wording rather than derived from the lookup, and the blockers the structured readiness report lists for the failing service:

```text
============================================================
Notebook 03: Deployment Readiness — Giving an Agent Tools
============================================================

Model Provider: mock
Using mock model (no API calls)

=== Part 1: Tools Are Just Functions ===

lookup_image_digest('checkout') → checkout: v4.2.1 (signed), built 2026-08-11T09:14:00Z, digest sha256:9f2c1a7e4b3d0c8a6f5e2d1b0a9c8f7e6d5c4b3a2f1e0d9c8b7a6f5e4d3c2b1a
lookup_service_health('search') → search.svc.example.com · DNS TTL 3600s · 2/5 replicas healthy · health check failing

=== Part 2: Agent with Tools ===

Prompt: Are we clear to deploy Checkout? Check the image and the service health.
Response: Go on checkout: the image is signed and current, and all 6 replicas are healthy.
Tool calls made: 1

=== Part 3: Optional Arguments and Defaults ===

deployment_window_open() → production: change window closed until 22:00 UTC (business hours freeze)
deployment_window_open('staging') → staging: change window open
rollback_target('checkout') → checkout: rolling back 1 would land on v4.2.0
rollback_target('checkout', 2) → checkout: rolling back 2 would land on v4.1.7

=== Part 4: Tool Execution Events ===

Prompt: Run the pre-deploy checks on Search.
Events:
  → calling lookup_image_digest({'service': 'Search'})
  ← lookup_image_digest: Search: v1.9.0 (UNSIGNED), built 2026-06-02T17:40:00Z, digest sha256:1a2b3c4d5e6
  ✓ No-go on search: the image is unsigned and only 2 of 5 replicas are healthy with the health check fa

=== Part 5: Structured Return Types ===

checkout: v4.2.1 · 6/6 replicas
  blockers: none
search: v1.9.0 · 2/5 replicas
  blockers: image is unsigned, 3 replica(s) unhealthy, health check failing

Agent: No-go: the readiness report lists blockers that have to clear first.

============================================================
Next: Notebook 04 — A chat loop: actually talking to your agent
============================================================
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_03_agent_with_tools.py`"

    ````python
    --8<-- "examples/notebook_03_agent_with_tools.py"
    ````
