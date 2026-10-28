# Agent Hooks

Hooks are middleware for agents. Subclass `HookProvider`, override the
callbacks you need, and Locus invokes them at eight lifecycle points:
before/after the invocation, at the start/end of each loop iteration,
before/after each model call, and before/after each tool call. Use them
to add logging, timing, validation, guardrails, or any cross-cutting
concern without touching the agent or its tools.

The scenario here is a cloud-ops agent that scales service instances
with an `adjust_capacity` tool. The hooks give it an audit trail,
latency telemetry, a quota clamp that keeps the bill in check, and a
secrets guardrail over untrusted deployment-manifest text — all bolted
on from the outside.

What you'll learn:

- Writing a `HookProvider` and registering it on an `Agent`.
- The four callback points this notebook uses, and what they receive —
  see [Hooks](../concepts/hooks.md) for all eight.
- Using `HookPriority` to control execution order.
- Mutating `event.arguments` from `on_before_tool_call` to rewrite the
  call before the tool runs — here, clamping `adjust_capacity` to a
  maximum instance count.
- Composing several hooks on one agent.

Run it:

```
.venv/bin/python examples/notebook_09_agent_hooks.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Prerequisite: the agent-streaming notebook.

## Source

````python
--8<-- "examples/notebook_09_agent_hooks.py"
````
