# Agent Hooks

Hooks are middleware for agents. Subclass `HookProvider`, override the
callbacks you need, and the SDK invokes them at nine lifecycle points:
before/after the invocation, at the start/end of each loop iteration,
before context compaction, before/after each model call, and
before/after each tool call. Use them to add logging, timing,
validation, guardrails, or any cross-cutting concern without touching
the agent or its tools.

The scenario here is a cloud-ops agent that scales service instances
with an `adjust_capacity` tool. The hooks give it an audit trail,
latency telemetry, a quota clamp that caps instance counts, and a
secrets guardrail over untrusted deployment-manifest text, all added
from outside the agent and its tools.

What this notebook covers:

- Writing a `HookProvider` and registering it on an `Agent`.
- The four callback points this notebook uses, and what they receive —
  see [Hooks](../concepts/hooks.md) for all nine.
- Declaring a `HookPriority` band on each hook, and why the `Agent` still
  runs hooks in list order: only a `HookRegistry` sorts by priority (see
  [Priorities](../concepts/hooks.md#priorities-the-ordering-rules)).
- Mutating `event.arguments` from `on_before_tool_call` to rewrite the
  call before the tool runs — here, clamping `adjust_capacity` to a
  maximum instance count.
- Composing two hooks on one agent.

Prerequisite: the agent-streaming notebook.

The drawing places every hook point on the agent loop and follows each
event to the providers in the agent's `hooks` list. This notebook uses
four of the nine points: `on_before_invocation` and `on_after_invocation`
wrap the whole run, and `on_before_tool_call` and `on_after_tool_call`
wrap each tool call. Every event reaches the providers in list order on
the way in and in reverse on the way out:

{{ locus_diagram hook-lifecycle }}

## Key lines

A hook is a `HookProvider` subclass that declares a priority band and
overrides only the callbacks it needs; this one prints an audit line at
each of the four points the notebook uses:

{{ locus_source examples/notebook_09_agent_hooks.py::AuditTrailHook }}

The `event.arguments` mapping is writable in `on_before_tool_call`, so a
hook can clamp an out-of-range capacity change before `adjust_capacity`
receives it:

{{ locus_source examples/notebook_09_agent_hooks.py::ValidationHook }}

Two hooks on one agent: the `Agent` runs them in list order on the way
in and in reverse on the way out, and the run that follows collects a
structured audit log:

{{ locus_source examples/notebook_09_agent_hooks.py::example_multiple_hooks }}

## Run it

```bash
python examples/notebook_09_agent_hooks.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Source

??? note "Full source — `examples/notebook_09_agent_hooks.py`"

    ````python
    --8<-- "examples/notebook_09_agent_hooks.py"
    ````
