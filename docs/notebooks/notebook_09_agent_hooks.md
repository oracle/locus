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
  maximum instance count — and re-signing it with `event.resign(by=...)`
  so the signed call still runs.
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
receives it. Tool calls are signed over their arguments, so the hook then
calls `event.resign(by=...)`: the executor records which hook changed the
call instead of refusing it as tampered with.

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

## Output

Running it against a live model — OCI Generative AI, `openai.gpt-4.1` — prints the audit hook's trail around one tool call, the timing report, the validation hook clamping `base` and `delta` to 100 before the tool runs (so the tool returns 200), and the guardrail hook flagging `password` in a tool argument; the timestamped audit log of Part 4 is elided (`...`); the model's wording and the timings differ from run to run:

```text
=== Part 1: Understanding Hooks ===

Running agent with audit-trail hook:

  [AUDIT] Run started: 'Scale the service: base 5, delta 3...'
  [AUDIT] Tool call: adjust_capacity({'base': 5, 'delta': 3})
  [AUDIT] Tool done: adjust_capacity -> 8
  [AUDIT] Run finished: success=True

Result: The service has been scaled. The new instance count is 8 (base 5 + delta 3). If you need further adjustments or details, let me know!

=== Part 2: Timing Hook ===


  Timing Report:
    Total: 2581.6ms
    adjust_capacity: 0.4ms
Result: The service has been scaled from a base of 10 instances by a delta of 20, resulting in a new total of 30 instances.

=== Part 3: Validation Hook ===

  [VALIDATION] Clamping base=5000 to 100
  [VALIDATION] Clamping delta=3000 to 100
Result: The adjusted capacity value returned is 200.

If you need a different calculation (such as with your requested base 5000 and delta 3000), please clarify. Otherwise, this is the value based on the last operation.

=== Part 4: Multiple Hooks ===


  Timing Report:
    Total: 2477.9ms
    adjust_capacity: 0.5ms
Result: The service has been scaled from a base of 7 by a delta of 8, resulting in a new total of 15 instances.

...

=== Part 5: Guardrails Hook ===

  [GUARDRAIL] Blocked pattern detected: 'password'
  [GUARDRAIL] Warning: 'password' in analyze_manifest args
Result: Here is the analysis of your manifest:

- Character count: 24
- Word count: 3
- Unique words: 3
- SHA-256 digest: 13ca8e50e366

Let me know if you need a deeper analysis or further breakdown of this manifest.

Blocked calls detected: 1

============================================================
Next: Notebook 10 — RELEASE GUARD: guardrails around a deploy-ops agent
============================================================
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_09_agent_hooks.py`"

    ````python
    --8<-- "examples/notebook_09_agent_hooks.py"
    ````
