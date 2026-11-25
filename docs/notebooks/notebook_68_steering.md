# Steering

`SteeringHook` runs a second large language model (LLM), the steering model, in front of every tool call. Before a tool runs, the steering model reads a natural-language policy together with the agent's recent tool calls and answers with one of three actions:

- `PROCEED` lets the call run.
- `GUIDE` cancels the call, and the agent reads `Steering blocked: <reason>` in place of a tool result.
- `INTERRUPT` also cancels the call, and the agent reads `REQUIRES APPROVAL: <reason>` in its place.

Neither refusal pauses the run or waits for a person: the hook records the decision and the agent loop continues, with the message standing in for the tool's result.

The result is a guardrail written in plain English rather than in a rules
engine or a policy language. This example puts a read-only policy in front of
an on-call diagnostics agent. The agent has two tools, a metrics query and a
service restart; the policy allows metric and log queries and forbids
restarts, scaling and terminations, so the steering model is expected to let
the query through and refuse the restart.

The policy is advisory and can fail open: the judge is a model, any reply that does not begin with `GUIDE` or `INTERRUPT` counts as `PROCEED`, and so does a failed judge call. Use steering to correct an agent's course, not to stop it.

For an action that must not happen, put the decision in code the model cannot argue with: [`gate_tool`](../api/control.md#gating-a-tool) in the control layer, or the hook's own `interrupt_tools=` set, which refuses the named tools without consulting the judge. The [Safety page](../concepts/safety.md#steering-a-second-model-judges-every-tool-call) records a run in which the judge did not intervene.

What this notebook covers:

- `SteeringHook(model=..., policy="...")` — attached to the agent through
  the `hooks=` parameter, with the policy as plain text.
- `PROCEED`, `GUIDE` and `INTERRUPT` — the three decisions, and what the
  agent reads after each.
- `steering.decisions` — every decision, with the judge's reason for a
  refusal; the record for the post-incident review.

## Prerequisites

- An OCI profile in `~/.oci/config` (picked up automatically), or
  `LOCUS_MODEL_PROVIDER` set to `openai` or `anthropic` with the matching
  key, or `mock` for an offline run.

The drawing is the [control layer](../concepts/control-layer.md) as a gated tool call traverses it. The steering hook stands where the gate stands: between the call the agent loop proposes and its side effect, where the drawing has `Action` and `ControlPolicy`. `steering.decisions` plays the part of the `AuditTrail`, as a plain list rather than a hash-chained record.

In the drawing's terms a steering refusal, `GUIDE` or `INTERRUPT`, is a deny rather than a hold: the call is not executed and the refusal goes back to the model as its result, so there is no approval store and no resume. The checkpointer and grounding in the drawing play no part in this example either:

{{ locus_diagram control-layer-flow }}

## Key lines

`_make_agent` builds one agent per hook. The hook attaches through `hooks=`,
next to the two tools it will judge, one read-only and one mutating:

{{ locus_source examples/notebook_68_steering.py::_make_agent }}

`SteeringHook(model=..., policy=...)` takes the same model instance as the agent, chosen by `examples/config.py` (in production the steering model can be a smaller one), and the policy as plain text. `_POLICY` is three sentences: "This is a read-only diagnostics session. Only allow metric and log queries. Never allow mutating or destructive operations such as restarting services, scaling deployments, or terminating instances."

The first attempt asks for a restart, which the policy forbids; the second asks for a latency query, which it allows. Each attempt gets a fresh hook, so its `decisions` list holds only that run's verdicts:

{{ locus_source examples/notebook_68_steering.py::example_steering }}

## Run it

```bash
python examples/notebook_68_steering.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

To force the bundled mock model regardless of what `~/.oci/config` contains:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_68_steering.py
```

Under the mock model the steering model's canned reply begins with neither
`GUIDE` nor `INTERRUPT`, so both calls proceed and `restart_service` runs; a
live model is needed to see a refusal.

## See also

- [Safety, guardrails, and steering](../concepts/safety.md#steering-a-second-model-judges-every-tool-call)
  — why steering is advisory, and when to reach for `gate_tool` or
  `interrupt_tools` instead.
- [Control](../api/control.md#gating-a-tool) — `gate_tool`, which puts
  `admit()` in front of a tool: the structural gate the drawing shows.
- [Hooks](../concepts/hooks.md) — how `SteeringHook` plugs into the agent
  lifecycle.

## Source

??? note "Full source — `examples/notebook_68_steering.py`"

    ````python
    --8<-- "examples/notebook_68_steering.py"
    ````
