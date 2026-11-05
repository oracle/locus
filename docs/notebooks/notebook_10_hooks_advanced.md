# Advanced Hooks

The [agent-hooks notebook](notebook_09_agent_hooks.md) covered hook basics.
This one focuses on the safety properties the SDK enforces on the event objects
hooks see, and on the control lever a hook pulls during a run: `event.cancel`
to block a destructive tool call. A second lever, `event.retry`, is writable
on the after-events (`on_after_model_call`, `on_after_tool_call`) and is not
exercised here.

The scenario is RELEASE GUARD — the change-gating layer that sits between a
deploy-ops agent and the production cluster. A deploy agent that can delete a
namespace or destroy an environment on its own has an unbounded blast radius;
RELEASE GUARD is the bound on that agency.

What this notebook covers:

- Most fields on hook event objects are read-only. Mutating
  `event.tool_name` raises `AttributeError`; that is the framework
  protecting the agent's invariants (and your change log).
- `event.arguments`, `event.cancel` and `event.secret_arguments` *are*
  writable; the last carries extra input that is merged into the tool
  invocation only and is never checkpointed.
- Setting `event.cancel = "<reason>"` in `on_before_tool_call` skips the
  call and feeds the reason back as the tool's result — here, blocking a
  `delete_namespace` call during an active change freeze.
- Hooks run in list order on the "before" callbacks and unwind in reverse on
  the "after" callbacks, so cleanup is last in, first out (LIFO). The `Agent`
  keeps the order of its `hooks=[…]` list; `priority` orders hooks only when
  they pass through a `HookRegistry` — see
  [Priorities — the ordering rules](../concepts/hooks.md#priorities-the-ordering-rules).

Prerequisite: the agent-hooks notebook.

The drawing places every hook point in a run. RELEASE GUARD acts at
`on_before_tool_call`, where the orchestrator dispatches the hooks in list
order ahead of each tool call; the gate and ledger hooks of Part 3 also record
`on_after_tool_call`, where the orchestrator unwinds them in reverse order:

{{ locus_diagram hook-lifecycle }}

## Key lines

RELEASE GUARD sets `event.cancel` in `on_before_tool_call` for any tool whose
name contains `delete`, and the agent that carries it is asked to delete a
namespace:

{{ locus_source examples/notebook_10_hooks_advanced.py::example_cancel_tool }}

A `BeforeToolCallEvent` probed directly shows which fields accept a write and
which raise:

{{ locus_source examples/notebook_10_hooks_advanced.py::example_write_protection }}

Two tracing hooks, listed gate first, record the dispatch order on the way in
and the reversed order on the way out. Their `priority` values (50 in the
security band, 250 in the business band) label the hooks; the `Agent` does not
read them, so list position alone sets the order:

{{ locus_source examples/notebook_10_hooks_advanced.py::example_hook_order }}

## Run it

```bash
python examples/notebook_10_hooks_advanced.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Output

Running it offline — no credentials, bundled mock model — prints the blocked tool result RELEASE GUARD feeds back to the model, the read-only error that names the writable event fields, and the gate/ledger call sequence reversing on the way out:

```text
============================================================
Notebook 10: RELEASE GUARD — Guardrails Around a Deploy-Ops Agent
============================================================

Model Provider: mock
Using mock model (no API calls)

=== Part 1: Cancel Tool via Hook ===

Response: This is a mock response for testing purposes.
  Tool: delete_namespace → BLOCKED: delete_namespace is forbidden during an active change freeze

=== Part 2: Write Protection ===

arguments (writable): {'replicas': 3}
cancel (writable): blocked pending change-approval ticket
tool_name (read-only): Cannot set 'tool_name' on BeforeToolCallEvent — read-only. Writable fields: arguments, cancel, secret_arguments

=== Part 3: Hook ordering ===

Call sequence: before:gate → before:ledger → after:ledger → after:gate
  gate runs first on the way in; ledger unwinds first on the way out.

============================================================
Next: Notebook 11 — Model providers: one cloud-ops agent, swap models with a string
============================================================
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_10_hooks_advanced.py`"

    ````python
    --8<-- "examples/notebook_10_hooks_advanced.py"
    ````
