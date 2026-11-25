# Human in the loop

Pause a graph mid-execution, ask a human, then resume with their
answer. `interrupt(payload)` pauses the running node and returns
control to the caller with `result.is_interrupted = True`. The caller
inspects the payload, gets a response from an operations console, a
command-line prompt or a chat reply, then calls
`graph.execute(Command(update=..., resume=...))` to continue from the
paused node.

Rolling out a change to production infrastructure — a deploy, a
migration, a rollback — is the kind of action an agent should not take
on its own authority. The approval gate is the control against
excessive agency: a human signs off before the side effect happens.

What this notebook covers:

- `interrupt(payload)` — surface a payload to the caller.
- `Command(update=..., resume=...)` — resume execution.
- Multiple interrupts in one workflow.
- Conditional interrupts (only ask for higher-risk rollouts).
- `graph.config.interrupt_before = [...]` — pause before specific nodes
  without modifying them.

The drawing has two bands; every part of this notebook takes the graph path in the upper one, and Part 1 is the simplest case. The `approval` node calls `interrupt(payload)`, the call unwinds to the graph as an `InterruptException`, and `execute()` returns a `GraphResult` whose `interrupt` field carries the paused node and the question payload.

The caller shows the question, waits for the release manager, and resumes with `Command(update=result.final_state, resume="yes")`, which runs the `approval` node again from the top; this time `interrupt()` returns the answer instead of pausing.

Two notes in the drawing apply throughout:

- These graphs have no checkpointer, so the returned `final_state` travels back inside the `Command` rather than through the dashed checkpointer lane.
- Every node is `async def`, because a plain `def` body runs in a worker thread that never sees the resume value.

The lower band is the agent-side counterpart — an `ask_user` question or a held gated tool call, surfaced as an `InterruptEvent` and released with `agent.resume()` — which this notebook does not exercise; the [Interrupts](../concepts/interrupts.md) concept page covers it:

{{ locus_diagram interrupt-resume }}

## Key lines

A three-node deploy graph puts `interrupt()` inside the `approval` node;
the first `execute()` comes back with `is_interrupted` set and the
question in the payload, and the second, with
`Command(update=..., resume="yes")`, re-runs that node and carries on to
the deploy:

{{ locus_source examples/notebook_15_human_in_the_loop.py::example_basic_interrupt }}

A gate does not have to ask every time: the `approve` node auto-approves
a small blast radius and calls `interrupt()` only for medium and large
rollouts, with the payload naming the approver that is required:

{{ locus_source examples/notebook_15_human_in_the_loop.py::example_conditional_interrupt }}

`graph.config.interrupt_before = ["migrate"]` pauses the graph before a
node that never calls `interrupt()` itself, which is the option when the
sensitive step is code you cannot edit; the resume value is just the
go-ahead token:

{{ locus_source examples/notebook_15_human_in_the_loop.py::example_interrupt_before }}

## Run it

```bash
python examples/notebook_15_human_in_the_loop.py
```

This example builds no model and does not read `examples/config.py`, so
`LOCUS_MODEL_PROVIDER` has no effect; every approver's answer is scripted
in the file, so the run needs no input.

## Output

Running it offline — no model and no credentials — prints the pause-and-resume trail of every gate: where each graph paused, the question it surfaced, the answer given and the resulting state, including the blast-radius routing that auto-approves 2 pods and escalates 20 and 200 to the site reliability engineering (SRE) on-call and the release manager:

```text
============================================================
Notebook 15: Human approval gates for production changes
============================================================

=== Part 1: One interrupt ===

Starting deploy workflow...
PAUSED at: approval
Question: Approve deploy to checkout-api-prod?
Release manager responds: 'yes'
Result: Executed deploy to checkout-api-prod

=== Part 2: Several interrupts in a row ===

Change-request intake flow:
  Q: Which service are you changing?
  A: checkout-api
  Q: What change for checkout-api?
  A: rollback
  Q: Confirm: rollback on checkout-api?
  A: confirm

Final: Change ticket created

=== Part 3: Interrupt only when it matters ===

Rolling out to 2 pods...
  -> Rollout approved by auto
Rolling out to 20 pods...
  Needs approval: medium blast radius
  -> Rollout approved by SRE on-call
Rolling out to 200 pods...
  Needs approval: large blast radius
  -> Rollout approved by release manager

=== Part 4: interrupt_before — pause without modifying the node ===

Running a production schema migration...
PAUSED before: migrate
Current state: backed_up=True
DBA confirms the snapshot is good — resuming.
Result: migrated=True, verified=True

=== Part 5: A two-stage approval workflow ===

Change Request Workflow
------------------------------

SRE Review: Is this rollout safe to ship now?
  -> approve

Release Manager Approval: Approve deploying this change to production?
  -> approve

Result: Request CHG-001: approved

============================================================
Next: Notebook 16 — Advanced graph patterns
============================================================
```
<!-- notebook-output:end -->

## See also

- [Interrupts](../concepts/interrupts.md) — the agent-side pause:
  `ask_user`, `InterruptEvent`, `agent.resume()` and gated tool calls
  held for approval.
- [Multi-agent human in the loop](notebook_27_multiagent_human_in_loop.md) —
  approval gates, human-as-tool and a long pause in a multi-agent payments
  workflow.

## Source

??? note "Full source — `examples/notebook_15_human_in_the_loop.py`"

    ````python
    --8<-- "examples/notebook_15_human_in_the_loop.py"
    ````
