# Multi-agent human in the loop

The [Human in the loop](notebook_15_human_in_the_loop.md) notebook paused a
workflow that had no agents in it: a person answered, and the graph carried
on. Production payments operations typically combine a triage agent, several
specialists and a human gate for irreversible actions, because no one
releases a high-value wire on an agent's recommendation alone. This notebook
builds three such human-in-the-loop (HITL) combinations on the same two
primitives: `interrupt()` inside a node and `Command(resume=...)` from the
caller.

What this notebook covers:

- **Pattern A — Approval gate**: triage classifies a payment exception, a
  disbursement specialist drafts the release action, and a human approves
  before it executes.
- **Pattern B — Human-as-tool**: when triage is not confident, it asks the
  human for the transaction type instead of guessing, and the answer becomes
  part of the state the downstream route reads.
- **Pattern C — Long-pause workflow**: the state returned at the pause is
  held across the interrupt boundary and the disbursement resumes when the
  approver answers; a checkpointer extends the same shape to a
  multi-process, multi-day pause.
- `interrupt(payload)` — a function-level pause that any node can call. The
  graph catches the `InterruptException`, snapshots state and returns a
  `GraphResult` whose `interrupt` field is an `InterruptState` naming the
  paused node and carrying the payload.
- `graph.execute(Command(resume=<answer>, update=state))` — resumes from the
  interrupt: the paused node runs again and its `interrupt()` call returns
  the answer.

## Prerequisites

- [Basic graph](notebook_12_basic_graph.md).
- [Human in the loop](notebook_15_human_in_the_loop.md), the same gate in a
  graph without agents.

All three patterns take the pause-and-resume path drawn in the upper band: an `async def` node calls `interrupt(payload)`, the `InterruptException` unwinds to the graph, and `execute()` returns a `GraphResult` whose `interrupt` field holds the paused node and the question for the approver.

The caller answers with `Command(resume=...)`; the paused node runs again from the top, and this time `interrupt()` returns the answer instead of raising.

The graphs in this notebook have no [checkpointer](../concepts/checkpointers.md), so the returned `final_state` travels back inside the `Command`; the dashed path through the checkpointer is the multi-process variant Pattern C points to.

The lower band is the agent-side counterpart, which this notebook does not use; the [Interrupts](../concepts/interrupts.md) concept page covers it:

{{ locus_diagram interrupt-resume }}

## Key lines

`human_approval_node` is the approve step that `build_approval_graph` wires
between the disbursement draft and execution: `interrupt(payload)` pauses the
graph, hands the question and the drafted action to the approver, and on
resume returns the answer that decides whether `execute_or_cancel_node`
releases or cancels the payment:

{{ locus_source examples/notebook_27_multiagent_human_in_loop.py::human_approval_node }}

`smart_triage_node` is the human-as-tool of Pattern B: any answer outside the
explicit category set falls through to escalation, where `interrupt()` asks
the person for the transaction type and the reply becomes the `category` the
downstream route reads:

{{ locus_source examples/notebook_27_multiagent_human_in_loop.py::smart_triage_node }}

`demo_pattern_c` is the long pause of Pattern C: the `final_state` returned at the pause is the snapshot to hold, and the second `execute` call passes it back inside `Command(resume=...)` with the model re-attached, the same shape a multi-process resume keeps once the snapshot is loaded from a store you manage.

With `GraphConfig(checkpointer=..., thread_id=...)` the graph saves the snapshot at the pause and reloads it on resume, ignoring the `update` passed with the `Command`, so a state that carries a live model object under `__model__`, as this one does, is held by the caller instead:

{{ locus_source examples/notebook_27_multiagent_human_in_loop.py::demo_pattern_c }}

## Run it

```bash
python examples/notebook_27_multiagent_human_in_loop.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Output

Running it offline — no credentials, bundled mock model — prints the pause-and-resume trail of all three patterns: the node each graph pauses at, the approver's question, the resume value and the final outcome, with Pattern B's escalation routed to the human's answer and Pattern C's snapshot key count:

```text
============================================================
Notebook 27: human-approved disbursement — multi-agent HITL
============================================================

Model Provider: mock
Using mock model (no API calls)

=== Pattern A: Approval gate ===

  ⏸  Paused at: approve
     Question:  Approve releasing this high-value payment?
     Action:    This is a mock response for testing purposes.
  ▶  Human responds: 'yes'
  ✓ Final outcome: ✓ Payment released — funds disbursed

=== Pattern B: Human-as-tool (escalation) ===

  ⏸  Triage escalated. Asking human:
     Triage agent is not confident. Pick a transaction type for: 'intermittent partial refunds to a newly added payee (payee-sync.example), but only during nightly settlement batches?'
  ▶  Human responds: 'refund'
  ✓ Routed to: refund

=== Pattern C: Long-pause workflow (snapshot + resume) ===

  ⏸  Paused at approve
     Snapshot has 7 state keys — persist these to a checkpointer / a payment-approval queue / etc.
  ▶  Next shift: load snapshot, resume with the approver's answer
  ✓ Resumed and finished: ✓ Payment released — funds disbursed

============================================================
Next: Notebook 28 — emergent incident routing — the model picks the specialist protocol
============================================================
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_27_multiagent_human_in_loop.py`"

    ````python
    --8<-- "examples/notebook_27_multiagent_human_in_loop.py"
    ````
