# Multi-Agent HITL

The single-agent HITL notebook covered HITL for a single agent. Production payments
operations typically combine a triage agent, several specialists, and a human
gate for irreversible actions — nobody releases a high-value wire on an
agent's say-so alone. This notebook walks three combinations.

## Patterns

- **Pattern A — Approval gate**: triage classifies a payment exception, a
  disbursement specialist drafts the release action, a human approves before it
  executes.
- **Pattern B — Human-as-tool**: when triage isn't confident, it asks
  the human for the transaction type instead of guessing. The answer becomes
  part of state for downstream specialists.
- **Pattern C — Long-pause workflow**: state survives across an
  interrupt boundary so the approver on the next shift can come back hours later
  (different process, different caller) and the disbursement resumes.

## What the primitives do

- `interrupt(payload)` — function-level pause. Any node can call it.
  The graph catches the `InterruptException`, snapshots state, and
  returns an `InterruptState` to the caller.
- `graph.execute(Command(resume=<answer>, update=state))` — resume
  from the interrupt. The `interrupt()` call returns the resume value.
- Pair with a checkpointer for multi-process / multi-day pauses that
  preserve every specialist's context.

## Prerequisites

- Basic graph.
- Single-agent HITL.

## Run

```bash
python examples/notebook_27_multiagent_human_in_loop.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Source

````python
--8<-- "examples/notebook_27_multiagent_human_in_loop.py"
````
