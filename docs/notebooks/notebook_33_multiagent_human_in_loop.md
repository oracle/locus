# Multi-Agent HITL

Notebook 20 covered HITL for a single agent. Production systems
typically combine a triage agent, several specialists, and a human
gate for irreversible actions. This notebook walks three combinations.

## Patterns

- **Pattern A — Approval gate**: triage classifies a refund, a
  specialist drafts the response, a human approves before it ships.
- **Pattern B — Human-as-tool**: when triage isn't confident, it asks
  the human for the category instead of guessing. The answer becomes
  part of state for downstream specialists.
- **Pattern C — Long-pause workflow**: state survives across an
  interrupt boundary so the human can come back hours later (different
  process, different caller) and the workflow resumes.

## What the primitives do

- `interrupt(payload)` — function-level pause. Any node can call it.
  The graph catches the `InterruptException`, snapshots state, and
  returns an `InterruptState` to the caller.
- `graph.execute(Command(resume=<answer>, update=state))` — resume
  from the interrupt. The `interrupt()` call returns the resume value.
- Pair with a checkpointer for multi-process / multi-day pauses that
  preserve every specialist's context.

## Prerequisites

- Notebook 17 (basic graph).
- Notebook 20 (single-agent HITL).

## Run

```bash
python examples/notebook_33_multiagent_human_in_loop.py
```

The default provider is the bundled mock model. Set `TULIP_MODEL_PROVIDER`
(openai / anthropic) and credentials to use a live model. Set
`TULIP_MODEL_PROVIDER=mock` for offline runs.

## Source

```python
--8<-- "examples/notebook_33_multiagent_human_in_loop.py"
```
