# Supervisor + Critic

A researcher gathers notes, a writer drafts a response, a critic
either approves or sends it back for revision. The loop caps at two
revisions to bound runtime.

This notebook covers:

- Control flow as a `StateGraph` with conditional edges — no
  hand-rolled `while True`.
- Each role is its own `Agent` with a role-specific system prompt.
  Roles communicate only through state keys (`notes`, `draft`,
  `revision_request`).
- `stream(mode=StreamMode.NODES)` emits one event per node completion
  for live UI updates.
- `execute(...)` returns the authoritative final state plus a
  `GraphResult` with timing and iteration metrics.

```text
START → research → write → critique → END (approve)
                     ↑         │
                     └── revise (cap: 2)
```

## Prerequisites

- Notebook 17 (basic graph).
- Notebook 26 (agent handoff) for an alternative shape.

## Run

```bash
python examples/notebook_31_supervisor_critic_loop.py
```

The default provider is the bundled mock model. Set
`TULIP_MODEL_PROVIDER` (openai / anthropic) and credentials to use a
live model. Set
`TULIP_MODEL_PROVIDER=mock` for offline runs.

## Source

```python
--8<-- "examples/notebook_31_supervisor_critic_loop.py"
```
