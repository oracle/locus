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

The default provider is OCI Generative AI. With `~/.oci/config`
present the roles talk to a live OCI model; canonical picks are
`openai.gpt-4.1` or `meta.llama-3.3-70b-instruct`. Set
`LOCUS_MODEL_PROVIDER=mock` for offline runs.

## Source

```python
--8<-- "examples/notebook_31_supervisor_critic_loop.py"
```
