# Agent Handoff

A handoff is one agent saying "I'm done, please take this further." The
source packages the task, its findings, and a reason into a typed
`HandoffContext` so the target inherits the work state — not just a
string.

This notebook covers:

- `HandoffContext` — typed payload carrying source/target ids, task,
  findings dict, confidence, instructions, and the full chain.
- `HandoffReason` — `SPECIALIZATION`, `ESCALATION`, `DELEGATION`,
  `COMPLETION`, `FAILURE`. Drives prompt templating and audit trails.
- `HandoffManager` — registers a pool, enforces a `max_handoff_chain`
  cap, records every transfer.
- `manager.chain_handoff(agent_chain, task)` — walks a chain
  end-to-end, each agent inheriting prior findings.
- "Model B" slot (`TULIP_MODEL_ID_B`) — drives the triage seat with a
  cheaper model; falls back to Model A when unset.

## Prerequisites

- Notebook 08 (Agent basics).
- Notebook 25 (Swarm) for the peer-pull counterpoint to push-style
  handoffs.

## Run

```bash
python examples/notebook_25_agent_handoff.py
```

The default provider is the bundled mock model. Set `TULIP_MODEL_PROVIDER`
(openai / anthropic) and credentials to use a live model. Set
`TULIP_MODEL_PROVIDER=mock` for offline runs.

## Source

```python
--8<-- "examples/notebook_25_agent_handoff.py"
```
