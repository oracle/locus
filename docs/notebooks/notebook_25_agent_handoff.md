# Agent Handoff

A handoff is one support agent saying "I'm done, please take this
further." The source packages the ticket, its findings, and a reason into
a typed `HandoffContext` so the next tier inherits the case state — not
just a string. No re-asking the customer, no lost context.

The running case is a support tier escalation: a customer reports a
duplicate subscription charge, and the ticket walks L1 → L2 → L3 with
billing lookups and account history gathered along the way.

This notebook covers:

- `HandoffContext` — typed payload carrying source/target ids, task,
  findings dict, confidence, instructions, and the full chain.
- `HandoffReason` — `SPECIALIZATION`, `ESCALATION`, `DELEGATION`,
  `COMPLETION`, `FAILURE`. Drives prompt templating and audit trails.
- `create_handoff_manager(...)` — builds a `Handoff` manager that
  registers a pool, enforces a `max_handoff_chain` cap, records every
  transfer.
- `manager.chain_handoff(agent_chain, task)` — walks a chain
  end-to-end, each agent inheriting prior findings.
- "Model B" slot (`LOCUS_MODEL_ID_B`) — drives the L1 front-line seat
  with a cheaper model; falls back to Model A when unset.

## Prerequisites

- Agent basics.
- The swarm multi-agent notebook for the peer-pull counterpoint to
  push-style handoffs.

## Run

```bash
python examples/notebook_25_agent_handoff.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Source

````python
--8<-- "examples/notebook_25_agent_handoff.py"
````
