# Agent Handoff

A handoff is one support agent declaring its part of a case complete and
passing the ticket to the next tier. The source agent packages the ticket,
its findings and a reason into a typed `HandoffContext`, so the next tier
starts from the case state already gathered rather than from a bare
string, and the customer is not asked the same questions again.

The running case is a support tier escalation: a customer reports a
duplicate subscription charge, and the ticket walks from first-line
support (L1) through a specialist (L2) to an escalation manager (L3). A
hand-built `AgentState` stands in for the L1 transcript, including the
`lookup_account` and `query_billing` tool calls it records.

What this notebook covers:

- `HandoffContext` — the typed payload: source and target ids, reason,
  original task, conversation summary, findings, confidence, instructions
  and the handoff chain; `to_prompt()` renders it as the prompt body the
  target agent reads.
- `HandoffReason` — `SPECIALIZATION`, `ESCALATION`, `DELEGATION`,
  `COMPLETION`, `FAILURE`; the value is written into the prompt header by
  `to_prompt()` and kept with the context in the manager's history.
- `create_handoff_manager(...)` — returns the `Handoff` manager, which
  registers the pool, refuses an `execute_handoff` once its history holds
  `max_handoff_chain` contexts, and records every context it creates.
- `Handoff.create_handoff(...)` with `state=` — fills the conversation
  summary, the state snapshot (iteration, last five tool names, last three
  errors) and the confidence from an `AgentState`; key messages travel
  only with `preserve_full_history=True`.
- `Handoff.chain_handoff(agent_chain, task, initial_state)` — walks the
  chain pairwise with `HandoffReason.DELEGATION`, adds each step's output
  to the findings the next step receives, and stops at the first failed
  step.
- The "Model B" slot (`LOCUS_MODEL_ID_B`) — a second, typically cheaper
  model id for the L1 front-line seat; `get_model_b` falls back to
  `LOCUS_MODEL_ID` when it is unset.

## Prerequisites

- [Basic agent](notebook_02_basic_agent.md).
- [Swarm](notebook_24_swarm_multiagent.md), where agents pull work from a
  shared queue instead of being handed a case.

The drawing follows one `execute_handoff` call of the kind Part 4 makes, from the L1 agent to the L2 tier. It is shared with the [Handoff](../concepts/multi-agent/handoff.md) concept page, so its target is named `L2 Billing`; this notebook's L2 is the `L2 Support Specialist`.

The manager refuses with a failed `HandoffResult` when the target is not registered or its history already holds `max_handoff_chain` contexts. Otherwise it builds the `HandoffContext` from the task, findings, instructions and handoff chain and, because the call passes the L1 state, adds the source's confidence and a conversation summary; key messages from the transcript are attached only with `preserve_full_history` on.

The L2 agent reads that context as its prompt and answers with a `HandoffResult` carrying a success flag, its output and a final confidence:

{{ locus_diagram handoff }}

## Key lines

The example keeps the whole walk in `main`, so the excerpt below is the full program. Part 1 attaches the models, builds the three tiers with `create_handoff_agent`, declares the delegation and escalation paths (`can_delegate_to` and `can_escalate_to` are recorded on the agents but not checked by the manager) and sends the L1 seat a one-line smoke test through `receive_handoff`.

Part 2 builds a `HandoffContext` by hand and prints its `to_prompt()`; Part 3 lists the five `HandoffReason` values. Part 4 runs one `execute_handoff` from L1 to L2 with the L1 state, Part 5 builds a context from an `AgentState` with `create_handoff`, Part 6 walks L1 → L2 → L3 with `chain_handoff`, and Part 7 reads the manager's history back as the audit trail:

{{ locus_source examples/notebook_25_agent_handoff.py::main }}

## Run it

```bash
python examples/notebook_25_agent_handoff.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

The L1 seat is built with `get_model_b`, which reads `LOCUS_MODEL_ID_B` and
falls back to `LOCUS_MODEL_ID`, so with only `LOCUS_MODEL_ID` set all three
tiers share one model. Under a live provider, set at least
`LOCUS_MODEL_ID`: when neither variable is set, `get_model_b` passes an
empty model id, while `get_model` gives the other two tiers the provider
default. The mock model ignores model ids.

## See also

- [Handoff](../concepts/multi-agent/handoff.md) — the concept page: when to
  hand a case off rather than orchestrate it, and what transfers across the
  handoff.

## Source

??? note "Full source — `examples/notebook_25_agent_handoff.py`"

    ````python
    --8<-- "examples/notebook_25_agent_handoff.py"
    ````
