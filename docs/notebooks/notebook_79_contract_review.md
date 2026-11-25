# Contract review

Reviewing a data processor's Data Processing Agreement (DPA) involves
several stakeholders working in parallel, then a negotiation phase that can
run for more than one round, then sign-off. This notebook builds that review
as a `StateGraph`: a parser, three parallel reviewer agents, a synthesizer, a
negotiation gate that sends open blockers to a negotiate step, which loops
revised terms back through the reviewers, and a sign-off agent that returns
a typed decision.

The processor here is a marketing-analytics service that ingests the
controller's customer personal data, so a weak erasure service-level
agreement (SLA) or a vague international-transfer mechanism is a real
compliance gap, not a paperwork detail: the vendor's breach-notification
window must fit inside the 72-hour timeline of Article 33 of the General
Data Protection Regulation (GDPR) that the controller owes its own
supervisory authority. The data-protection officer (DPO) writes the typed
sign-off.

What this notebook covers:

- `Send`: `scatter_reviewers` returns three `Send`s, so the legal, governance
  and compliance reviewers run concurrently and `synthesize` collects their
  findings from state.
- `add_conditional_edges` on a graph with `allow_cycles=True`:
  `negotiation_gate` routes to `negotiate` while blockers remain, and revised
  terms go back to `parse` for another review round, with a hard cap of four
  review rounds (three negotiation passes).
- `interrupt()`: the negotiate node pauses for counsel's answer and the run
  resumes with `Command(resume=...)`; here `main()` scripts the answers.
- `output_schema=ContractDecision`: the sign-off agent returns a typed
  `ContractDecision`.

The diagram shows the graph `build_review_graph` wires: the three reviewers
fan out from the scatter node, the negotiation gate decides between sign-off
and negotiation after each synthesis, and the `Command` that `negotiate`
returns chooses between another review round and sign-off:

{{ locus_diagram contract-review }}

## Key lines

`build_review_graph` declares the loop with `GraphConfig(allow_cycles=True)`,
fans the parsed clauses out to the three reviewers through the scatter node,
hands the routing decision after `synthesize` to `negotiation_gate`, and
wires `negotiate` back to `parse` so revised terms are reviewed again. The
cycle check in `add_edge` follows fixed edges only, and this loop closes
through the conditional edge after `synthesize`, so the flag records the
loop rather than unlocking it; the round cap in `negotiation_gate` and
`max_iterations=20` are what bound the run:

{{ locus_source examples/notebook_79_contract_review.py::build_review_graph }}

`negotiate` pauses the run with `interrupt()`, whose payload carries the
round number, the open blockers and the options; when the run resumes, the
node runs again from the top and `interrupt()` returns counsel's answer. The
node turns that answer into a `Command`, and the executor follows the
`Command`'s `goto` instead of the fixed edge: a walk-away or a resolution
goes to `sign_off`, and any other text is treated as the vendor's redline and
sent back to `parse`:

{{ locus_source examples/notebook_79_contract_review.py::negotiate }}

`_drive` runs the graph once, then resumes each negotiation interrupt with `Command(resume=...)` until `result.interrupt` is `None`. The graph has no checkpointer, so a resumed run starts from whatever the `Command` carries in `update`: `_drive` passes back the `final_state` the interrupted run returned, which includes the executor's resume marker and the model object that rides along in state, and sets `__model__` again explicitly.

Counsel's answers are scripted: `main()` supplies two vendor redlines and then a RESOLVED answer, and `_drive` answers RESOLVED on its own if the list runs out, so the run completes without a prompt:

{{ locus_source examples/notebook_79_contract_review.py::_drive }}

## Run it

```bash
python examples/notebook_79_contract_review.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

The sign-off step relies on `output_schema=ContractDecision`, so the typed
decision needs a model that can honour a JSON schema. With the bundled mock
model the three negotiation rounds complete, but `sign_off` raises because the
agent returns no parsed `ContractDecision`; `execute()` records that as a
failed node and returns `success=False` instead of raising, and `main()`,
which does not check `result.success`, reports the decision as missing. Pin
a model that supports structured output:

```bash
export LOCUS_MODEL_PROVIDER=openai
export LOCUS_MODEL_ID=gpt-4o
python examples/notebook_79_contract_review.py
```

## See also

- [Interrupts and human-in-the-loop](../concepts/interrupts.md) — how
  `interrupt()` pauses a run and how the answer comes back.
- [StateGraph](../concepts/multi-agent/graph.md) — nodes, conditional edges,
  cycles and `Send` fan-out.
- [Structured output](../concepts/structured-output.md) — what
  `output_schema` puts on `AgentResult` and which providers support it.

## Source

??? note "Full source — `examples/notebook_79_contract_review.py`"

    ````python
    --8<-- "examples/notebook_79_contract_review.py"
    ````
