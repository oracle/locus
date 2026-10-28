# Supervisor + Critic

A privacy analyst gathers evidence notes, an author drafts a
data-exposure report, and a skeptical reviewer either approves or sends
it back for revision. The loop caps at two revisions to bound runtime.

The point is the *last* step, not the loop: a report that reads well is
not the same as a report that is grounded. Before anything ships, the
reviewer runs the drafted finding through `ground_finding` — the GSAR
grounding gate from `locus.control`. A finding is emitted only when its
evidence partition clears the proceed threshold; otherwise the call
returns an `Abstention` and nothing reaches the privacy queue. An
unproven PII-exposure claim is a false positive *by construction* and
never ships.

This notebook covers:

- Control flow as a `StateGraph` with conditional edges — no
  hand-rolled `while True`.
- Each role is its own `Agent` with a role-specific system prompt.
  Roles communicate only through state keys (`notes`, `draft`,
  `revision_request`).
- The reviewer node where prose review meets mechanical grounding:
  `ground_finding(...)` scores the evidence `Partition` of typed
  `Claim`s and returns an `Evidence` or an `Abstention`;
  `is_finding(...)` narrows the union.
- `stream(mode=StreamMode.VALUES)` emits one event per node completion
  for live UI updates; the terminal event carries the final state, so
  the graph runs once.

{{ locus_diagram supervisor-critic-loop }}

The seeded scenario: DLP scan `DLP-4471` flags an unmasked `email`
column in the analytics `customer_export` view. The grounded claims
trace to the DLP scan rows and the data-lineage graph; the lone
unproven claim — that the records were actually accessed by an
unauthorized third party — has no backing evidence, so the grounding
gate keeps it out of the shipped finding. The drafted finding is tagged
with standard risk-taxonomy ids (the `OwaspLLM` and `AtlasTechnique`
enums in the code) so the artifact files straight into a privacy
register or DPIA.

## Prerequisites

- Basic graph.
- Agent handoff, for an alternative shape.
- GSAR grounded findings, for the grounding primitive in depth.

## Run

```bash
python examples/notebook_21_supervisor_critic_loop.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one. The grounding gate is deterministic and exercises the same admit/abstain path on either.

## Source

````python
--8<-- "examples/notebook_21_supervisor_critic_loop.py"
````
