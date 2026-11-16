# Supervisor and critic loop

A privacy analyst gathers evidence notes, an author drafts a
data-exposure report, and a skeptical reviewer either approves or sends
it back for revision. The loop caps at two revisions to bound runtime.
There is no supervisor agent deciding who runs next: the conditional
edge after `review` plays that role, routing back to the author on
REVISE and to `END` on APPROVE or at the revision cap.

The point is the *last* step, not the loop: a report that reads well is
not the same as a report that is grounded. Before anything ships, the
reviewer node runs the finding's evidence partition through
`ground_finding`, the [GSAR (typed grounding)](../concepts/gsar.md) gate
from `locus.control`. A finding is emitted only when that partition
clears the proceed threshold; otherwise the call returns an `Abstention`
and nothing reaches the privacy queue. A personally identifiable
information (PII) exposure finding that its evidence cannot support is a
false positive *by construction* and never ships.

The seeded scenario: data loss prevention (DLP) scan `DLP-4471` flags an
unmasked `email` column in the analytics `customer_export` view. Three
claims trace to the DLP scan row, the schema review and the data-lineage
graph; a fourth, that the records were accessed by an unauthorized third
party, has no evidence behind it. The candidate finding handed to the
gate carries standard risk-taxonomy IDs (the `OwaspLLM` and
`AtlasTechnique` enums from the
[threat taxonomy](../api/control.md)), so a shipped
`Evidence` can be filed directly in a privacy register or data protection
impact assessment (DPIA).

What this notebook covers:

- Control flow as a `StateGraph` with conditional edges — no
  hand-rolled `while True`.
- Each role is its own `Agent` with a role-specific system prompt.
  Roles communicate only through state keys (`notes`, `draft`,
  `revision_request`).
- The reviewer node, where prose review meets mechanical grounding:
  `ground_finding(...)` over a `Partition` of typed `Claim`s, and
  `is_finding(...)` to narrow its result.
- `stream(mode=StreamMode.VALUES)` emits one event per node completion,
  which lets a user interface show progress live; the terminal event
  carries the final state, so the graph runs once.

## Prerequisites

- [Basic graph](notebook_12_basic_graph.md).
- [Agent handoff](notebook_25_agent_handoff.md), for an alternative shape.
- [GSAR typed grounding](notebook_55_gsar_typed_grounding.md), for the
  grounding primitive in depth.

The drawing follows the graph as `build_review_loop_graph` wires it:
`gather` hands the analyst's notes to `draft`, and on every pass the
review node runs two steps, the reviewer agent's verdict and then the
grounding gate. The verdict routes the loop (back to `draft` on REVISE
until two revisions are done, to `END` on APPROVE or at that cap); only
the gate's result from the last pass decides what ships:

{{ locus_diagram supervisor-critic-loop }}

## Key lines

`build_review_loop_graph` adds one node per role and joins them with fixed
edges from `START` through `gather` and `draft` to `review`; the conditional
edge after `review` hands the state to `route_after_review` and maps its
answer to `draft` (revise) or `END` (done):

{{ locus_source examples/notebook_21_supervisor_critic_loop.py::build_review_loop_graph }}

`review_node` first asks the reviewer `Agent` for a prose verdict and then,
whatever that verdict says, calls `ground_finding` over the finding's
evidence partition; the typed claims are what is scored, not the draft
text, and the result is an `Evidence` carrying the OWASP LLM and MITRE
ATLAS IDs, or an `Abstention`:

{{ locus_source examples/notebook_21_supervisor_critic_loop.py::review_node }}

`_evidence_partition` builds the `Partition` the gate scores: three grounded
`Claim`s, each with its evidence references, and the third-party-access
inference in `ungrounded`, where it counts against the grounding score and,
having no evidence references of its own, adds nothing to a shipped
`Evidence`'s `evidence_refs`; it stays there until access logs confirm it:

{{ locus_source examples/notebook_21_supervisor_critic_loop.py::_evidence_partition }}

## Run it

```bash
python examples/notebook_21_supervisor_critic_loop.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one. The grounding gate is
deterministic and exercises the same admit/abstain path with a live model or
the mock.

To run offline regardless of what `~/.oci/config` holds:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_21_supervisor_critic_loop.py
```

## Source

??? note "Full source — `examples/notebook_21_supervisor_critic_loop.py`"

    ````python
    --8<-- "examples/notebook_21_supervisor_critic_loop.py"
    ````
