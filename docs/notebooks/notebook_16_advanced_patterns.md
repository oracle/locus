# Advanced graph patterns

The graph primitives that turn a `StateGraph` into a general-purpose
runtime, shown here as a customer-support ticket pipeline. They apply
once a fixed topology is no longer enough: routing decided inside a
node, fan-out to many tickets (or many classifiers), reusable
subgraphs, and a key/value store that persists across graph runs in
the same process.

The pipeline — call it HELPDESK — classifies an incoming ticket and
routes it to the right queue, auto-acknowledges a batch of tickets in
parallel, asks three classifiers about one ticket at once, validates
ticket payloads with a reusable sub-pipeline, remembers customers
across runs and, in its last part, lets a model choose the `Command`
target for a ticket update.

The triage step is confidence-gated: a ticket matched by both strong
classifiers is auto-resolved, one matched by a single strong classifier
gets a drafted reply for an agent to review, and one flagged only by the
length heuristic is escalated to a human agent — so an uncertain
automated decision never reaches the customer on its own.

What this notebook covers:

- `Command(update=..., goto=...)` — write state and pick the next node
  in one return value.
- `goto()` and `end()` — short helpers for common `Command` shapes.
- `scatter("worker", items, key=...)` — fan a list of tickets out so the
  worker node runs once per ticket, in parallel.
- `broadcast(nodes, payload)` — send one ticket to several different
  classifier nodes, then adjudicate the merged signals.
- Subgraph-as-node — call one `StateGraph` from inside another to
  validate a ticket payload.
- `InMemoryStore` — key/value space that outlives a single graph run and
  is shared across runs in one process (data is lost when the process
  exits).

The drawing shows the shared `StateGraph` shape — fixed edges carry the
merged state from node to node and a conditional edge after `review`
picks the next hop; in this notebook that decision moves inside the
node, which returns either a `Command` naming its own successor or a
list of `Send` objects from `scatter` or `broadcast`, whose targets run
at once on the current state plus each payload before a fixed edge
carries the merged results to the fan-in node:

{{ locus_diagram graph }}

## Key lines

The `classify` node has no outgoing edges; the `Command` that
`classify_ticket` returns writes the queue into state and names the
branch that runs next, replacing a conditional edge and a separate state
writer:

{{ locus_source examples/notebook_16_advanced_patterns.py::example_command_routing }}

`broadcast` sends one ticket to the intent, knowledge-base and heuristic
classifiers at once; the fan-in node merges their `send_*` results and
auto-resolves only when both strong signals agree, drafts a suggestion
on one, and escalates when nothing stronger than the length heuristic
fired:

{{ locus_source examples/notebook_16_advanced_patterns.py::example_broadcast }}

The combined pipeline reads VIP status from an `InMemoryStore` to choose
between priority and standard handling, fans the priority actions out
with `scatter`, and records every ticket in the store under the
customer's namespace:

{{ locus_source examples/notebook_16_advanced_patterns.py::example_combined }}

## Run it

```bash
python examples/notebook_16_advanced_patterns.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Parts 1 to 5 are deterministic and make no model call; only the Part 6
`triage` node sends a ticket update to a model. To stay offline
regardless of the OCI profile on the machine, force the mock model; it
answers in prose rather than with one of the three labels, so the clamp
in `triage` escalates every sample:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_16_advanced_patterns.py
```

## See also

- [StateGraph](../concepts/multi-agent/graph.md) — the primitive in full:
  per-node policies, `Send` fan-out and Mermaid output.
- [The built-in store](../concepts/checkpointers.md#the-built-in-store-inmemorystore)
  — `InMemoryStore` and the `BaseStore` backends that persist beyond one
  process.
- [Conditional routing](notebook_13_conditional_routing.md) — the
  `add_conditional_edges` router the diagram shows (notebook 13).
- [Scatter-gather with Send](notebook_20_map_reduce_support_triage.md) —
  `Send` fan-out to analyst agents and a reducer that reads their results
  back from state (notebook 20).

## Output

Running it against a live model — OCI Generative AI, `openai.gpt-4.1` — prints the branch and owner each `Command` selects, `goto()` and `end()` on a support-hours token, `scatter()` over three tickets, the confidence-gated triage after `broadcast()`, a subgraph accepting and rejecting tickets, a `Store` that remembers the previous outcome, and a model choosing the `Command` target; the model's wording and the timings differ from run to run:

```text
=== Part 1: Command — state and routing in one return ===

billing: branch=billing_resolution, owner=billing team
technical: branch=technical_support, owner=engineering support
unknown: branch=general_triage, owner=front-line agent

=== Part 1b: goto() and end() ===

Hours token 'hours-open': Live agent picked up the ticket
Hours token 'hours-closed': Outside support hours — queued for the morning

=== Part 2: scatter() ===

Acknowledged 3 tickets
Results: ['auto-reply sent for password reset', 'auto-reply sent for refund request', 'auto-reply sent for shipping delay']

=== Part 2b: broadcast() + confidence-gated triage ===

confident: AUTO_RESOLVE (confidence=high, reasons=['intent matched a known self-service issue', 'ticket carries a vetted knowledge-base tag'])
suspected: ESCALATE (confidence=low, reasons=['ticket length exceeded the heuristic threshold'])

=== Part 3: Subgraph as a node ===

Well-formed ticket: state = accepted
Missing customer field: state = rejected

=== Part 4: Store — memory that outlives one graph run ===

Ticket 1:
  New issue for this team — handle with care

Ticket 2:
  Seen before — last outcome: resolved in 40s

=== Part 5: Command + scatter + Store together ===

Standard customer, urgency 20: standard handling
Standard customer, urgency 95: priority handling
VIP customer, urgency 10: priority handling

=== Part 6: LLM-decided Command target ===

  [model call: 1.60s · 47→1 tokens]
  'The customer confirmed the password rese…' → closed as resolved
  [model call: 1.57s · 48→1 tokens]
  'Sent the customer troubleshooting steps …' → awaiting customer reply — kept open
  [model call: 0.99s · 49→3 tokens]
  'Customer is furious about a double charg…' → escalated to a senior agent

============================================================
Next: Notebook 17 — Resilient cloud-provisioning workflows
============================================================
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_16_advanced_patterns.py`"

    ````python
    --8<-- "examples/notebook_16_advanced_patterns.py"
    ````
