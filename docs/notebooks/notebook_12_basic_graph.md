# Basic graph

Build a data subject access request (DSAR) triage workflow as a graph of nodes that pass state to each other. A `StateGraph` is a directed graph in which each node is a function — async here — that receives the current state and returns a dict of updates to merge back.

Use it when one `Agent` is not enough: a multi-step privacy pipeline (parse → enrich → verdict) or fan-out and fan-in across enrichment lookups. The same primitives carry the branching logic of [Notebook 13](notebook_13_conditional_routing.md) and the human approval gates of [Notebook 15](notebook_15_human_in_the_loop.md) before a DSAR response goes out.

What this notebook covers:

- Nodes and edges; the `START` and `END` sentinel node ids.
- Sequential and parallel flow on the same primitives; conditional routing
  follows in Notebook 13.
- `GraphResult` — final state plus per-node status, timing, and order. The
  final state also keeps each node's raw output under a `_node_<id>` key,
  which the Part 3 `receives` lines show and the final-state printouts
  filter out.
- Streaming node updates and pushing custom progress events from inside a
  node.
- An `Agent` embedded inside a graph node.

The drawing shows the general shape of a `StateGraph`: `START` hands the
input state to a chain of nodes joined by fixed edges, each node merging its
return value into the shared state, and a conditional edge after the last
node either loops back for another pass or reaches `END` with the final
state. The triage pipelines in this notebook are built from the same
`add_node` and `add_edge` calls; the dashed edges, chosen by a router, are
the one primitive they do not use yet:

{{ locus_diagram graph }}

## Key lines

Three triage nodes, `parse`, `enrich` and `verdict`, are added by name and
joined by fixed edges from `START` to `END`; `graph.execute` runs them in
order, and each node's return value merges into the state the next one
receives:

{{ locus_source examples/notebook_12_basic_graph.py::example_sequence }}

Nodes that become ready in the same step run concurrently by default
(`GraphConfig.parallel` is `True`); the example sets
`graph.config.parallel = True` only to make that explicit. The three
enrichment nodes, each with an edge from `START`, are started together,
their results are merged into the state one after another, and the single
`combine` node they all lead to runs in the next step and reads them from
the merged state:

{{ locus_source examples/notebook_12_basic_graph.py::example_parallel }}

`graph.stream` drives the run on a background task and yields a
`StreamEvent` as each node completes; in `UPDATES` mode the event carries
the dict the node returned. `emit_custom` pushes a `CUSTOM` event onto the
same stream from inside a node while it is still running (its `node_id` is
`None` unless the caller passes one), and is a silent no-op under
`graph.execute`, so the node body works unchanged either way:

{{ locus_source examples/notebook_12_basic_graph.py::example_streaming }}

## Run it

```bash
python examples/notebook_12_basic_graph.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Parts 1, 2, 4 and 7 call the model once each; Parts 3, 5 and 6 make no
model call. To stay offline regardless of what `~/.oci/config` holds, force
the mock model:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_12_basic_graph.py
```

## See also

- [StateGraph](../concepts/multi-agent/graph.md) — the primitive in full:
  per-node policies, `Send` fan-out and Mermaid output.
- [Graph streaming](../concepts/graph-streaming.md) — stream modes,
  `emit_custom` and real-time delivery.
- [Conditional routing](notebook_13_conditional_routing.md) — the
  `add_conditional_edges` router the diagram shows (Notebook 13).

## Output

Running it against a live model — OCI Generative AI, `openai.gpt-4.1` — prints a one-node graph around a model call, a three-node sequence, the keys each node receives as state accumulates, a fan-out whose branches run concurrently, the fields of `GraphResult`, the `CUSTOM` and `UPDATE` events of a streamed run, and an agent called inside a node; the model's wording and the timings differ from run to run:

```text
=== Part 1: A one-node graph ===

  [model call: 2.29s · 49→39 tokens]
Input:   subject = 'Please delete all data you hold about me'
Output:  assessment = 'This message is a clear data erasure request (right to be forgotten) under data protection laws such as GDPR and should be promptly forwarded to the appropriate privacy or compliance team for verification and processing.'
Success: True

=== Part 2: A sequence of nodes — parse → enrich → verdict ===

  [model call: 1.10s · 42→14 tokens]
Input:    request = '  please export my health record and contact details  '
Parsed:   has_content = True
Enriched: mentions_special_category = True
Verdict:  8 tokens, special_category=True — Priority triage: expedite due to mention of special-category personal data.

=== Part 3: How state accumulates ===

Executing graph...
  Intake receives:    ['request_id']
  Correlate receives: ['request_id', 'intake_note', 'exposure_score', '_node_intake']
  Score receives:     ['request_id', 'intake_note', 'exposure_score', '_node_intake', 'correlate_note', 'combined_exposure', '_node_correlate']

Final state:
  request_id: DSAR-1042
  intake_note: request ingested
  exposure_score: 10
  correlate_note: matched a sibling request
  combined_exposure: 20
  score_note: final score issued
  final_exposure: 25

=== Part 4: Fan-out and fan-in ===

  [model call: 0.91s · 64→1 tokens]
Input: 'subject id:8842 email jordan@example.com phone +1-555-0100'
Enrichment: {'sensitivity': 'high', 'identifiers': 2, 'consent': 'withdrawn-no-active-basis'}
Time: 909ms (parallel nodes run concurrently)

=== Part 5: Reading GraphResult ===

GraphResult fields:
  .success         = True
  .graph_id        = graph_6b3694e9
  .duration_ms     = 0.0
  .iterations      = 1
  .execution_order = ['normalize']

  .final_state:
    raw_score: 21
    normalized: True
    exposure_score: 42

  .node_results:
    normalize: status=completed

=== Part 6: Streaming with emit_custom ===

Streaming UPDATES + CUSTOM events as they arrive:
  [CUSTOM]  enrich: {'phase': 'starting', 'record_count': 21}
  [CUSTOM]  enrich: {'phase': 'halfway', 'matched_so_far': 42}
  [UPDATE]  enrich: {'matched_records': 42}
  [UPDATE]  report: {'report_entries': 52}

=== Part 7: An Agent inside a node ===

  [model call: 1.03s · 45→22 tokens]
AI summary: We process and analyze email open data by linking it to individual CRM profiles to evaluate and optimize marketing campaign effectiveness.

============================================================
Next: Notebook 13 — Routing requests by privacy risk
============================================================
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_12_basic_graph.py`"

    ````python
    --8<-- "examples/notebook_12_basic_graph.py"
    ````
