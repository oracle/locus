# State Reducers

Control how state updates combine instead of overwriting each other.
By default, when two nodes write to the same field, the second one
wins — so the fraud screen's signals vanish the moment the Address
Verification Service (AVS) check reports. A reducer is a function
attached to a field that defines how to merge an incoming update —
append to a list, sum numbers, merge dicts, keep the maximum, and so
on — so every payment check's signals land in one authorization state.

What this notebook covers:

- `Annotated[type, reducer]` on a Pydantic state schema declares the rule.
- Built-in reducers: `add_messages`, `add_numbers`, `merge_dict`,
  `append_list`, `last_value`.
- `@reducer` turns any `(current, new) -> merged` function into a custom one.
- Multiple reducers on one schema — each field merges independently.
- Two nodes that each call a large language model (LLM) and append to
  the same case log.

The drawing shows a `StateGraph` with a review loop: `START` hands the input state to `plan`, and fixed edges carry it through `investigate` and `report` to `review`; after `review` a conditional edge reads the merged state to loop back or finish.

That merge is the step this notebook is about. Each time a node returns, its output is merged into the state through the schema's reducers before the next node or the router reads it, which is how the router sees what `review` just wrote.

The payment-check chains here — `fraud`, then `avs`, then `threeds` — are straight lines of fixed edges, so the only thing that changes between Part 1, the same chain with no reducer, and Part 1b is how that merge treats the `signals` field:

{{ locus_diagram graph }}

## Key lines

Annotating `signals` with `append_list` on the Pydantic state schema is
the only difference from Part 1: the same three checks run in the same
order, and all three signals survive because each later update is
appended to what the field already holds instead of replacing it:

{{ locus_source examples/notebook_14_state_reducers.py::example_with_reducers }}

`@reducer` turns any `(current, new) -> merged` function into a
reducer. Here `max_value` keeps the highest fraud score reported by any
of the three checks and `unique_append` collects flagged cards without
duplicates (`locus.core` ships `max_value` and `unique_append_list` as
built-ins; the example writes its own to show the decorator):

{{ locus_source examples/notebook_14_state_reducers.py::example_custom_reducer }}

One authorization schema with a different reducer on each field:
signals append, the risk score sums (the allowlist review returns a
negative delta, which `add_numbers` sums in as a deduction), adjustments
merge as a dict, status keeps the latest write and the audit messages
accumulate:

{{ locus_source examples/notebook_14_state_reducers.py::example_complex_state }}

## Run it

```bash
python examples/notebook_14_state_reducers.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Only Part 6 sends two short prompts to a model; Parts 1 to 5 are plain
graph code, so the offline variant shows every merge:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_14_state_reducers.py
```

## Output

Running it against a live model — OCI Generative AI, `openai.gpt-4.1` — prints the two signals a plain state key loses and `append_list` keeps, each built-in reducer's merged value, two custom reducers, `last_value` next to an accumulated timeline, a schema mixing reducers, and two model-producing nodes appending to one `add_messages` list; the model's wording and the timings differ from run to run:

```text
=== Part 1: Why reducers exist ===

Without reducers: signals = ['3-D Secure authentication failed']
  (Only the 3-D Secure signal - we lost the fraud and AVS signals!)

=== Part 1b: Same graph, with append_list ===

With append_list reducer: signals = ['Velocity: 8 charges in 2 minutes', 'AVS mismatch: billing ZIP does not match', '3-D Secure authentication failed']
  (All three checks' signals preserved!)

=== Part 2: Built-in reducers ===

add_messages reducer:
  [user] New chargeback on account A-204 — anything related?
  [assistant] Correlated: same card BIN as case 1042.

add_numbers reducer:
  flagged_txn_count = 5

=== Part 2b: merge_dict ===

Final gateway config: {'step_up_3ds': True, 'timeout': 60, 'retries': 3}
  (All settings merged together)

=== Part 3: Custom reducers ===

Peak fraud score (max): 90
Flagged cards (unique): ['411111******1111', '511111******2222', '601111******3333']

=== Part 4: last_value ===

Status (last value): declined
Timeline (accumulated): ['Risk screening complete', 'Manual review complete', 'Decline decision logged']

=== Part 5: Mixing reducers on one schema ===

Final Authorization State:
  Signals: 2 signals
  Risk score: 58.50
  Adjustments: {'allowlist_review': 6.5}
  Status: finalized
  Messages: 4 audit entries

=== Part 6: add_messages with two LLM-producing nodes ===

  [model call (summary): 1.90s · 41→32 tokens]
  [model call (recommendation): 0.99s · 31→8 tokens]
  [summary] Fraudsters used stolen cards for rapid-fire purchases, but mismatched AVS data and failed 3-D Secure authentication flagged the scam before any goods shipped.
  [recommendation] Flag transaction; escalate for immediate investigation.

============================================================
Next: Notebook 15 — Human approval gates for production changes
============================================================
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_14_state_reducers.py`"

    ````python
    --8<-- "examples/notebook_14_state_reducers.py"
    ````
