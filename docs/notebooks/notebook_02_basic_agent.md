# Basic agent

The smallest end-to-end example. Build LEDGER, a tier-1
transaction-triage agent for a payments team, send it a prompt two ways
(blocking and streaming), and look at what comes back.

The sample transactions map to common payments patterns: card testing
(many small declines then a charge), an authorized recurring
subscription, and a friendly-fraud chargeback.

What this notebook covers:

- How an `Agent` pairs a model with a system prompt.
- The difference between `agent.run_sync(...)` (one result) and
  `agent.run(...)` (an async stream of events).
- The fields on `AgentResult` the example reads: `message`, `success`,
  `stop_reason`, `confidence` and `metrics`.
- Reusing the same agent to triage many transactions in a row.

Both the blocking and the streaming call drive the same agent loop;
LEDGER has no tools, so each triage passes the termination checks, goes
through Think once and ends on the model's answer without a trip through
Execute:

{{ locus_diagram agent-loop }}

## Key lines

One model and one system prompt make LEDGER; a short `run_sync` call
confirms the provider answers before any triage begins:

{{ locus_source examples/notebook_02_basic_agent.py::example_create_agent }}

The blocking form waits for the whole run and hands back one
`AgentResult`, with the message, success flag and stop reason on it:

{{ locus_source examples/notebook_02_basic_agent.py::example_run_sync }}

The streaming form yields typed events as the loop runs, so a caller sees
each step instead of one answer at the end:

{{ locus_source examples/notebook_02_basic_agent.py::example_run_streaming }}

## Run it

```bash
python examples/notebook_02_basic_agent.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Output

Running it against a live model — OCI Generative AI, `openai.gpt-4.1` — prints the provider round trip behind a one-word smoke reply, LEDGER's triage of a card-testing pattern from `run_sync`, the `think` and `terminate` events the streaming form yields, the fields and metrics on `AgentResult`, and one triage per transaction in the queue; the model's wording and the timings differ from run to run:

```text
=== Part 1: Creating an Agent ===

Agent created with model: OCIChatCompletionsModel
System prompt: You are LEDGER, a tier-1 transaction triage agent for a paym...
  [provider call: 2.68s · 95→1 tokens]
  Smoke reply: ready

=== Part 2: Blocking Execution (run_sync) ===

Transaction: txn_8841 (Northwind Coffee, $1.00)
LEDGER: This pattern is consistent with possible card testing: many small transactions from the same BIN across multiple merchants in a short period. A human should review for fraud and consider blocking further authorizations from this BIN or velocity controls.
Success: True
Stop reason: complete

=== Part 3: Streaming Execution (run) ===

Transaction: txn_8843 (Harbourline Outfitters, $289.40 chargeback)
Events:
  think: This pattern is consistent with possible friendly fraud or a cardholde
  terminate: This pattern is consistent with possible friendly fraud or a cardholde

=== Part 4: Understanding Results ===

AgentResult fields:
  .message     = This pattern is consistent with a recurring subscription or membership payment that has proceeded without dispute for 9 months. No action needed unless the cardholder raises an issue; continue to monitor for any changes in pattern or new disputes.
  .success     = True
  .stop_reason = complete
  .confidence  = 0.0

Metrics:
  .metrics.iterations  = 1
  .metrics.tool_calls  = 0
  .metrics.duration_ms = 1968

=== Part 5: A Triage Queue ===

▸ txn_8841 — card testing
  This pattern is consistent with possible card testing activity: many small-amount authorizations from the same BIN across multiple merchants in a short time frame. A human should review the transaction for fraud risk, consider blocking the BIN temporarily, and investigate related activity.

▸ txn_8842 — recurring subscription
  This pattern is consistent with a recurring subscription payment that has been stable and undisputed. No action needed unless the cardholder reports an issue.

▸ txn_8843 — chargeback
  This pattern is consistent with a possible friendly fraud or a dispute by an authorized user, given the delivery was signed for at the billing address and prior undisputed orders exist. A human should review the delivery proof and account history, and consider submitting evidence of delivery and account usage to contest the chargeback.

============================================================
Next: Notebook 03 — Giving an Agent Tools
============================================================
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_02_basic_agent.py`"

    ````python
    --8<-- "examples/notebook_02_basic_agent.py"
    ````
