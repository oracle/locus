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

## Source

??? note "Full source — `examples/notebook_02_basic_agent.py`"

    ````python
    --8<-- "examples/notebook_02_basic_agent.py"
    ````
