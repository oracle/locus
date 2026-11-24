# Agent Streaming

`agent.run(prompt)` returns an async iterator of events. Watch the agent
think, call tools, and terminate — live, in order. This is what lets you
build progress bars, dashboards, audit logs, and Server-Sent Events (SSE)
endpoints.

This notebook follows a data-privacy scenario: triaging data-subject
access requests (DSARs) under Article 15 of the General Data Protection
Regulation (GDPR). The agent extracts personally identifiable information
(PII), checks consent, classifies requests, and files reports — all on
fabricated subjects (`example.com`, `example.org` and `example.net`
addresses, 555-01xx phone numbers).

What this notebook covers:

- The event types: `ThinkEvent`, `ToolStartEvent`, `ToolCompleteEvent`,
  `TerminateEvent`.
- Event-level streaming by default — the assistant's text arrives in one
  piece when a step completes — and token-level output with
  `stream_tokens=True`, which adds `ModelChunkEvent`s as the model emits
  text (quoted under Key lines below).
- Filtering with `isinstance(event, EventType)`.
- Building a live console UI from the stream.
- Rolling event counts into per-run metrics.
- Drawing a progress bar from `ToolCompleteEvent`.
- A pointer to `StructuredStream`, which parses the chunk stream into
  partial Pydantic objects and so also needs `stream_tokens=True`.

Prerequisite: the [Agent with Tools](notebook_03_agent_with_tools.md)
notebook.

The drawing shows the paths each DSAR run in this notebook can take:
`agent.run(...)` yields the loop events on every run and adds a
`ModelChunkEvent` per chunk only with `stream_tokens=True`; the `async for`
loops written by hand here could equally hand each event to a console, SSE
or buffering handler, or wrap the run in `StructuredStream` for partial
Pydantic objects.

{{ locus_diagram streaming-paths }}

## Key lines

One `async for` over `agent.run(...)` receives every event type and
dispatches on it with `isinstance`:

{{ locus_source examples/notebook_06_agent_streaming.py::example_all_events }}

Keeping only the tool events turns the same stream into an audit log:

{{ locus_source examples/notebook_06_agent_streaming.py::example_event_filtering }}

Passing `stream_tokens=True` adds `ModelChunkEvent`s: text deltas arrive on
`content` and, when the model streams it, chain-of-thought on its own
`reasoning` channel, while the tool and termination events are unchanged:

{{ locus_source examples/notebook_06_agent_streaming.py::example_token_streaming }}

## Run it

```bash
python examples/notebook_06_agent_streaming.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## See also

- [Streaming](../concepts/streaming.md) — consuming the event stream, and
  the handlers in `locus.streaming`.
- [Events](../concepts/events.md) — the event classes and their fields.
- [Streaming partial objects](../concepts/structured-output.md#streaming-partial-objects)
  — `StructuredStream` in full.
- [SSE streaming](notebook_64_sse_streaming.md) — the same events turned
  into SSE frames by `SSEHandler` and `AsyncSSEHandler` (notebook 64).

## Output

Running it against a live model — OCI Generative AI, `openai.gpt-4.1` — prints the console view of a two-tool triage (`[Thinking...]`, `[Running] … Done`), the counts and timings collected from the event stream, a progress bar advanced by tool completions and a reply streamed as it is generated; Parts 1 and 3 and the closing `StructuredStream` note are elided (`...`); the model's wording and the timings differ from run to run:

```text
=== Part 2: Console UI ===

Query: Triage data-subject request DSR-1042 and check its marketing consent

[Thinking...]
  Planning to call: fetch_record
[Running] fetch_record... Done (505ms)
[Thinking...]
  Planning to call: check_consent
[Running] check_consent... Done (0ms)
[Thinking...]

[Complete] complete

Verdict: The data-subject request DSR-1042 pertains to jane.doe@example.com and involves a "marketing profile" data category. Consent for this category was withdrawn 9 days ago, and the retention basis has expired. This indicates there is no longer valid consent to retain or process this data.

...

=== Part 4: Collecting Metrics ===

Execution Metrics:
  Think events:    3
  Tool starts:     2
  Tool completes:  2
  Tool time:       0.3ms
  Iterations:      3
  Total time:      3927.1ms

=== Part 5: Progress Tracking ===

Processing pending queue...
  [###-------] 33% - fetch_records complete
  [######----] 67% - scan_records complete
  [##########] 100% - file_report complete

Done! Final message: Fetch, scan, and reporting on the pending queue is complete:

- 3 requests were processed (DSR-1042, DSR-1043, DSR-1044).
- Total data scanned: 226 bytes.
- Case file checksum: sha256:a295d7e9db89.
- Triage report has been successfully filed.

Let me know if you need details on any specific request or further actions!

=== Part 6: Token-level streaming ===

Streaming the reply as it is generated:


  [tool: extract_pii]
  Analysis:
- The record contains an email address: jane.doe@example.com.
- The text also references a phone number (555-0142).

Both the email address and the phone number are types of personally identifiable information (PII) and should be handled according to data privacy regulations.

  stop reason: complete
  chain-of-thought streamed separately: 0 chars

...

============================================================
Next: Notebook 07 — Structured output: typed support schemas a help desk can consume
============================================================
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_06_agent_streaming.py`"

    ````python
    --8<-- "examples/notebook_06_agent_streaming.py"
    ````
