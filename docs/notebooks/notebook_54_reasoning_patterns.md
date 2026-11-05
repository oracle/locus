# Reasoning Patterns

Walks the reasoning toolkit one piece at a time against a
data-privacy postmortem: a privacy-team exercise seeded a synthetic
personally identifiable information (PII) canary, a fabricated Social
Security number, into a data export and the redaction pipeline never
flagged it. The parts that call a model print a
`[model call Part N: X.XXs · prompt→completion tokens · iters=N]` banner
so the round-trip is visible.

The thesis the reasoning layer enforces: an ungrounded claim in a
postmortem is how a false root cause gets written into a runbook, so each
claim is scored against tool evidence before it is allowed to stand.

Read end to end, the parts form one pipeline — typed claims, a grounding
score, replan guidance, a causal chain and Reflexion over a live run —
which the [Reasoning](../concepts/reasoning.md) concept page describes
piece by piece.

What this notebook covers:

- `@tool` + `Agent(tools=...)` — let the agent call real Python functions
  over the redaction pipeline's log (`read_redaction_logs`) and its data
  loss prevention (DLP) stats (`query_dlp`).
- `Reflector` and `Agent(reflexion=True)` — Reflexion is a self-critique
  step that needs no model call: the `Reflector` scores a trajectory from
  its successful calls, failed calls and repeated tool patterns, and the
  agent flag runs that check inside the loop and feeds its guidance back
  as a system message.
- `Agent(output_schema=...)` — typed JSON for postmortem claims and event
  timelines.
- `GroundingEvaluator` — score each claim against tool evidence and
  decide whether to replan.
- `CausalChain` / `build_causal_chain` — build and walk a cause/effect
  graph from the stale classifier and saturated scan queue to PII shipped
  in the clear, then flag a cycle as a conflict.

## Prerequisites

- A live, JSON-capable model: OCI Generative AI (the default when
  `~/.oci/config` has a usable profile), or an OpenAI or Anthropic key
  with the matching `LOCUS_MODEL_PROVIDER`.
- For Parts 4 and 6, a model that returns JSON matching the schema; a
  reply that does not parse into `ClaimList` or `EventList` stops the run
  with a `RuntimeError` that names stronger models. Under the bundled mock
  model, `check_structured_output_capable()` exits cleanly before Part 1
  runs.

## Key lines

Part 1 gives the postmortem agent one tool over the redaction log, then
hands the finished trajectory to a `Reflector` whose loop threshold,
success weight and error penalty are the library defaults spelled out:

{{ locus_source examples/notebook_54_reasoning_patterns.py::part1_reflexion }}

Part 2 reuses that `Reflector` on a hand-built `AgentState` whose
`tool_history` names `query_dlp` four times. In this release the loop
check runs only once `state.iteration` reaches the loop threshold and
reads the repeated tool sets from `reasoning_steps`, so a state carrying
only `tool_history` is assessed `on_track` with no loop pattern; a live
agent run fills both fields.

Part 4 carries the thesis: one agent fetches the DLP stats through
`query_dlp`, a second returns three claims as a typed `ClaimList`, and
`GroundingEvaluator` scores each claim against what the tool returned
rather than the model's paraphrase of it:

{{ locus_source examples/notebook_54_reasoning_patterns.py::part4_grounding }}

Part 6 asks for the timeline as a typed `EventList`, links each event to
the one before it and lets `build_causal_chain` classify the nodes from
root cause to symptom:

{{ locus_source examples/notebook_54_reasoning_patterns.py::part6_causal_chain }}

## Run it

```bash
python examples/notebook_54_reasoning_patterns.py
```

Parts 4 and 6 need a live model that can produce JSON. Model choice comes
from `examples/config.py`: OCI Generative AI when `~/.oci/config` has a
usable profile; otherwise set `LOCUS_MODEL_PROVIDER` to `openai` or
`anthropic` with `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the
environment. There is no offline run: under the bundled mock model,
`check_structured_output_capable()` prints a notice with example
provider settings and exits cleanly (status 0) before Part 1 runs. When
a live model returns no parsable `ClaimList` or `EventList`, Parts 4 and
6 raise `RuntimeError` with a pointer to a stronger model.

To force a provider and model:

```bash
export LOCUS_MODEL_PROVIDER=openai
export LOCUS_MODEL_ID=gpt-4o
python examples/notebook_54_reasoning_patterns.py
```

## See also

- [Reasoning](../concepts/reasoning.md) — Reflexion and grounding as
  `Agent(...)` flags, `build_causal_chain()` as a standalone builder, and
  the reflect-first, ground-last order.
- [Structured output](../concepts/structured-output.md) — what
  `Agent(output_schema=…)` puts on `result.parsed` and how a failed parse
  is repaired.
- [GSAR typed grounding](notebook_55_gsar_typed_grounding.md) — notebook
  55, the typed-evidence grounding layer with per-type evidence weights
  and a three-tier recovery; the [GSAR concept page](../concepts/gsar.md)
  defines it.

## Source

??? note "Full source — `examples/notebook_54_reasoning_patterns.py`"

    ````python
    --8<-- "examples/notebook_54_reasoning_patterns.py"
    ````
