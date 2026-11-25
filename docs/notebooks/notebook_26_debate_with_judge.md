# Debate with a judge

An INCIDENT advocate and a NOISE advocate take turns arguing over a
cloud-monitoring alert. After a fixed number of rounds a judge reads the
full transcript and emits a typed `Verdict` — call, confidence, key
points, reasoning — that downstream systems (incident tooling, audit
logs, on-call paging) can consume directly.

`JUDGE_PROMPT` tells the judge to call `incident` or `noise` only when
one side clearly outargued the other and to return `inconclusive`
otherwise, which keeps a two-agent debate from manufacturing false
certainty about an ambiguous alert.

What this notebook covers:

- `Turn(side, round, text)` accumulated into a `list[Turn]` in graph
  state — the transcript.
- `output_schema=Verdict` on the judge `Agent`, so `result.parsed` is a
  populated Pydantic object, not a JSON string.
- The judge node raises rather than fabricating a verdict if the
  configured model cannot honor the schema.
- `check_structured_output_capable()` short-circuits the notebook with
  setup guidance when running under the mock model, which returns plain
  text and cannot produce JSON.

## Prerequisites

- [Structured output](notebook_07_structured_output.md).
- [Basic graph](notebook_12_basic_graph.md).

With `N_ROUNDS = 2`, the graph alternates the two advocates and routes to
the judge once `route_after_noise` has seen two complete rounds:

```text
incident r0 → noise r0 → incident r1 → noise r1 → judge → END
```

## Key lines

`Verdict` is the schema the judge must populate. `output_schema` turns
its `Field` descriptions and the confidence bounds into the JSON schema
the model reads — appended to the system prompt, and sent as a
`response_format` to providers that advertise native structured output
— and into the validation its reply must pass. `call` is a plain `str`
whose description lists the three values, so the choice among them is
instructed by `JUDGE_PROMPT` rather than enforced by validation:

{{ locus_source examples/notebook_26_debate_with_judge.py::Verdict }}

`judge_turn` builds the judge with `output_schema=Verdict`, hands it the
whole transcript, and raises when `final.parsed` is `None`, which
happens only after the agent loop's own schema-repair re-prompts
(`output_schema_retries`, two by default) have failed. Transient
provider failures are retried inside the agent loop by
`AgentConfig.model_retry`, so the node carries no retry loop of its own:

{{ locus_source examples/notebook_26_debate_with_judge.py::judge_turn }}

`build_debate_graph` wires the two advocates into a loop and routes to
the judge once `route_after_noise` has seen `N_ROUNDS` complete rounds.
The edge back from `noise` to `incident` is a conditional edge; the
graph's cycle check inspects only edges added with `add_edge`, so the
loop is accepted without `GraphConfig(allow_cycles=True)`:

{{ locus_source examples/notebook_26_debate_with_judge.py::build_debate_graph }}

## Run it

```bash
python examples/notebook_26_debate_with_judge.py
```

The judge needs a live model that can produce JSON. Model choice comes from `examples/config.py`: OCI Generative AI when `~/.oci/config` has a usable profile; otherwise set `LOCUS_MODEL_PROVIDER` to `openai` or `anthropic` with `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the environment, or to any other registry provider together with `LOCUS_MODEL_ID` and its own credentials.

There is no offline run: under the bundled mock model, `check_structured_output_capable()` prints a notice with example provider settings and exits cleanly (status 0) before the graph is built.

When a live model returns no parsable `Verdict`, `judge_turn` raises `RuntimeError` naming a model with constrained decoding to use instead; the graph records the judge node as failed, and the driver reports it and exits with an error.

To force a provider and model:

```bash
export LOCUS_MODEL_PROVIDER=oci
export LOCUS_MODEL_ID=openai.gpt-4o
python examples/notebook_26_debate_with_judge.py
```

## Source

??? note "Full source — `examples/notebook_26_debate_with_judge.py`"

    ````python
    --8<-- "examples/notebook_26_debate_with_judge.py"
    ````
