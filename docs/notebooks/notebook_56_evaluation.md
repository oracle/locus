# Evaluation

Treat WARDEN, the tier-1 data-access reviewer, like any other piece of
code: declare cases, run them, read the report. A prompt or model change
that quietly starts approving exports of raw customer data, or asserting
findings no facts support, fails the suite before it reaches production.
The SDK ships a small evaluation harness as part of the SDK, so you do not
need an external evaluation framework for the common cases.

WARDEN classifies each data-handling request as `ALLOW` or `BLOCK`. The
suite pins a known-disallowed export of personally identifiable information
(PII), a known-allowed aggregate report, and a negative case that guards
against the agent inventing a regulation citation or a breach finding no
facts support
([OWASP LLM09, Misinformation](../concepts/threat-scenarios.md#owasp-llm-top-10-2025)).

What this notebook covers:

- `EvalCase`: a request prompt plus the substrings the reply must contain
  (`expected_output_contains`) or must not (`expected_output_not_contains`),
  with an optional `max_iterations` budget.
- `EvalRunner.run`: calls the agent once per case and grades every check
  the case declares.
- `EvalReport`: pass and fail counts, an average score, and a `summary()`
  that names each failed check.

The three `EvalCase` objects go to the `EvalRunner`, which calls WARDEN
once per case with the case's prompt, grades each reply against the checks
the case declares (here `expected_output_contains`,
`expected_output_not_contains` and the `max_iterations` budget) and
aggregates the results into the `EvalReport` the script prints; the dashed
rubric branch needs `arun()` and a judge, so this synchronous `run()` does
not take it:

{{ locus_diagram evaluation-flow }}

## Key lines

One function carries the whole suite: WARDEN is built from the model
`examples/config.py` selects and a system prompt that classifies each
request as `ALLOW` or `BLOCK`, the three cases pin the two expected
decisions and the claims the agent must not invent, and `EvalRunner.run`
grades every case and returns the report:

{{ locus_source examples/notebook_56_evaluation.py::example_evaluation }}

## Run it

```bash
python examples/notebook_56_evaluation.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

To run offline regardless of the local OCI configuration:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_56_evaluation.py
```

Under the mock model every reply is the same placeholder sentence, so the
two decision cases fail their `output_contains` check and only
`no_invented_regulation` passes; the report reads `1/3 passed`. That run
shows the report format, including how a failed case lists the check it
missed; the suite tests WARDEN itself only against a live model.

## See also

- [Evaluation](../concepts/evaluation.md) — the built-in checks, tags,
  wiring a suite into continuous integration (CI), and grading open-ended
  answers with a `rubric` and an `LLMJudge`.
- [Evaluation API](../api/evaluation.md) — the `EvalCase`, `EvalRunner`,
  `EvalReport` and `LLMJudge` reference.

## Source

??? note "Full source — `examples/notebook_56_evaluation.py`"

    ````python
    --8<-- "examples/notebook_56_evaluation.py"
    ````
