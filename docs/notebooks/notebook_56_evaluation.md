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

- Part 1, a structural suite: `EvalCase` declares the substrings a reply
  must contain (`expected_output_contains`) or must not
  (`expected_output_not_contains`), with an optional `max_iterations`
  budget; `EvalRunner.run` grades every case; `EvalReport.summary()` names
  each failed check, and `by_tag` counts the cases per tag.
- Part 2, a `Rubric` with dimensions: the judge scores each dimension
  0-100 and code computes the composite, the letter and the pass; the
  verdict carries its provenance (judge model, rubric revision, channel).
- Part 3, receipts: `require_receipts=True` fails an answer that states a
  figure no tool returned, with no model call.
- Part 4, repeats: `repeats=3` reads a case three times; it passes only if
  every reading passed, and `consistency` flags readings that disagreed.
- Part 5, two reports of one suite: saved, loaded, diffed with
  `diff_reports`, and gated with `EvalGate` against the baseline.

The three `EvalCase` objects go to the `EvalRunner`, which calls WARDEN
once per case with the case's prompt, grades each reply against the checks
the case declares and aggregates the results into the `EvalReport` the
script prints; the rubric branch needs `arun()` and a judge, which Part 2
supplies:

{{ locus_diagram evaluation-flow }}

## Key lines

Part 1 is the suite as a regression test: WARDEN is built from the model
`examples/config.py` selects and a system prompt that classifies each
request as `ALLOW` or `BLOCK`, the three cases pin the two expected
decisions and the claims the agent must not invent, and `EvalRunner.run`
grades every case and returns the report:

{{ locus_source examples/notebook_56_evaluation.py::example_structural_suite }}

Part 2 hands the same request to a judge under a three-dimension rubric.
The judge returns one score per dimension; the composite, the letter and
the pass are computed from the rubric, and the verdict names the judge
model and the rubric revision it was measured under:

{{ locus_source examples/notebook_56_evaluation.py::example_rubric_and_judge }}

Part 3 runs two scripted triage agents against one case with
`require_receipts=True`. Both read the same metric; one states a memory
figure the tool never returned, and the check names it:

{{ locus_source examples/notebook_56_evaluation.py::example_receipts }}

Part 5 saves a baseline and a candidate report of the same suite, loads
them back, prints the per-case diff and applies the default gate, which
fails on the regression:

{{ locus_source examples/notebook_56_evaluation.py::example_diff_and_gate }}

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

Under the mock model every WARDEN reply in Part 1 is the same placeholder
sentence, so the two decision cases fail their `output_contains` check and
only `no_invented_regulation` passes; the report reads `1/3 passed`. Part 2
has no second model to judge with offline, so a scripted judge submits one
fixed verdict and the part shows how a verdict is composed and stamped.
Parts 3 to 5 use the `locus.testing` doubles by design and read the same
with or without credentials; the suite tests WARDEN itself only against a
live model.

## See also

- [Evaluation](../concepts/evaluation.md) — the built-in checks, tags,
  the rubric as data, receipts and claims, repeats and panels, and
  comparable reports.
- [Evaluate an agent in CI](../how-to/evaluate-in-ci.md) — the same flow
  from the command line with `locus eval`.
- [Evaluation API](../api/evaluation.md) — the `EvalCase`, `EvalRunner`,
  `EvalReport`, `Rubric`, `LLMJudge`, `EvalGate` reference.

## Source

??? note "Full source — `examples/notebook_56_evaluation.py`"

    ````python
    --8<-- "examples/notebook_56_evaluation.py"
    ````
