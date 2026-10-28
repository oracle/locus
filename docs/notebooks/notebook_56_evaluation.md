# Evaluation

Treat the privacy-review agent like any other piece of code. Declare
cases, run them, read the report. Locus ships a small, dependency-free
harness so you don't need an external eval framework for the common cases.

- `EvalCase` declares prompt plus expected substrings (positive or negative).
- `EvalRunner` runs the agent against every case.
- `EvalReport` summarises pass/fail counts and an average score.

Here WARDEN — a tier-1 data-access reviewer — classifies each
data-handling request as `ALLOW` or `BLOCK`. The suite pins a
known-disallowed PII export, a known-allowed aggregate report, and a
negative case that guards against the agent inventing a regulation
citation no facts support (OWASP LLM09, Misinformation). Run it on every
prompt or model change before it reaches production.

Run it (OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider):

    python examples/notebook_56_evaluation.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_56_evaluation.py

## Source

````python
--8<-- "examples/notebook_56_evaluation.py"
````
