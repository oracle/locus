# Evaluation

Treat the agent like any other piece of code. Declare cases, run them,
read the report. Locus ships a small, dependency-free harness so you
don't need an external eval framework for the common cases.

- `EvalCase` declares prompt plus expected substrings (positive or negative).
- `EvalRunner` runs the agent against every case.
- `EvalReport` summarises pass/fail counts and an average score.

Run it (OCI Generative AI is the default — the notebook harness picks it when `~/.oci/config` exists; set `OCI_PROFILE` to choose a non-default profile):

    python examples/notebook_55_evaluation.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_55_evaluation.py

## Source

```python
--8<-- "examples/notebook_55_evaluation.py"
```
