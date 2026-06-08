# Evaluation

Treat the agent like any other piece of code. Declare cases, run them,
read the report. Tulip ships a small, dependency-free harness so you
don't need an external eval framework for the common cases.

- `EvalCase` declares prompt plus expected substrings (positive or negative).
- `EvalRunner` runs the agent against every case.
- `EvalReport` summarises pass/fail counts and an average score.

Run it (defaults to the bundled mock model; set `TULIP_MODEL_PROVIDER` to `openai` / `anthropic` for a live model):

    python examples/notebook_55_evaluation.py

Offline:

    TULIP_MODEL_PROVIDER=mock python examples/notebook_55_evaluation.py

## Source

```python
--8<-- "examples/notebook_55_evaluation.py"
```
