# Evaluation

An agent that worked yesterday may not work today — the model
changed, a tool was renamed, the prompt got a one-line tweak. Tulip ships a small evaluation harness
so regressions become **failing tests**, not customer tickets.

```python
from tulip.evaluation import EvalCase, EvalRunner

cases = [
    EvalCase(
        name="weather_lookup",
        prompt="What's the weather in NYC?",
        expected_tools=["get_weather"],
        expected_output_contains=["temperature", "New York"],
        max_iterations=5,
    ),
]

report = EvalRunner(agent=agent).run(cases)
print(report.summary())
```

## When to reach for an eval suite

| Situation | Run evals? |
|---|---|
| You changed a tool's signature, default args, or system prompt | **yes — every commit that touches it** |
| You're swapping models (gpt-4o → gpt-5, llama-3.3 → llama-4) | **yes — same suite, two providers, diff the report** |
| You're debating "is the agent better than last week?" | **yes — nightly soak with `n=20` per case to see variance** |
| One-shot exploration, scratch agent | no — overhead's not worth it |
| Heavy LLM-as-judge needed (open-ended quality) | the harness covers structural checks; pair it with a custom judge tool for free-text grading |

## Getting started

### 1. Define cases

`EvalCase` is a Pydantic model — every field is optional except
`name` and `prompt`. The runner only checks fields you set.

```python
from tulip.evaluation import EvalCase

books_real = EvalCase(
    name="books_real_flight",
    prompt="Book TK-12 for customer C-42.",
    expected_tools=["book_flight"],
    expected_output_contains=["TK-12", "booked"],
    max_iterations=4,
)

rejects_unknown = EvalCase(
    name="rejects_unknown_flight",
    prompt="Book ZZ-999.",
    expected_output_contains=["not found"],
    expected_output_not_contains=["booked", "confirmed"],
)
```

### 2. Run them

```python
from tulip.evaluation import EvalRunner

runner = EvalRunner(agent=agent)
report = runner.run([books_real, rejects_unknown])

print(report.summary())
# Eval Report: 2/2 passed (avg score: 1.00)
# Total duration: 4321ms
#   [PASS] books_real_flight (score: 1.00, 1872ms)
#   [PASS] rejects_unknown_flight (score: 1.00, 2449ms)
```

`run()` returns an `EvalReport` — a Pydantic model with per-case
results, aggregate pass/fail counts, average score, and total
duration. JSON-serialisable, drop into CI artifacts.

### 3. Wire it into CI

```python
# tests/test_agent_evals.py
import pytest
from tulip.evaluation import EvalRunner

def test_agent_passes_eval_suite(agent):
    report = EvalRunner(agent=agent).run(load_cases())
    failures = [r for r in report.results if not r.passed]
    assert not failures, report.summary()
```

## Built-in checks

Every check runs only when the corresponding field is set on the
case. Each check contributes equally to the per-case score.

| Field | Passes when |
|---|---|
| `expected_tools` | All listed tools appear in the run's tool executions. |
| `expected_output_contains` | Every string is a case-insensitive substring of the final message. |
| `expected_output_not_contains` | None of the strings appear in the final message. |
| `max_iterations` | The run finished in ≤ N ReAct turns. |
| `max_duration_ms` | Wall-clock duration ≤ N milliseconds. |

A case **passes** when every check passed; the **score** is the
fraction of checks that passed (handy for partial-credit scoring
across a soak).

## Tags and filtering

```python
EvalCase(name="..." , prompt="..." , tags=["smoke", "happy-path"])
EvalCase(name="..." , prompt="..." , tags=["adversarial"])

# Run only smoke cases on every commit; full suite nightly.
smoke = [c for c in all_cases if "smoke" in c.tags]
runner.run(smoke)
```

`tags` is just a list — slice it however your CI matrix expects.

## LLM-as-judge for open-ended quality

The built-in checks are structural ("did the right tool fire?", "did
the answer mention 'temperature'?"). For free-text quality
("is this answer empathetic?", "is the explanation correct?"), wrap a
judge model as a tool and key on its verdict:

```python
from tulip.tools.decorator import tool

@tool
def judge(answer: str) -> dict:
    """LLM-graded quality verdict (0.0–1.0 + reasoning)."""
    return judge_model.run_sync(f"Grade this answer: {answer}").message

# Then in the case:
EvalCase(
    name="empathetic_response",
    prompt="My order is late and I'm upset.",
    expected_tools=["judge"],
    expected_output_contains=["sorry"],  # at minimum
)
```

A future SDK release may bundle a typed judge directly into
`EvalCase`; for today, this pattern is the path.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Case passes locally, fails in CI | Model output varies between runs. Pin the model id, lower `temperature`, run with `n=5` and look at variance. |
| `max_duration_ms` flakes | Cold-start network latency. Use a wall-clock budget at the suite level, not per-case, or bump the per-case budget by 2×. |
| `expected_tools` reports failure even though the tool ran | Case-sensitive name match — `book_flight` != `Book_Flight`. |
| Score is 0.5 every time | One of two checks is consistently failing. Read `result.checks` — it carries the full pass/fail map. |

## Source and notebook

- [`notebook_55_evaluation.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_55_evaluation.py) — runnable end-to-end suite.
- [`tulip.evaluation.framework`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/evaluation/framework.py) — `EvalCase`, `EvalRunner`, `EvalReport`.

## See also

- [Reasoning](reasoning.md) — `reflexion=True` and `grounding=True` reduce the kind of failures you'd otherwise catch only in evals.
- [Termination](termination.md) — `max_iterations` on `EvalCase` mirrors `MaxIterations` on the agent.
- [Hooks](hooks.md) — record per-eval traces with a `TelemetryHook` for offline review.
