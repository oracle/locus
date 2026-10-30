# Evaluation

An agent that worked yesterday may not work today — the model
changed, a tool was renamed, the prompt got a one-line tweak. Locus ships a small evaluation harness
so regressions become **failing tests**, not silent failures in production.

```python
from locus.evaluation import EvalCase, EvalRunner

cases = [
    EvalCase(
        name="order_lookup",
        prompt="Look up order ord-4821.",
        expected_tools=["lookup_order"],
        expected_output_contains=["status", "ord-4821"],
        max_iterations=5,
    ),
]

report = EvalRunner(agent=agent).run(cases)
print(report.summary())
```

The runner takes that list, runs the agent once per case within the
case's budget, grades what came back against the checks the case
declares, and folds one `EvalResult` per case into an `EvalReport`:

{{ locus_diagram evaluation-flow }}

## When to reach for an eval suite

| Situation | Run evals? |
|---|---|
| You changed a tool's signature, default args, or system prompt | **yes — every commit that touches it** |
| You're swapping models (gpt-4o → gpt-5, llama-3.3 → llama-4) | **yes — same suite, two providers, diff the report** |
| You're debating "is the agent better than last week?" | **yes — nightly soak that runs each case 20 times to see variance** |
| One-shot exploration, scratch agent | no — overhead's not worth it |
| Heavy LLM-as-judge needed (open-ended quality) | **yes — set a `rubric` on the case and give the runner an `LLMJudge` ([below](#llm-as-judge-for-open-ended-quality))** |

## Getting started

### 1. Define cases

`EvalCase` is a Pydantic model — every field is optional except
`name` and `prompt`. The runner only checks fields you set.

```python
from locus.evaluation import EvalCase

refunds_eligible = EvalCase(
    name="refunds_eligible_order",
    prompt="Refund order ord-4821 — return approved.",
    expected_tools=["issue_refund"],
    expected_output_contains=["ord-4821", "refunded"],
    max_iterations=4,
)

rejects_unknown = EvalCase(
    name="rejects_unknown_order",
    prompt="Refund order zz-999.",
    expected_output_contains=["not found"],
    expected_output_not_contains=["refunded"],
)
```

### 2. Run them

```python
from locus.evaluation import EvalRunner

runner = EvalRunner(agent=agent)
report = runner.run([refunds_eligible, rejects_unknown])

print(report.summary())
# Eval Report: 2/2 passed (avg score: 1.00)
# Total duration: 4321ms
#   [PASS] refunds_eligible_order (score: 1.00, 1872ms)
#   [PASS] rejects_unknown_order (score: 1.00, 2449ms)
```

`run()` returns an `EvalReport` — a Pydantic model with per-case
results, aggregate pass/fail counts, average score, and total
duration. JSON-serialisable, drop into CI artifacts.

### 3. Wire it into CI

```python
# tests/test_agent_evals.py
import pytest
from locus.evaluation import EvalRunner

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
| `expected_tool_sequence` | The listed tools were called in that order; extra calls around them are allowed unless `exact_tool_sequence=True`. |
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
the answer mention 'reputation'?"). For free-text quality
("is the triage rationale sound?", "is the finding correctly grounded?"),
write a `rubric` on the case and give the runner an `LLMJudge`:

```python
import asyncio

from locus.evaluation import EvalCase, EvalRunner, LLMJudge
from locus.models import get_model

case = EvalCase(
    name="sound_triage_rationale",
    prompt="Explain why alert SOC-4821 is a false positive.",
    expected_output_contains=["benign"],  # structural checks still run
    rubric="Passes if it names the evidence that makes the alert benign.",
)

judge = LLMJudge(model=get_model("{{ locus_example_model }}"))
runner = EvalRunner(agent=agent, judge=judge)
report = asyncio.run(runner.arun([case]))
```

Grading requires `await runner.arun(cases)` — the judge is a model call,
and `run()` is synchronous. `run()` cannot call a judge, so a rubric
case passed to it fails under a `rubric:requires_arun` check rather than
reporting green with the rubric unevaluated. On `arun()`, a rubric case
with no judge configured fails loudly too, under a
`rubric:no_judge_configured` check.

On a graded case, `EvalResult.score` is the judge's 0–1 score, capped by
the fraction of checks passed, so a case with a failed structural check
never scores 1.0. A judge reply that cannot be parsed yields
`Verdict(unparseable=True)` and is recorded as a failing
`judge_unparseable:...` check rather than a silent pass. Use a different
model from the one under test where you can — a model grading its own
output is measuring self-consistency, not correctness. `LLMJudge` and
`Verdict` are documented in the
[evaluation API reference](../api/evaluation.md#grading-an-answer-that-has-no-single-right-string).

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Case passes locally, fails in CI | Model output varies between runs. Pin the model id, lower `temperature`, run the case five times and look at variance. |
| `max_duration_ms` flakes | Cold-start network latency. Use a wall-clock budget at the suite level, not per-case, or bump the per-case budget by 2×. |
| `expected_tools` reports failure even though the tool ran | Case-sensitive name match — `issue_refund` != `Issue_Refund`. |
| Score is 0.5 every time | One of two checks is consistently failing. Read `result.checks` — it carries the full pass/fail map. |

## Evals are audit evidence

The runner drives the same agent through the same loop as production, so
an eval run is made of the same typed events — and any admission
decisions land on the same hash-chained `AuditTrail` (each entry commits
to the one before it, so editing any record breaks verification) the
gate always writes. A passing eval is audit evidence, not a screenshot. For
grounded-claim scoring — did each claim survive contact with its
evidence? — pair the structural checks here with the
[GSAR evaluation layer](gsar.md).

## Source and notebook

- [`notebook_56_evaluation.py`](https://github.com/oracle/locus/blob/main/examples/notebook_56_evaluation.py) — runnable end-to-end suite.
- [`locus.evaluation.framework`](https://github.com/oracle/locus/blob/main/src/locus/evaluation/framework.py) — `EvalCase`, `EvalRunner`, `EvalReport`.

## See also

- [Reasoning](reasoning.md) — `reflexion=True` and `grounding=True` reduce the kind of failures you'd otherwise catch only in evals.
- [Termination](termination.md) — `max_iterations` on `EvalCase` mirrors `MaxIterations` on the agent.
- [Hooks](hooks.md) — record per-eval traces with a `TelemetryHook` for offline review.
