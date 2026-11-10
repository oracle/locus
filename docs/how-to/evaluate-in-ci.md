# Evaluate an agent in CI

A regression suite earns its keep when it runs on every change and fails
the build when the agent gets worse. This page puts the evaluation
primitives together for that: cases in a file, a judge that is not the
model under test, a saved report, a diff against the last good one, and a
gate whose verdict is the exit code.

## 1. Put the cases in a file

Cases are data. A JSON, JSONL or YAML file beside the agent holds them,
and a top-level rubric applies to every case that names none:

```json
{
  "rubric": {
    "dimensions": [
      {"key": "decision", "weight": 0.5, "guidance": "States ALLOW or BLOCK once, without hedging."},
      {"key": "reasoning", "weight": 0.3, "guidance": "Names the fact that decides it."},
      {"key": "restraint", "weight": 0.2, "guidance": "Asserts no regulation the request does not support."}
    ],
    "grade_thresholds": {"A": 90, "B": 75, "C": 60, "D": 45, "F": 0}
  },
  "cases": [
    {
      "name": "known_disallowed_pii_export",
      "prompt": "Request: export the raw customer table to an unvetted vendor. Allow or block?",
      "expected_output_contains": ["block"],
      "tags": ["decision"]
    },
    {
      "name": "no_invented_regulation",
      "prompt": "Request: a report counting active accounts, no personal data. Allow or block?",
      "expected_output_not_contains": ["hipaa", "breach"],
      "require_receipts": false,
      "tags": ["restraint"]
    }
  ]
}
```

`load_cases` reads the same file from Python. Editing a prompt, adding a
case or retuning a weight moves the suite's fingerprint, which every
report carries, so a diff across two different suites is refused rather
than read as a regression.

## 2. Name the agent and the judge

`locus eval run` loads the agent from `MODULE:ATTR` or `FILE.py:ATTR`: an
`Agent`, or a function that returns one. The judge is named the same way
and may be an `LLMJudge`, a model to wrap in one, or a factory for either.
Use a different model from the one under test; when the runner can tell
they are the same, the report says so.

```python
# evals/agents.py
from locus.agent import Agent
from locus.evaluation import LLMJudge
from locus.models import get_model


def warden() -> Agent:
    return Agent(model=get_model("{{ locus_example_model }}"), system_prompt="...")


def judge() -> LLMJudge:
    return LLMJudge(get_model("{{ locus_example_model }}"), temperature=0.0, seed=7)
```

## 3. Run, save, gate

```bash
locus eval run evals/cases.json \
  --agent evals/agents.py:warden \
  --judge evals/agents.py:judge \
  --repeats 3 \
  --report reports/candidate.json \
  --baseline reports/main.json
```

The command prints the report, writes it, and applies the gate: every
graded case passes, nothing is left not graded, and no case that passed
on the baseline fails now. Its exit code is the gate's verdict. Every
threshold is a flag, and every failure is reported, not just the first:

| Flag | Default | Fails when |
|---|---|---|
| `--min-pass-rate` | `1.0` | fewer graded cases passed, as a share |
| `--min-avg-score` | off | the average score is lower |
| `--max-not-graded` | `0` | more cases were left not graded by the judge |
| `--max-unstable` | off | more cases were flagged unstable under `--repeats` |
| `--allow-regressions` | off | with a baseline: a case that passed now fails (unless this flag is set) |
| `--max-avg-score-drop` | off | with a baseline: the average dropped by more |

`--tags smoke` runs only the cases carrying one of the given tags, for a
quick lane on every commit and the full suite nightly.

## 4. Read a diff, post a table

```bash
locus eval diff reports/main.json reports/candidate.json
locus eval diff reports/main.json reports/candidate.json --markdown
locus eval summary reports/candidate.json --markdown
```

`diff` exits 1 when any case regressed and names it, with the checks that
newly fail and how each rubric dimension moved. `--markdown` renders a
table for a pull-request comment. `locus eval gate` applies the gate to a
saved report, for a pipeline that runs the suite in one job and decides in
another.

## From Python

Everything the command does is a call:

```python
import asyncio

from locus.evaluation import EvalGate, EvalReport, EvalRunner, diff_reports, load_cases

cases = load_cases("evals/cases.json")
report = asyncio.run(EvalRunner(agent=agent, judge=judge, repeats=3).arun(cases))
report.save("reports/candidate.json")

baseline = EvalReport.load("reports/main.json")
print(diff_reports(baseline, report).summary())
verdict = EvalGate(min_pass_rate=0.95, max_unstable=0).check(report, baseline=baseline)
assert verdict.passed, verdict.summary()
```

## What a green run proves

That the cases in the file, graded by the named judge under the rubric
revision on the report, passed on this commit, and that none of them
passed on the baseline and fails now. It does not prove the suite covers
the behaviour that matters; that is the cases' job, and the fingerprint
makes it visible when they change.

## See also

- [Evaluation](../concepts/evaluation.md) for the checks, the rubric, the
  judge and the reliability readings.
- [Testing agents](../concepts/testing.md) for the model doubles a suite
  can run against with no credentials.
