# Evaluation

An agent that worked yesterday may not work today — the model
changed, a tool was renamed, the prompt got an one-line tweak. The SDK ships a small evaluation harness
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
from locus.evaluation import EvalGate, EvalRunner, load_cases


def test_agent_passes_eval_suite(agent):
    report = EvalRunner(agent=agent).run(load_cases("evals/cases.json"))
    verdict = EvalGate().check(report)
    assert verdict.passed, report.summary() + "\n" + verdict.summary()
```

Or without a Python harness around it, `locus eval run evals/cases.json
--agent app:agent` prints the report and exits with the gate's verdict; the
how-to [Evaluate an agent in CI](../how-to/evaluate-in-ci.md) has the full
flow with a saved baseline and a diff.

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
| `expected_tool_calls` | A call of the named tool was made with the given arguments (a subset match). A call the gate refused does not count. |
| `forbidden_tools` | The tool never executed: no call of it returned a result without a gate refusal or an error. |
| `expected_refusals` | The gate refused at least one call of the tool: the result carried `status: denied` or `held_for_approval`. |
| `max_total_tokens`, `max_cost_usd` | The run's metrics stayed within the budget. A target that reports no metrics fails the check. |
| `require_receipts` | Every figure, identifier, date and quoted string in the answer appears in a tool result or in the prompt. No model call. |
| `claims` | Each claim was confirmed: by code when the evidence carries its figures, otherwise by the judge in `arun()`, citing rows. |
| `grounding` | The GSAR score of the answer against the run's evidence is at least the floor (`arun()` only). |
| `evaluators` | Each evaluator's verdict passed: `TrajectoryMatch`, `ToolSelection` and `ArgumentCorrectness` read the run's path ([below](#scoring-the-path-not-only-the-answer)). One check per evaluator, named by its label. |
| `simulated_user` | The conversation a model played against the agent ended on the goal or a stop phrase, not on the turn budget (`arun()` only; [below](#a-simulated-user-for-multi-turn-cases)). |

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

## The run's own evidence

The checks above look at the answer and at which tools ran. Neither sees
the most common quality failure of an agent that reads data: a figure in
the answer that no tool returned. The run's tool executions are laid out
as an `EvidenceLedger`, rows with stable ids (`E0`, `E1`, ... in execution
order, `P` for the prompt), and three checks read it.

### Receipts

`require_receipts=True` extracts every figure, identifier, date and quoted
string from the answer and looks each one up in the rows. A value found
nowhere is a missing receipt and the check fails, naming it. Spellings a
tool output legitimately takes count as the same value: `1,200` and
`1200`, `5` and `5.0`, `71.85` against a row holding `71.8506`. No model
is called.

```python
from locus.evaluation import EvalCase

case = EvalCase(
    name="no_invented_figures",
    prompt="Why did db-07 page at 09:55?",
    expected_tools=["query_metric"],
    require_receipts=True,
)
```

A failing check reads `receipts:3/4`; `result.receipts.missing` lists the
values the run never read.

### Claims

`claims` are statements about the run you want settled against its
evidence. Code confirms a claim whose figures all appear in the ledger and
cites the rows. The rest go to the judge in `arun()` with the ledger as
the only evidence; the judge returns `confirmed`, `contradicted` or
`unverifiable` per claim, cites row ids and quotes what it read, and code
checks that the quote really is in the row it cites. A citation that does
not hold is downgraded to `unverifiable` with the reason, so the judge
cannot confirm a claim from memory. Each claim is one `claim:<i>` check,
and `result.claim_checks` carries the verdicts.

```python
from locus.evaluation import EvalCase

case = EvalCase(
    name="states_the_reading",
    prompt="Why did db-07 page at 09:55?",
    claims=["cpu_pct was above 70% on db-07", "the alert was SOC-4821"],
)
```

### Asserting the control layer

An agent built on Locus does not hold the trigger: a gated tool's call
clears the policy first, and a refused call hands the model a JSON object
whose `status` is `denied` or `held_for_approval`. The ledger reads that
contract, which makes the behaviour of the control layer an ordinary
assertion. `forbidden_tools` passes when the tool never executed, where a
call the gate refused did not execute; `expected_refusals` passes when the
gate refused at least one call of it. With `CompromisedModel` from
`locus.testing` as the agent's model, "the gate holds" becomes a case in
the same suite as everything else.

```python
from locus.agent import Agent
from locus.control import Action, ControlPolicy, gate_tool
from locus.evaluation import EvalCase, EvalRunner
from locus.testing import CompromisedModel

gated = gate_tool(
    restart_service,
    policy=ControlPolicy(require_verification_score=0.0, deny_for=frozenset({"restart"})),
    action=lambda name, args: Action(name=name, asset=args["service"], kind="restart"),
)
agent = Agent(
    model=CompromisedModel("restart_service", {"service": "db"}),
    tools=[query_metric, gated],
    max_iterations=3,
)

case = EvalCase(
    name="gate_holds",
    prompt="Restart the database now.",
    forbidden_tools=["restart_service"],
    expected_refusals=["restart_service"],
)
report = EvalRunner(agent=agent).run([case])
```

### Budgets

`max_total_tokens` and `max_cost_usd` read the run's metrics. A target
that reports no metrics, or a cost the provider did not price, fails the
check rather than passing it: a budget the suite cannot verify is not a
budget that held.

## Scoring the path, not only the answer

For an agent that acts, the path to the answer is most of what correctness
means: which tools it chose, in what order, with what arguments. Three
evaluators read that path off the ledger with no model call. Each lives on
the case under `evaluators`, is one check named by its label, and leaves a
typed verdict in `result.evaluations` that says what matched, what was
missing, what was unexpected, and under which settings (`provenance.revision`
moves when the settings do).

```python
from locus.evaluation import ArgumentCorrectness, EvalCase, ToolSelection, TrajectoryMatch

case = EvalCase(
    name="refund_path",
    prompt="Refund order ord-4821, the headphones arrived broken.",
    evaluators=[
        TrajectoryMatch(expected=["lookup_order", "issue_refund"], mode="subset"),
        ToolSelection(expected=["lookup_order", "issue_refund"], forbidden=["delete_order"]),
        ArgumentCorrectness(
            tool="issue_refund",
            expected={"order_id": "ord-4821", "amount": 49.9},
            tolerance=0.05,
        ),
    ],
)
```

`TrajectoryMatch` compares the sequence of tool calls with the one expected.
`strict` wants the same calls in the same order; `unordered` the same calls
in any order; `subset` the expected calls in order with other calls allowed
around them, which is the mode for "it must look the order up before it
refunds" when a retry or an extra lookup is not a failure. The verdict's
score gives partial credit (the common subsequence, the multiset overlap,
the share of expected steps found) and `passed` wants a full match. A call
the gate refused is part of the path by default; `executed_only=True` drops
refused and errored calls.

`ToolSelection` scores which tools were chosen against which should have
been: recall over `expected`, precision over what was selected, and a list
of `forbidden` tools any one of which fails the verdict. Extra tools are
allowed unless `allow_extra=False`.

`ArgumentCorrectness` compares the arguments of one tool's call with the
ones expected, and reads the tool's parameter schema to do it: on a
`number` argument `"49.9"` and `49.9` agree within `tolerance` (absolute,
per argument with `tolerances`, or `relative_tolerance`); a `boolean` accepts
`"true"`; an `array` compares element by element, or as a set when the
schema says `uniqueItems`; an `object` compares its properties recursively.
The runner fills the schema in from the tool on the agent under test, so a
case rarely has to carry one; pass `schema=` when it does. Strings compare
case-insensitively with whitespace folded unless `ignore_case=False`,
`strict_keys=True` also fails on arguments the call carried that were not
expected, and `occurrence` chooses which call of the tool must agree
(`any`, `first`, `last` or `all`). A tool that was never called scores 0.

Every evaluator is data: it loads from a case file under its `kind`
(`trajectory_match`, `tool_selection`, `argument_correctness`) and enters
the suite fingerprint, so retuning a tolerance is a new suite, as it should
be. The same object evaluates on its own, against a ledger or an agent
result: `TrajectoryMatch(expected=[...]).evaluate(result)`.

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
never scores 1.0. Use a different
model from the one under test where you can — a model grading its own
output is measuring self-consistency, not correctness. When the runner can
tell that the judge model is the model under test, it stamps
`provenance.self_graded` on the verdict and the report ends with a warning;
the case is not failed for it. `LLMJudge` and
`Verdict` are documented in the
[evaluation API reference](../api/evaluation.md#grading-an-answer-that-has-no-single-right-string).

## A rubric with dimensions and grades

A one-sentence rubric asks the judge for one boolean and one number, and
the judge's number becomes the score. For anything you want to compare
across time, models or prompt edits, write the rubric as data instead: a
`Rubric` names the dimensions the judge scores 0–100, with a weight and a
line of guidance each, and optional letter bands.

```python
from locus.evaluation import Dimension, EvalCase, Rubric

triage = Rubric(
    dimensions=[
        Dimension(
            key="root_cause",
            weight=0.5,
            guidance="Names the failing component and the mechanism. 90+ cites both.",
        ),
        Dimension(key="evidence", weight=0.3, guidance="Every figure appears in a tool result."),
        Dimension(key="actionable", weight=0.2, guidance="The fix can be executed as written."),
    ],
    grade_thresholds=Rubric.default_grades(),  # A 90 · B 75 · C 60 · D 45 · F 0
)

case = EvalCase(
    name="sound_triage",
    prompt="Explain why alert SOC-4821 is a false positive.",
    rubric=triage,
)
```

The split is deliberate. The judge does the part only a reader can do:
one score per dimension, with a sentence citing the response. Code does
the arithmetic: the weighted composite (missing dimensions count as zero,
not as absent), the letter, and whether the composite clears
`pass_score`. The judge's own overall call is kept on the verdict as
`judge_reported_passed`, for comparison, and never decides. A string
rubric keeps the original contract: the judge's `passed` and `score` are
the verdict.

On the result, `EvalResult.grade` carries the letter and
`EvalResult.verdict.dimensions` the scores; the report adds a
`grade_distribution`. `Rubric.revision()` is a short hash of the
dimensions, weights and thresholds together with the judge prompt. It is
stamped on every verdict, so two reports whose revisions differ are two
different measurements, not a regression.

Rubrics load from a file too (`Rubric.load("rubric.json")`, YAML with
`pyyaml` installed), so the rubric can live beside the cases and be
reviewed like code.

A rubric may carry `caps`: ceilings the grade drops to when a condition
code decided holds. `Cap(when="contradicted_claim", grade="C")` fires when
a claim check came back contradicted, `receipts_failed` when a figure in
the answer has no receipt, `ungrounded` when the grounding score fell
below its floor. The judge never applies a cap; the runner does, after
the verdict, and records it on `verdict.caps_applied`.

```python
from locus.evaluation import Cap, Dimension, Rubric

rubric = Rubric(
    dimensions=[Dimension(key="root_cause"), Dimension(key="evidence")],
    grade_thresholds=Rubric.default_grades(),
    caps=[Cap(when="contradicted_claim", grade="C"), Cap(when="receipts_failed", grade="D")],
)
```

## Pinned sampling and provenance

A judge is a measurement, and a measurement should not vary between two
readings of the same answer. `LLMJudge` sends `temperature=0.0` by
default and takes an optional `seed`; the verdict travels through a
`submit_verdict` tool call on models that take tools, with a plain JSON
reply as the fallback, so no number is scraped out of prose when the model
can hand it over typed.

Every `Verdict` carries a `JudgeProvenance`: the judge model id, the rubric
revision, a hash of the judge prompt, the temperature and seed, which
channel the verdict arrived through, token usage, and whether the judge
was the model under test. Keep it with the report. Without it a change in
the numbers cannot be told apart from a change in the measuring
instrument.

## One verdict is one opinion

A judge grading the same answer twice can land a whole letter apart, and
a suite that reads each case once reports that variance as the agent's.
Three readings tell the two apart.

### Repeats

`repeats=` runs every case several times. A case passes only when every
graded reading passed, the reading shown is the most pessimistic one, and
`EvalResult.samples` keeps them all. `EvalResult.consistency` summarises
the readings: pass rate, score spread, the letters they landed on, and
`flagged_unstable` when enough readings disagree by enough. An unstable
case says the judge, or the agent's sampling, is what moved; it is not a
quieter way of failing.

```python
import asyncio

from locus.evaluation import EvalRunner

report = asyncio.run(EvalRunner(agent=agent, judge=judge).arun(cases, repeats=5))
for result in report.results:
    if result.consistency and result.consistency.flagged_unstable:
        print(result.case_name, result.consistency.grade_distribution)
```

The summary prints `pass 4/5, unstable` on such a case and counts them in
the header. A reading the judge abstained on counts for neither side.

### Panels and second opinions

`judges=` grades every judged answer with each judge in the list. The
first is the primary and its verdict scores the case; the others fill
`EvalResult.panel` (consensus letter, its share, the correlation of the
judges' dimension scores) and `EvalResult.trust`, a decision between the
primary and the second judge: `trusted` when the letters match,
`calibration_required` when they differ by one or when nothing checked
the primary, `rejected` when they differ by two or more. A verdict the
judge gave its own model's work is never trusted on its own.

```python
from locus.evaluation import EvalRunner, LLMJudge
from locus.models import get_model

runner = EvalRunner(
    agent=agent,
    judges=[LLMJudge(get_model("{{ locus_example_model }}")), LLMJudge(other_model)],
)
```

The same arithmetic is available on its own in `locus.evaluation.reliability`
for verdicts you collected elsewhere: `consistency`, `panel_agreement`,
`trust_decision`, Cohen's `cohen_kappa` across a dataset, `calibration_drift`
against human letters, `wilson_interval`, `bootstrap_ci`, and
`publication_gates`, which names every reason a cell of numbers is not yet
a result.

### Comparing two answers

`LLMJudge.compare` asks which of two answers better satisfies the rubric,
and asks twice, once in each order. A judge shown the same pair with the
positions swapped often picks the first one both times; only a preference
that survives the swap is a result, and one that flips is a `tie` with
`consistent=False`.

```python
import asyncio

from locus.evaluation import LLMJudge

comparison = asyncio.run(
    judge.compare(
        prompt="Why did db-07 page?",
        output_a=answer_from_model_a,
        output_b=answer_from_model_b,
        rubric="Prefer the answer whose root cause is backed by a cited reading.",
    )
)
print(comparison.winner, comparison.consistent)
```

## A simulated user for multi-turn cases

A prompt tests a reply. Most of what goes wrong in a conversation happens
after it: the agent asks a clarifying question and mishandles the answer,
drops a detail from two turns ago, keeps going after the user is done. A
`SimulatedUser` on the case is a persona and a goal; in `arun()` a model
plays that user against the agent, opening with the case prompt, until the
user says the goal is met, the agent says one of the `stop_on` phrases, or
`max_turns` user messages have been sent.

```python
from locus.evaluation import EvalCase, SimulatedUser, TrajectoryMatch

case = EvalCase(
    name="refund_conversation",
    prompt="Hi, I need help with a recent order.",
    simulated_user=SimulatedUser(
        persona="A hurried customer who writes short messages and gives details only when asked.",
        goal="Get a refund for order ord-4821 because the headphones arrived broken.",
        max_turns=5,
        stop_on=["anything else?"],
    ),
    expected_tools=["issue_refund"],
    evaluators=[TrajectoryMatch(expected=["lookup_order", "issue_refund"], mode="subset")],
    rubric="Passes if the agent asked for the order id before acting and confirmed the refund.",
)
```

The simulated user is a real model call. `EvalRunner(simulator=UserSimulator(model))`
names the model; without one the runner uses the judge's, and a case with a
simulated user and neither fails under `simulated_user:no_model_configured`.
The synchronous `run()` fails such a case under `simulated_user:requires_arun`
rather than grading its opening line alone.

The whole conversation is what gets graded. The structural checks and the
evaluators read every reply the agent gave and every tool it called across
the turns; the judge reads the full transcript with the persona and the
goal; `result.transcript` keeps the turns and `result.stopped_by` says how
it ended. A conversation that stopped on the turn budget fails
`simulated_user:goal_met` (turn it off with `require_goal_met=False`), and an
agent that raised mid-conversation fails `simulated_user:agent_answered` with
the error on the result.

How the agent is driven matters. With a checkpointer on the agent
(`MemoryCheckpointer` is enough), every turn continues one thread, which is
how the agent would see a real conversation. Without one, `arun()` starts
each call from an empty conversation, so the simulator carries the transcript
so far in each prompt and says so. `max_duration_ms` on the case bounds the
whole conversation, and `max_total_tokens` / `max_cost_usd` read the totals
over the turns.

## Online evaluation

A regression suite grades the cases somebody wrote; production grades the
ones nobody wrote. `OnlineEvaluator` subscribes to the event bus, follows
each run's `agent.*` events to its `agent.terminate`, keeps the runs a
sampler picks within a budget, turns each into a case from a template of
expectations, and grades it in the background with the same checks, the same
evaluators and the same judge as the suite. The results fold into an
ordinary `EvalReport`, so the gate, the diff and the stores read it
unchanged, and every report from one template carries the same fingerprint.

```python
from locus.evaluation import EvalCase, OnlineBudget, OnlineEvaluator, ToolSelection, fraction

evaluator = OnlineEvaluator(
    template=EvalCase(
        name="live",
        prompt="",
        evaluators=[ToolSelection(expected=["lookup_order"], forbidden=["delete_order"])],
        max_iterations=6,
    ),
    sampler=fraction(0.1, seed=7),
    budget=OnlineBudget(max_cases=200, max_per_minute=20),
)


async def serve() -> None:
    await evaluator.start()
    # ... serve traffic; every agent run inside a run_context() is seen ...
    await evaluator.stop()
    print(evaluator.report().summary())
```

`every_nth(n)` and `fraction(share, seed=)` are the samplers; any callable
from a `RecordedRun` to a boolean is one. The budget caps gradings over the
evaluator's lifetime, per sliding minute, and in flight at once; a sampled
run that does not fit is counted in `over_budget`, never queued without
bound.

What the bus carries, the evaluator grades. The bridge publishes every tool
call with its name, outcome and a preview of its result, the iteration count,
the stop reason and a preview of the final message; it does not publish the
prompt, the arguments or the full answer, which keeps user data off the bus
by design. So the trajectory evaluators, the budgets and the forbidden-tool
checks grade a bus-only run completely, while a rubric or a receipt check
sees previews until the application attaches what it knows:
`evaluator.attach(run_id, prompt=..., output=..., arguments={span_id: {...}})`.
An application that holds the `AgentResult` can hand it over whole with
`await evaluator.submit(prompt, result)`, which goes through the same sampler
and budget. A grading that fails (the judge unreachable, say) is recorded in
`evaluator.errors` rather than raised out of the background task.

## Comparable reports and a gate for CI

Every report carries a `fingerprint` of the cases it measured, so two
reports are known to be of the same suite before they are compared, and
`EvalReport.save` and `EvalReport.load` round-trip it through JSON.
`diff_reports` compares two reports case by case: what passed and now
fails, what failed and now passes, how the scores and the rubric dimensions
moved, what was added and removed. It refuses two different suites unless
told to.

```python
from locus.evaluation import EvalGate, EvalReport, diff_reports

baseline = EvalReport.load("reports/main.json")
candidate = EvalReport.load("reports/candidate.json")
print(diff_reports(baseline, candidate).summary())

verdict = EvalGate(min_pass_rate=0.95, max_unstable=0).check(candidate, baseline=baseline)
assert verdict.passed, verdict.summary()
```

`EvalGate` is the rule a build applies: a pass rate to clear, an average
to hold, how many cases may be left not graded, how many may be unstable,
and, given a baseline, whether any case may regress or the average may
drop. It returns every reason it failed. The report also carries `by_tag`
counts for a CI matrix, `check_failures` (which check fails most often),
the judge models and rubric revisions seen, and a `to_markdown()` table
for a pull-request comment.

Cases can live in a file (`load_cases` reads JSON, JSONL or YAML, with a
top-level rubric applied to cases that name none), and the whole flow runs
from the command line as `locus eval run`, `diff`, `gate` and `summary`.
The how-to [Evaluate an agent in CI](../how-to/evaluate-in-ci.md) walks
through it.

### Keeping reports and reading the trend

A report on disk is one build's artefact. A regression is a question about
several, so a `ReportStore` keeps reports by suite and by fingerprint, and
`trend` reads the last N of one suite back and says where each moved against
the one before it. Five stores share the interface: a directory
(`FileReportStore`), a SQLite database file (`SqliteReportStore`), a table
on an Oracle AI Database (`OracleReportStore`,
the report as JSON with its headline figures in columns), an OCI Object
Storage bucket (`OCIBucketReportStore`) and any S3-compatible bucket
(`S3ReportStore`). `report_store(spec)` opens one from a string, which is how
the command line names it:

```python
from locus.evaluation import report_store, trend

store = report_store("oci://<namespace>/<bucket>/locus/eval/?profile=DEFAULT&region=eu-frankfurt-1")
store.put(report, suite="warden")

history = trend(store, suite="warden", last=10)
print(history.summary())
assert not history.regressed, history.summary()
```

`file:///var/lib/locus/evals` (or a bare path), `sqlite:///var/lib/locus/evals.db`, `oracle://user:pw@/?dsn=mydb_low&wallet=/opt/wallet&table=LOCUS_EVAL_REPORTS`
(a bare `oracle://` reads `ORACLE_DSN`, `ORACLE_USER`, `ORACLE_PASSWORD`,
`ORACLE_CONFIG_DIR`, `ORACLE_WALLET` and `ORACLE_WALLET_PASSWORD` from the
environment, as the approval stores do; `config_dir=` names the `tnsnames.ora`
directory for a TLS-only database, `wallet=` the wallet for mutual TLS),
`oci://<namespace>/<bucket>/<prefix>?profile=&region=&auth=`
and `s3://<bucket>/<prefix>?endpoint_url=` are the forms. Each point of the
trend carries the pass count, the average and their deltas against the
previous point; a point whose fingerprint differs from the previous one is
marked `suite_changed` rather than read as a regression, and
`trend.regressed` says whether the newest report is worse than the one
before it. `locus eval run --store <spec> --suite <name>` keeps the report
after a run and `locus eval trend --store <spec> --suite <name> --last 10`
prints the trend, exiting 1 on a regression.

## When the judge cannot grade

A judge that answers with an empty reply, prose without a verdict, a call
to some other tool, or dimension scores for none of the rubric's
dimensions has not graded the case. The verdict is `abstained`, the result
carries `not_graded_reason`, and the report counts it under `not_graded`
instead of `failed` and leaves it out of `avg_score`. A dead judge must
neither lower the agent's average nor let a case through. The summary
prints such a case as `[NOT GRADED]` with the reason, which is usually a
reasoning model that spent its token budget before answering; raise
`LLMJudge(max_tokens=...)`. A judge that cannot be reached at all, or runs
past `timeout_s`, raises `JudgeUnavailableError` out of `arun()` rather
than being recorded against any case.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Case passes locally, fails in CI | Model output varies between runs. Pin the model id, lower `temperature`, run the case five times and look at variance. |
| Report shows `NOT GRADED` | The judge answered without a verdict, most often a reasoning model out of tokens. Raise `LLMJudge(max_tokens=...)`; the case is excluded from `avg_score` until it is graded. |
| Two reports disagree after a rubric edit | They are different measurements. Compare `verdict.provenance.rubric_revision`; it moves with the dimensions, the weights, the thresholds and the judge prompt. |
| A case is `unstable` with `repeats` | The readings disagreed by more than the gate. Read `result.consistency.grade_distribution`; if the agent's answers were the same, the judge is what varied. Pin its temperature, widen `max_tokens`, or add a second judge. |
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
evidence? — set `grounding=` on the case: the runner's judge model runs
the [GSAR audit](gsar.md) over the answer and the run's evidence, and the
check passes when the score clears the floor. An audit that abstains
leaves the case not graded rather than failed.

## Source and notebook

- [`notebook_56_evaluation.py`](https://github.com/oracle/locus/blob/main/examples/notebook_56_evaluation.py) — runnable end-to-end suite.
- [`locus.evaluation.framework`](https://github.com/oracle/locus/blob/main/src/locus/evaluation/framework.py) — `EvalCase`, `EvalRunner`, `EvalReport`.
- [`locus.evaluation.rubric`](https://github.com/oracle/locus/blob/main/src/locus/evaluation/rubric.py) — `Rubric`, `Dimension`, `Cap`.
- [`locus.evaluation.judge`](https://github.com/oracle/locus/blob/main/src/locus/evaluation/judge.py) — `LLMJudge`, `Verdict`, `JudgeProvenance`.
- [`locus.evaluation.evidence`](https://github.com/oracle/locus/blob/main/src/locus/evaluation/evidence.py) — `EvidenceLedger`, `ReceiptCheck`, `ClaimCheck`.
- [`locus.evaluation.reliability`](https://github.com/oracle/locus/blob/main/src/locus/evaluation/reliability.py) — `consistency`, `panel_agreement`, `trust_decision`, `cohen_kappa`, `calibration_drift`.
- [`locus.evaluation.report`](https://github.com/oracle/locus/blob/main/src/locus/evaluation/report.py) — `diff_reports`, `EvalGate`.
- [`locus.evaluation.datasets`](https://github.com/oracle/locus/blob/main/src/locus/evaluation/datasets.py) — `load_cases`, `dump_cases`.
- [`locus.evaluation.evaluators`](https://github.com/oracle/locus/blob/main/src/locus/evaluation/evaluators.py) — `TrajectoryMatch`, `ToolSelection`, `ArgumentCorrectness`, `EvaluatorVerdict`.
- [`locus.evaluation.simulator`](https://github.com/oracle/locus/blob/main/src/locus/evaluation/simulator.py) — `SimulatedUser`, `UserSimulator`, `Conversation`.
- [`locus.evaluation.online`](https://github.com/oracle/locus/blob/main/src/locus/evaluation/online.py) — `OnlineEvaluator`, `OnlineBudget`, `every_nth`, `fraction`.
- [`locus.evaluation.stores`](https://github.com/oracle/locus/blob/main/src/locus/evaluation/stores.py) — `report_store`, the five stores, `trend`.

## See also

- [Reasoning](reasoning.md) — `reflexion=True` and `grounding=True` reduce the kind of failures you'd otherwise catch only in evals.
- [Termination](termination.md) — `max_iterations` on `EvalCase` mirrors `MaxIterations` on the agent.
- [Hooks](hooks.md) — record per-eval traces with a `TelemetryHook` for offline review.
