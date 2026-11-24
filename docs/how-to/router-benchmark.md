# Benchmark the router

The router's claim is structural: the model fills a typed `GoalFrame`, the
registry picks the protocol, and nothing it says can name a shape outside
the registered set. Its accuracy is a measurement, not a claim, and this
page is how to take it. The benchmark asks one question of every prompt in
a labelled corpus, "which protocol would be selected?", on the evaluation
primitives: cases in a file, a runner, a report split per tag, a diff and a
gate. Nothing is built or run; selection stops at the frame.

## 1. The corpus

`benchmarks/router/cases.json` holds 198 `EvalCase` objects, read by
`load_router_corpus`, which checks their shape: 180 labelled prompts,
twenty-six of them traps, and eighteen adversarial ones.

| Tag | Cases | Expectation |
|---|---|---|
| `labelled`, `shape:<id>`, `goal:<task>` | a prompt with a known right protocol | `expected_output_contains` is that id |
| `difficulty:<level>` | on every labelled case: `low`, `medium` or `high`, the difficulty its author intends | splits the report, grades nothing |
| `trap` | a labelled case whose verb points at an expensive shape while the work is one step | the low-cost shape |
| `adversarial`, `injection` | tells the model to leave the protocol list | the answer is not `unbounded` |
| `adversarial`, `ambiguous` | nothing in the catalogue fits | the answer is not `unbounded` |

Every target answers with one of three strings: a registered protocol id,
`unbounded` for an id outside the registry, or `refused` when nothing was
selected (the frame was rejected, no protocol could take it, the provider
failed). A refusal misses a label but is still bounded, which is the answer
an adversarial prompt is allowed to get.

A label is derivable, not a second opinion. Every labelled case states the
frame its author intends in `metadata` (`expected_goal`,
`expected_complexity`, `expected_risk` and, where the prompt names them,
`capabilities`), and the label is the protocol the compiler selects for
that frame; a unit test holds the file to it. The thirteen goals have
thirteen or fourteen prompts each, spread over the three difficulties. The
traps cover the nine goals whose canonical shape is not the low-cost one:
a prompt that says plan, build, modify, diagnose, monitor, compare,
coordinate, research or write code about work that is one step.

The intended difficulty is the author's reading of the prompt. It is the
hypothesis a difficulty measured from outcomes replaces, and the
`difficulty:` rows of the table are where the two can be compared.

The file is the one source of truth: the routing benchmark in
`tests/integration/test_router_benchmark.py` and the gate below read the
same cases, and the report's fingerprint changes when a prompt does, so a
diff across two versions of the corpus is refused rather than read as a
regression.

## 2. Three targets

`locus.router.benchmark` adapts selection to `EvalRunner`'s agent
contract, and the three targets select over the same corpus:

```python
from locus.models import get_model
from locus.router.benchmark import freeform_target, typed_target

model = get_model("{{ locus_example_model }}")
typed = typed_target(model)  # the router, default extractor
baseline = freeform_target(model)  # the model names the id directly
```

The typed target runs `Router.extract` for the frame and
`CognitiveCompiler.explain` for the pick, which resolves the frame exactly
as `compile` would: the low-complexity short-circuit, the three gates, the
ranker, the risk fallback. The free-form target asks the model for a
protocol id in a flat schema that accepts any string, so an invented id is
measurable.

A classifier plugs into the typed target through the router's front seam.
Anything with `async classify(text) -> GoalFrame` will do; a trained model,
a lookup over a closed set of requests, a cache in front of the extractor:

```python
from locus.router import Complexity, GoalFrame, Risk, TaskType
from locus.router.benchmark import typed_target


class RuleClassifier:
    async def classify(self, text: str) -> GoalFrame:
        goal = TaskType.COMPARE if " vs " in text else TaskType.ANSWER
        return GoalFrame(
            primary_goal=goal, domain="ops", complexity=Complexity.LOW, risk=Risk.LOW
        )


with_classifier = typed_target(model, classifier=RuleClassifier())
```

Everything after the frame is unchanged, so the comparison isolates the
classifier.

## 3. Run it

```bash
python scripts/router_benchmark.py --model {{ locus_example_model }} --repeats 3
```

The model string goes through `get_model`; an `oci:` model reads
`OCI_PROFILE`, `OCI_REGION` and `OCI_COMPARTMENT_ID` from the environment.
`--targets typed freeform` runs a subset, `--classifier module:attr` names
a `RequestClassifier` factory for the `typed_classifier` target (without
one, the default extractor goes through the seam, so that target measures
the same path as `typed` until a trained classifier replaces it),
`--verbose` prints every prompt with its expected protocol and what each
reading answered. 198 cases at three readings is 594 model calls per
target.

Each report goes to a `FileReportStore` under `benchmarks/router/reports/`
(`--reports` changes the directory), one suite per target, so
`locus eval trend --store benchmarks/router/reports --suite router-typed`
reads the history back and `locus eval diff`, `summary` and `gate` work on
the stored files.

## 4. Read the table and the gate

The script prints one table, a column per target and a row per tag, with
`all` first:

```text
| tag | n | typed | typed_classifier | freeform |
|---|---:|---|---|---|
| all | 198 | strict (mean) | ... | ... |
| labelled | 180 | ... | ... | ... |
| trap | 26 | ... | ... | ... |
| injection | 8 | ... | ... | ... |
| difficulty:low | 50 | ... | ... | ... |
```

Each cell is `strict (mean)`. With three readings a case passes only when
every reading did, so the strict rate is the one the gate uses; the mean is
the share of readings that were right, the accuracy figure a single run
would report. Then the gate, as `GateResult.to_markdown()`, for each typed
target against the free-form report of the same run (or the saved one
`--baseline` names):

```text
**Gate: FAILED**, 2 reason(s)

| Failed check |
|---|
| pass rate <candidate> below <baseline> (<passed>/<graded> graded cases passed) |
| regressed against the baseline: <case>, <case> |
```

`router_gate` fails for any of: a pass rate below the baseline's on the
whole corpus; a pass rate below the baseline's on the `trap` cases; an
`injection` case that was `unbounded`; a case left not graded; and, with
`per_case`, a case the baseline passed that the candidate fails (`EvalGate`
with `no_regressions`; the diff behind it refuses two reports of different
suites either way). Every reason is listed, not the first. The exit code is
the gate's.

Per-case parity is what a trained classifier is held to: the script applies
it to `typed_classifier` when `--classifier` names one, or to every typed
target with `--per-case`. The default extractor is not a classifier, and a
one-step prompt it frames LOW goes to `direct_response` whatever the label
says, so a run of the default router gates on the floors and prints the
cases the baseline passed that it fails (`regressions`), the list a
classifier has to clear.

A reading that never reached a verdict is not a routing decision: a target
retries a dropped connection, a timeout, a rate limit or a server-side
failure (`is_transient_error`, three attempts with a doubling wait) before
it counts as `refused`, and the audit line names the attempts.

## 5. The gate in a job

The same gate is a test, behind the opt-in the routing benchmark uses:

```bash
RUN_ROUTER_BENCHMARK=1 LOCUS_TEST_MODEL={{ locus_example_model }} \
  pytest tests/integration/test_router_classifier_gate.py -s
```

It runs the three targets with `EvalRunner(repeats=3)`, stores the reports
in `ROUTER_BENCHMARK_REPORTS` (else the test's temporary directory), gates
`typed_classifier` against the free-form report (`ROUTER_BENCHMARK_BASELINE`
names a saved one instead) and prints the table and the Markdown verdict.
`ROUTER_BENCHMARK_CLASSIFIER=module:attr` puts a classifier under test, and
only then is the per-case check asserted; `ROUTER_BENCHMARK_REPEATS` changes
the readings.

No baseline is committed. The free-form rate moves with every model
release, so a baseline is a report in the store, named by the run that
produced it. The pull-request pipeline has no model credential; the gate
belongs to a scheduled job that runs the suite, puts the reports in a store
and posts the Markdown.

## 6. Difficulty from outcomes

The `difficulty:` tag is the author's reading of a prompt.
`locus.router.difficulty` measures it instead. Every labelled prompt is
answered in one reply, with no tools, by models of increasing strength (the
tiers), and a judge grades each answer against one rubric: the reply, on
its own, completes what the request asks for. The level is read from which
tier passes:

| Measured | Meaning |
|---|---|
| `low` | the cheapest tier's single reply does the work |
| `medium` | only a stronger tier's single reply does |
| `high` | no tier's single reply does: the request needs tools, several steps, a check or an approval |
| `not_measured` | a tier was not graded for the case (the judge abstained, the answer errored) and no graded tier settled it |

```bash
python scripts/router_difficulty.py \
  --tiers oci:<small-model> oci:<large-model> --judge oci:<judge-model> --repeats 3
```

The tiers are given cheapest first, two or more. The judge should be none
of them: a model grading its own answer measures self-consistency. With
three readings a tier passes a case only when every reading passed, and a
tier whose model raised on any reading of a case decides nothing for it,
whatever its other readings said: the case is `not_measured` unless a
graded tier settled it, and the operator reruns. The
script stores one report per tier (`router-difficulty-<tier>` suites in the
report store), prints the intended level against the measured one with the
traps on a row of their own, lists the cases where the two differ, and
writes the measurement as JSON with `--out`.

```python
import asyncio

from locus.evaluation import EvalRunner, LLMJudge
from locus.models import get_model
from locus.router.benchmark import load_router_corpus
from locus.router.difficulty import (
    answer_cases,
    confusion_table,
    direct_agent,
    disagreements,
    measure_difficulty,
)

cases = answer_cases(load_router_corpus("benchmarks/router/cases.json"))
judge = LLMJudge(get_model("{{ locus_example_model }}"))


async def measure() -> None:
    reports = {}
    for tier in ("oci:<small-model>", "oci:<large-model>"):
        runner = EvalRunner(direct_agent(get_model(tier)), judge=judge, repeats=3)
        reports[tier] = await runner.arun(cases)
    measured = measure_difficulty(cases, reports)
    print(confusion_table(measured))
    for item in disagreements(measured):
        print(item.case, item.intended, "->", item.measured)


asyncio.run(measure())
```

Three things to keep in mind when reading a measurement:

- **It is per model family.** The same prompt can be `low` on one ladder of
  tiers and `medium` on another, so a measurement names its tiers and its
  judge, and is taken again when they change.
- **`high` means one reply is not enough, whatever the reason.** A prompt
  that asks for a metric to be read is `high` because a reply cannot read
  one, not because the reasoning is hard. What a request needs (tools,
  several agents, an approval) is a separate part of the frame.
- **A case can carry its own rubric.** `metadata["answer_rubric"]` on a
  case replaces the shared criterion for that prompt, for a request whose
  right answer is specific enough to state.

## From Python

```python
import asyncio

from locus.evaluation import EvalRunner
from locus.models import get_model
from locus.router.benchmark import (
    freeform_target,
    load_router_corpus,
    router_gate,
    tag_table,
    typed_target,
)

model = get_model("{{ locus_example_model }}")
cases = load_router_corpus("benchmarks/router/cases.json")
typed = asyncio.run(EvalRunner(typed_target(model), repeats=3).arun(cases))
freeform = asyncio.run(EvalRunner(freeform_target(model), repeats=3).arun(cases))
print(tag_table({"typed": typed, "freeform": freeform}, cases))
print(router_gate(typed, freeform).to_markdown())
```
