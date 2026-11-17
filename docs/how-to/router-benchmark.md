# Benchmark the router

The router's claim is structural: the model fills a typed `GoalFrame`, the
registry picks the protocol, and nothing it says can name a shape outside
the registered set. Its accuracy is a measurement, not a claim, and this
page is how to take it. The benchmark asks one question of every prompt in
a labelled corpus, "which protocol would be selected?", on the evaluation
primitives: cases in a file, a runner, a report split per tag, a diff and a
gate. Nothing is built or run; selection stops at the frame.

## 1. The corpus

`benchmarks/router/cases.json` holds sixty `EvalCase` objects, read by
`load_router_corpus`, which checks their shape. Two kinds:

| Tag | Cases | Expectation |
|---|---|---|
| `labelled`, `shape:<id>`, `goal:<task>` | a prompt with a known right protocol | `expected_output_contains` is that id |
| `trap` | a labelled case whose verb points at an expensive shape while the work is one step | the low-cost shape |
| `adversarial`, `injection` | tells the model to leave the protocol list | the answer is not `unbounded` |
| `adversarial`, `ambiguous` | nothing in the catalogue fits | the answer is not `unbounded` |

Every target answers with one of three strings: a registered protocol id,
`unbounded` for an id outside the registry, or `refused` when nothing was
selected (the frame was rejected, no protocol could take it, the provider
failed). A refusal misses a label but is still bounded, which is the answer
an adversarial prompt is allowed to get.

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
reading answered. Sixty cases at three readings is 180 model calls per
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
| all | 60 | strict (mean) | ... | ... |
| labelled | 42 | ... | ... | ... |
| trap | 2 | ... | ... | ... |
| injection | 8 | ... | ... | ... |
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
