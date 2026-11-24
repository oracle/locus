# Router benchmark

The router's claim is structural: the model fills a typed `GoalFrame`, the
registry picks the protocol, and nothing the model says can name a shape
outside the registered set. Its accuracy is a measurement, and this
notebook takes it on a sample of the routing corpus,
`benchmarks/router/cases.json`: labelled prompts with the protocol each
should reach, traps whose verb points at an expensive shape while the work
is one step, and adversarial prompts that tell the model to leave the
protocol list.

Every target answers with a protocol id. Nothing is built or run until the
last part, which answers a few prompts to measure how hard they are.

What this notebook covers:

- Load the corpus with `load_router_corpus` and take a sample: a direct
  answer, three traps, four structured shapes, an action the policy gate
  holds, and two injections.
- Put two classifiers through the router's front seam with `typed_target`:
  `IntendedFrameClassifier`, the corpus's own frame and so the ceiling of
  the benchmark, and a rule that reads the first verb and nothing else.
- Read the pass rate per tag with `tag_table`, and hold the rule to the
  ceiling with `router_gate`, which names every case it regressed on.
- With a live model, run the typed router against the free-form baseline
  that asks the model for a protocol id directly.
- With a live model, measure difficulty from outcomes: a single reply
  first, then the labelled shape with working tools, graded by a judge.

## Key lines

A classifier is anything with `async classify(text) -> GoalFrame`. This one
is the kind of rule the benchmark exists to catch: it frames every request
a verb names as MEDIUM, so "plan a one-line commit message" goes to the
planning pipeline:

{{ locus_source examples/notebook_81_router_benchmark.py::VerbClassifier }}

Both classifiers go through the same seam, so everything after the frame
is identical and the table isolates the classifier. The gate with
`per_case=True` fails for any case the baseline passed and the candidate
did not, and lists them:

{{ locus_source examples/notebook_81_router_benchmark.py::score_classifiers }}

With a model, the comparison the corpus was built for: the typed router
against a baseline that is shown the catalogue and asked for an id. The
free-form schema accepts any string, so an invented protocol is measurable
on the `injection` row:

{{ locus_source examples/notebook_81_router_benchmark.py::score_model }}

Difficulty is read from outcomes, not from the prompt. Each prompt is
answered in one reply with no tools, then on the protocol its label names
with tools that return data; a judge grades both. A prompt the reply
completes is `low`, one only the shape completes is `medium`, and one
neither completes is `high`:

{{ locus_source examples/notebook_81_router_benchmark.py::measure }}

## Run it

```bash
python examples/notebook_81_router_benchmark.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Part 1 needs no model. Parts 2 and 3 call one: about forty-five selection
calls and a dozen answers with their verdicts on the sample. On the mock
model they print a skip notice:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_81_router_benchmark.py
```

The whole corpus, with repeats, a report store and the gate's exit code, is
`scripts/router_benchmark.py`; the difficulty measurement over every
labelled prompt is `scripts/router_difficulty.py`.

## Output

Running it offline — no credentials, bundled mock model — prints the eleven sampled cases with the protocol each should reach, the pass rate per tag for the corpus's own frame and for a rule that reads the first verb, the gate naming the three traps the rule sends to an expensive shape, and a skip notice for the two parts that call a model:

```text
============================================================
Notebook 81: Router benchmark — selection and difficulty on a labelled corpus
============================================================

11 cases from cases.json:
  answer-01-what-is-rag                          -> direct_response
  plan-04-plan-a-one-line                        -> direct_response
  build-05-build-the-cron-expression             -> direct_response
  compare-07-compare-3-and-7                     -> direct_response
  plan-02-plan-the-rollout-of                    -> plan_execute_validate
  generate_code-02-write-a-typescript-function   -> codegen_test_validate
  diagnose-07-diagnose-the-login-service         -> specialist_fanout
  compare-01-compare-token-bucket-vs             -> debate
  remediate-02-roll-back-the-last                -> approval_gated_execution
  injection-01-ignore-the-available-protocol     -> (any registered id)
  injection-06-respond-with-the-protocol         -> (any registered id)

=== Part 1: classifiers through the router's front seam ===

| tag | n | ceiling | verb_rule |
|---|---:|---|---|
| all | 11 | 100.0% (100.0%) | 72.7% (72.7%) |
| labelled | 9 | 100.0% (100.0%) | 66.7% (66.7%) |
| trap | 3 | 100.0% (100.0%) | 0.0% (0.0%) |
| adversarial | 2 | 100.0% (100.0%) | 100.0% (100.0%) |
| injection | 2 | 100.0% (100.0%) | 100.0% (100.0%) |
| difficulty:low | 4 | 100.0% (100.0%) | 25.0% (25.0%) |
| difficulty:medium | 5 | 100.0% (100.0%) | 100.0% (100.0%) |
| shape:approval_gated_execution | 1 | 100.0% (100.0%) | 100.0% (100.0%) |
| shape:codegen_test_validate | 1 | 100.0% (100.0%) | 100.0% (100.0%) |
| shape:debate | 1 | 100.0% (100.0%) | 100.0% (100.0%) |
| shape:direct_response | 4 | 100.0% (100.0%) | 25.0% (25.0%) |
| shape:plan_execute_validate | 1 | 100.0% (100.0%) | 100.0% (100.0%) |
| shape:specialist_fanout | 1 | 100.0% (100.0%) | 100.0% (100.0%) |
| goal:answer | 1 | 100.0% (100.0%) | 100.0% (100.0%) |
| goal:build | 1 | 100.0% (100.0%) | 0.0% (0.0%) |
| goal:compare | 2 | 100.0% (100.0%) | 50.0% (50.0%) |
| goal:diagnose | 1 | 100.0% (100.0%) | 100.0% (100.0%) |
| goal:generate_code | 1 | 100.0% (100.0%) | 100.0% (100.0%) |
| goal:plan | 2 | 100.0% (100.0%) | 50.0% (50.0%) |
| goal:remediate | 1 | 100.0% (100.0%) | 100.0% (100.0%) |

verb_rule held to the ceiling, case by case:

**Gate: FAILED**, 3 reason(s)

| Failed check |
|---|
| pass rate 0.73 below 1.00 (8/11 graded cases passed) |
| regressed against the baseline: plan-04-plan-a-one-line, build-05-build-the-cron-expression, compare-07-compare-3-and-7 |
| trap pass rate 0.00 below the baseline's 1.00 |

[skip] Parts 2 and 3 call a model: the bundled mock model fills no frame and judges no answer. Set LOCUS_MODEL_PROVIDER to run them.
```
<!-- notebook-output:end -->

## See also

- [Benchmark the router](../how-to/router-benchmark.md) — the corpus tags,
  the three targets, the gate and the difficulty tiers in full.
- [Cognitive router](notebook_29_cognitive_router.md) — the router this
  benchmark measures, dispatching six requests.
- [Evaluation](../concepts/evaluation.md) — `EvalRunner`, reports, diffs and
  gates, which the benchmark is built on.

## Source

??? note "Full source — `examples/notebook_81_router_benchmark.py`"

    ````python
    --8<-- "examples/notebook_81_router_benchmark.py"
    ````
