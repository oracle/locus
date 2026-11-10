# Evaluation

Test harness for agents — define `EvalCase`s, run them with `EvalRunner`, and
collect results into an `EvalReport`.

For the concepts, start with [Evaluation](../concepts/evaluation.md) and
[Testing agents](../concepts/testing.md).

::: locus.evaluation.framework.EvalCase
::: locus.evaluation.framework.EvalResult
::: locus.evaluation.framework.EvalReport
::: locus.evaluation.framework.EvalRunner

## Grading an answer that has no single right string

Structural checks verify shape, not quality. `expected_output_contains`
in particular fails in both directions: it passes on an answer that happens to
contain the word, and fails on a correct answer phrased differently. Where the
right answer is not one exact string, something has to read it.

`LLMJudge` grades against a rubric and returns a typed `Verdict`. Three
things it deliberately does not do:

- **It does not retry until it passes.** A judge you re-roll is not a judge.
- **It does not score zero when it cannot be reached.** An unusable judge
  raises, because a "failing" eval that actually means *the judge was down* is
  worse than no eval.
- **It does not turn a reply it cannot read into a number.** That verdict is
  `abstained`, and the runner reports the case as not graded.

Use a different model from the one under test where you can — a model grading
its own output is measuring self-consistency, not correctness. The runner
stamps `provenance.self_graded` when it can tell.

::: locus.evaluation.judge.LLMJudge
::: locus.evaluation.judge.Verdict
::: locus.evaluation.judge.JudgeProvenance
::: locus.evaluation.judge.Comparison
::: locus.evaluation.judge.model_id_of

## The rubric as data

A string rubric is one pass criterion the judge answers. A `Rubric` names
dimensions the judge scores 0–100; the composite, the letter and the pass are
then computed here, from the weights and thresholds, and the judge's own
overall call survives only as `judge_reported_passed`. `revision()` identifies
the rubric and prompt a verdict was measured under.

::: locus.evaluation.rubric.Rubric
::: locus.evaluation.rubric.Dimension
::: locus.evaluation.rubric.Cap

## The run's own evidence

The run's tool executions, as rows with stable ids a claim can cite. Receipts
and the claims whose figures the evidence carries are settled by code; the
judge settles the rest and its citations are verified against the rows. The
gate's refusal contract (`status: denied` or `held_for_approval` on a tool
result) is what `forbidden_tools` and `expected_refusals` read.

::: locus.evaluation.evidence.EvidenceLedger
::: locus.evaluation.evidence.EvidenceRow
::: locus.evaluation.evidence.ReceiptCheck
::: locus.evaluation.evidence.ClaimCheck
::: locus.evaluation.evidence.ToolCallExpectation
::: locus.evaluation.evidence.settle_receipts
::: locus.evaluation.evidence.settle_claims

## How far to trust a verdict

Pure functions over verdicts and grade pairs. `EvalRunner` calls
`consistency` with `repeats=` and `panel_agreement` / `trust_decision` with
`judges=`; the rest read verdicts you collected elsewhere.

::: locus.evaluation.reliability.consistency
::: locus.evaluation.reliability.Consistency
::: locus.evaluation.reliability.panel_agreement
::: locus.evaluation.reliability.PanelAgreement
::: locus.evaluation.reliability.trust_decision
::: locus.evaluation.reliability.TrustDecision
::: locus.evaluation.reliability.TrustGates
::: locus.evaluation.reliability.cohen_kappa
::: locus.evaluation.reliability.Kappa
::: locus.evaluation.reliability.calibration_drift
::: locus.evaluation.reliability.CalibrationDrift
::: locus.evaluation.reliability.wilson_interval
::: locus.evaluation.reliability.bootstrap_ci
::: locus.evaluation.reliability.publication_gates
::: locus.evaluation.reliability.ReliabilityGates
::: locus.evaluation.reliability.HoldReason

## Asserting the order, not just the membership

`expected_tools` asks whether a tool appeared *somewhere* in the run. It cannot
tell "looked the order up, then refunded it" from "refunded it, then looked it
up" — and for an agent that acts, that ordering is most of what correctness
means.

::: locus.evaluation.judge.check_trajectory

## Running a suite against a graph

`EvalRunner` expects an agent: it calls `run_sync(prompt)` and reads `.message`,
`.iterations` and `.tool_executions`. A `StateGraph` takes a dict and returns a
`GraphResult` with none of those. `as_eval_target` adapts one to the other.

It has to make two translations a graph cannot make for itself:

- **A prompt is not a state.** `input_key` says which field of the graph's
  initial state the case prompt becomes. There is no universally right default,
  so `"prompt"` is a starting guess to correct.
- **Nodes are the graph's tools.** `expected_tools` and
  `expected_tool_sequence` match on **node ids**, which makes "a change sent
  this down the wrong branch" — the regression a graph suite exists to catch —
  an ordinary assertion.

Addressing the answer takes two arguments rather than one, because
`final_outputs` is keyed by node id while a graph's result accumulates in
`final_state`: `output_key` reads the state, `output_node` reads one node's own
output, and passing both raises rather than silently preferring one.

::: locus.evaluation.graph.as_eval_target
::: locus.evaluation.graph.GraphEvalTarget
