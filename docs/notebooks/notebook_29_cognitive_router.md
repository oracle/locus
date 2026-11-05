# Cognitive router

The `locus.router` package compiles a natural-language operations
request onto existing primitives. The large language model (LLM)
never chooses the topology: it fills a typed `GoalFrame`, the router
selects a protocol deterministically from a `ProtocolRegistry`, and the
`CognitiveCompiler` emits a real `Agent`, `SequentialPipeline`,
`ParallelPipeline` or `LoopAgent` from the matching builder.

`PolicyGate` is the control that keeps an autonomous router from
over-reaching. It compares the frame's `risk` band with the selected
protocol's `risk_max` and with its own two thresholds, `max_risk` and
`require_approval_above` (`Risk.HIGH` and `Risk.MEDIUM` by default), and
that comparison, not any free-form judgement by the model, decides
whether an action runs.

A LOW-risk "summarize this runbook" request routes to a direct answer and
runs straight through. A HIGH-risk "drain a production node and restart
the payments deployment" request compiles behind an approval callback;
the notebook keeps the compiler's default callback, which denies, so the
action is held and never executes.

What this notebook covers:

- Define a small infrastructure and DevOps capability set as annotated
  tools.
- Register all eight built-in protocols.
- Load the SKILL.md packages under `examples/skills/` and tag each one by
  domain, so the compiler attaches the matching skills to every `Agent`
  it emits.
- Stand up a `Router` with an `Agent(output_schema=GoalFrame)` extractor
  and a `CognitiveCompiler`.
- Dispatch six requests: five reach five different protocols
  (`direct_response`, `plan_execute_validate`, `specialist_fanout`,
  `debate`, `codegen_test_validate`) and print the protocol that fired
  and the runtime shape it compiled to; the sixth is HIGH risk, so
  `PolicyGate` requires approval and the compiled runnable is held
  instead of executed.

## Key lines

The router is assembled in one place: a `CapabilityIndex` over the three
annotated tools, the eight built-in protocols, a `PolicyGate` with its
default thresholds (`require_approval_above=Risk.MEDIUM`, so a HIGH-risk
frame needs approval; `max_risk=Risk.HIGH`, the top of the `Risk` scale,
so the gate's own ceiling denies nothing), a `CognitiveCompiler` that
keeps the default approval callback, and an `Agent` extractor whose
`output_schema` is `GoalFrame`, so the model fills a typed frame and never
chooses the topology. With no `protocol_picker` configured, the compiler
ranks the candidates that pass the filter by rule: the protocol that
lists the goal in `primary_for` wins when its cost is within one step of
the frame's complexity, then the closest cost-to-complexity fit, then
lower cost, then fewer `handles`. A frame whose risk is above every
eligible protocol's cap is not an error: it compiles onto
`approval_gated_execution` and is held for approval:

{{ locus_source examples/notebook_29_cognitive_router.py::build_router }}

The compiler's skill catalogue comes from the SKILL.md packages under
`examples/skills/`. Each skill is registered under the `domain` its
frontmatter `metadata` block declares (`incident-triage` is
`observability`; `ioc-enrichment` and `siem-query` are `security`), or as
global when it declares none (`code-review`). At compile time every
builder looks up `SkillIndex.for_domain(frame.domain)` and attaches the
result, global skills included, to each `Agent` it emits as a
`SkillsPlugin`:

{{ locus_source examples/notebook_29_cognitive_router.py::_load_skills }}

Each request goes through `Router.dispatch`, which runs the extractor,
compiles the frame and executes the runnable. The mock model returns
canned text rather than JSON, so `Router.extract` emits
`router.frame.failed` and raises `FrameExtractionError`, and the
hand-built fallback frame is compiled and executed directly instead;
protocol selection, the policy gate and the compiled runtime still run
end to end. Because this path calls `router.compiler.compile` without a
run id, the `on_frame` callback and the selection, verdict, compile and
execution events (`router.protocol.selected`, `router.policy.verdict`,
`router.runnable.*`) appear only with a live provider. For the sixth
request, `approval_gated_execution` is the only protocol that accepts a
`REMEDIATE` goal at HIGH risk (`plan_execute_validate` caps at MEDIUM);
the gate allows it with `require_approval=True`, the compiler wraps the
runnable in an approval check, the default callback denies, and `execute`
raises `PolicyDeniedError`, which `main` prints as the `[verdict]` line:

{{ locus_source examples/notebook_29_cognitive_router.py::_dispatch_with_fallback }}

## Run it

```bash
python examples/notebook_29_cognitive_router.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

With a live provider the extractor fills each `GoalFrame` from the prompt
and the `[frame]` line shows what it chose. To force the offline path,
where the extractor returns no valid frame and each request falls back to
its hand-built `GoalFrame`:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_29_cognitive_router.py
```

## Output

Running it offline — no credentials, bundled mock model — prints a fallback notice for each request (the mock extractor produces no `GoalFrame`, so the hand-built frame is used), the protocol the router selected and the runtime shape it compiled to for the five requests that run, the `PolicyGate` verdict that holds the sixth, HIGH-risk node drain for approval, and a closing protocol histogram:

```text
============================================================
Notebook 29: Cognitive router — risk-gated routing for infra/devops requests
============================================================

Model Provider: mock
Using mock model (no API calls)

[router] 8 protocols · 3 capabilities · 4 skills

=== Prompt 1 ===
  Summarize what the runbook says about connection pool exhaustion.
  [extractor unavailable: Extractor did not produce a GoalFrame. parse_error='JSON parse error: Expecting value: line 1 column 1 (char 0)', message='This is a mock response for testing purposes.'; using fallback frame]
  [protocol] direct_response
  [shape]    Agent (single call)
  [result]
  This is a mock response for testing purposes.

=== Prompt 2 ===
  Write a three-step plan to roll out the new autoscaling policy to every prod service, including a validation step.
  [extractor unavailable: Extractor did not produce a GoalFrame. parse_error='JSON parse error: Expecting value: line 1 column 1 (char 0)', message='This is a mock response for testing purposes.'; using fallback frame]
  [protocol] plan_execute_validate
  [shape]    SequentialPipeline of 3 Agents (planner → executor → validator)
  [result]
  This is a mock response for testing purposes.

=== Prompt 3 ===
  Diagnose the spike in 5xx errors — pull recent alerts and the error_rate metric, correlate them.
  [extractor unavailable: Extractor did not produce a GoalFrame. parse_error='JSON parse error: Expecting value: line 1 column 1 (char 0)', message='Alert assessment: likely true positive. Recommend containment.'; using fallback frame]
  [protocol] specialist_fanout
  [shape]    ParallelPipeline of N tool-bound Agents
  [result]
  Alert assessment: likely true positive. Recommend containment.

  ---

  Alert assessment: likely true positive. Recommend containment.

=== Prompt 4 ===
  Compare two rollback strategies for a bad deploy: full blue-green swap vs canary roll-back. Which minimizes downtime?
  [extractor unavailable: Extractor did not produce a GoalFrame. parse_error='JSON parse error: Expecting value: line 1 column 1 (char 0)', message='Go on checkout: the image is signed and current, and all 6 replicas are healthy.'; using fallback frame]
  [protocol] debate
  [shape]    ParallelPipeline of 2 debaters + judge Agent
  [result]
  Go on checkout: the image is signed and current, and all 6 replicas are healthy.

=== Prompt 5 ===
  Generate a Python function that returns the SHA-256 of a file for build-artifact integrity checks, with a doctest that verifies it on empty input.
  [extractor unavailable: Extractor did not produce a GoalFrame. parse_error='JSON parse error: Expecting value: line 1 column 1 (char 0)', message='This is a mock response for testing purposes.'; using fallback frame]
  [protocol] codegen_test_validate
  [shape]    LoopAgent (PASS/FAIL stop)
  [result]
  This is a mock response for testing purposes.

=== Prompt 6 ===
  Drain and cordon node prod-node-17 and restart the payments deployment in the prod cluster.
  [extractor unavailable: Extractor did not produce a GoalFrame. parse_error='JSON parse error: Expecting value: line 1 column 1 (char 0)', message='Go on checkout: the image is signed and current, and all 6 replicas are healthy.'; using fallback frame]
  [policy]   held for approval — not executed
  [verdict]  approval denied for protocol='approval_gated_execution': approval required: frame.risk='high' above approval threshold 'medium'

=== Protocol histogram ===
  (held for approval)          × 1
  codegen_test_validate        × 1
  debate                       × 1
  direct_response              × 1
  plan_execute_validate        × 1
  specialist_fanout            × 1

============================================================
Next: Notebook 30 — A2A — cross-org payment-intel sharing between spec-compliant peers
============================================================
```
<!-- notebook-output:end -->

## See also

- [Emergent routing](notebook_28_emergent_routing.md) — notebook 28: the
  rule-based router beside an emergent one whose compiler carries an
  `LLMProtocolPicker` for the last-mile choice when more than one protocol
  survives the filter.
- [Skills](../concepts/skills.md) — the SKILL.md format and the
  progressive disclosure the compiler's `SkillsPlugin` relies on.

## Source

??? note "Full source — `examples/notebook_29_cognitive_router.py`"

    ````python
    --8<-- "examples/notebook_29_cognitive_router.py"
    ````
