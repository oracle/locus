# Guardrails

A security operations centre (SOC) triage agent reads text it cannot trust:
ticket bodies, scan results and threat-intelligence lookups. The dangerous
case is indirect prompt injection, an instruction smuggled into the output of
a tool the agent itself called, which then attempts to direct the agent to
exfiltrate case data or invoke a destructive tool. This example attaches
`GuardrailsHook` and `ContentFilterHook` to that agent and shows what each
refuses: dangerous input before the model sees it, and dangerous tool calls
before they run.

Part 4 is the showcase. Four sample results from a `search_intel` tool, three of them carrying a smuggled payload, go through `GuardrailsHook.on_after_tool_call`, which checks the result's length and scans it for personally identifiable information (PII), and then through `on_before_invocation`, where the SQL-injection, path-traversal and command-injection signatures fire.

The example turns the first detection into a `Partition` of two grounded claims, and `ground_finding` scores it with Grounding-Stratified Adaptive Replanning ([GSAR](../concepts/gsar.md)): the tool-output row that carried the instruction is the evidence, so an `Evidence` ships with its score, its evidence references and its taxonomy tags. An ungrounded claim would lower the score, and below the proceed threshold `ground_finding` returns an `Abstention` instead.

Four parts attach a hook to a live agent run through `Agent(hooks=[...])`;
Parts 3 to 7 call the hook methods directly on sample ticket text, tool output
and tool-call events, so each rule can be read on its own; Parts 1 and 10
declare configurations and ask the model to comment on them, and Part 11 asks
the model for a list of best practices. Every part makes at least one model
round-trip through `_llm_call`, which prints the elapsed time and the prompt
and completion token counts.

What this notebook covers:

- `GuardrailsHook` with a typed `GuardrailConfig`: tool block list, prompt and
  tool-result length caps, a default action and per-rule overrides.
- PII detection on ticket text in `on_before_invocation`: under `BLOCK` the
  hook raises before the model is called; under a `REDACT` or `WARN` override
  the text is admitted and the violation is recorded in the agent state.
- Injection-pattern blocking (SQL injection, path traversal, command injection)
  on text arriving in tickets and tool output.
- Indirect prompt injection detected in tool output and surfaced as a grounded
  `Evidence` through `ground_finding`, tagged OWASP LLM01, OWASP LLM02 and
  MITRE ATLAS AML.T0051.
- Tool denylist versus allowlist in `on_before_tool_call`.
- Secret-leakage filtering with `ContentFilterHook`, and stacked hooks through
  `HookRegistry`.

The guardrails here are hooks:

- `on_before_invocation` scans the prompt.
- `on_after_tool_call` scans a tool's result.
- `on_before_tool_call` refuses a tool on the block list, such as `exec` or `shell`, before it runs.

The control layer drawn below is what a gated tool applies to the calls a hook lets through: the gate turns each one into an Action, `admit` decides it and records the decision on the audit trail, and the side effect runs, waits for approval or is denied.

This example stops at the hooks and the `ground_finding` call, which applies the GSAR score to one security finding rather than to the agent's answer as the drawing's grounding step does; the gate, the audit trail, the approval store and the checkpointer in the drawing play no part here:

{{ locus_diagram control-layer-flow }}

## Key lines

Part 10 gathers the rules the earlier parts exercise one at a time into a
production preset: nine blocked tools, tighter prompt and tool-result caps,
`BLOCK` as the default action, and overrides keyed by the rule name a violation
carries (`pii_email`, not `email`; `GuardrailConfig` rejects a name no
violation can carry). `REDACT` rewrites the PII in a tool's string arguments
before the tool runs; on the prompt it admits the text and records the
violation:

{{ locus_source examples/notebook_69_guardrails_security.py::main.production_config }}

The development preset keeps only `exec` and `eval` on the block list, raises
both caps and makes `WARN` the default, so a pattern or PII match in the prompt
is recorded in the agent state and the run continues, while a tool on the block
list is still refused:

{{ locus_source examples/notebook_69_guardrails_security.py::main.development_config }}

## Run it

```bash
python examples/notebook_69_guardrails_security.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## See also

- [Hooks](../concepts/hooks.md) — the lifecycle phases a hook can subscribe
  to, and the built-in `GuardrailsHook` and `ContentFilterHook`.
- [GSAR — typed grounding](../concepts/gsar.md) — the score and decision
  behind `ground_finding`.
- [Trace hygiene](notebook_70_guardrails_advanced.md) — `OutputFilterHook`,
  which redacts PII from the model's reply and applies `TopicPolicy` and
  `ContentPolicy` to it.

## Source

??? note "Full source — `examples/notebook_69_guardrails_security.py`"

    ````python
    --8<-- "examples/notebook_69_guardrails_security.py"
    ````
