# Trace hygiene

An agent's replies are written somewhere durable: the investigation trace, the audit record, the ticket, the report. Each of those inherits whatever the model put in the reply — a reporter's email address, a phone number, content the model was manipulated into drafting.

The `GuardrailsHook` of [notebook 69](notebook_69_guardrails_security.md) screens the prompt, the tool calls and their results; this notebook adds the output side. `OutputFilterHook` filters the model's reply inside the agent loop, so what the loop records and returns is already clean, and `TopicPolicy` and `ContentPolicy` give it declarative rules to apply.

The example runs an agent whose system prompt asks it to quote a contact
address, with the hook redacting personally identifiable information (PII)
from the reply. It then builds a `TopicPolicy` for two offensive-tooling
topics and a `ContentPolicy` for two harmful-content categories, checks sample
text against each, and makes one short model call in each of those parts.

Both policies match declared word lists: you supply `TopicPolicy`'s keywords,
and `ContentPolicy` provides defaults per category, so they act on the subject
of a message rather than on its length or shape. The matching is literal,
case-insensitive substring matching, not classification: treat these as an
inexpensive first filter and put a moderation API or a classifier behind them
for anything adversarial.

What this notebook covers:

- `OutputFilterHook`: after each model call in the agent loop, redact PII in
  the agent's reply, or replace the reply when it violates a topic or content
  policy; the filtered reply is what the loop records and returns. It does
  not cover streamed chunks (with token streaming on, the chunks are forwarded
  unfiltered and the hook acts on the assembled reply) or the separate
  no-tools calls a run can make: the summary at the iteration limit, the
  follow-up after an empty reply, and structured-output repair.
- `TopicPolicy`: declarative topic blocking with keyword maps.
- `ContentPolicy`: harmful-content categories (violence, illegal activity).

`OutputFilterHook` runs inside the agent loop drawn here, after each model
call, so the reply the loop records is the filtered one; the gate, approval
store, audit trail, checkpointer and grounding in the drawing play no part in
this notebook:

{{ locus_diagram control-layer-flow }}

## Key lines

`OutputFilterHook(redact_pii=True)` is added to `AgentConfig.hooks`; the system
prompt asks the model to quote a contact email, and the `REDACTED_EMAIL` check
on `result.message` shows whether the reply was filtered before the loop
recorded it. The default patterns cover email addresses, US phone numbers, US
Social Security numbers and payment card numbers; pass `pii_patterns` to
replace them:

{{ locus_source examples/notebook_70_guardrails_advanced.py::example_pii_redaction }}

`TopicPolicy` takes a set of blocked topics and a map from each topic to its
keywords; a blocked topic with no entry in `keywords` is matched on its own
name, and an empty list matches nothing. `check()` returns the matched topic
name, or `None` when nothing matches; handed to
`OutputFilterHook(topic_policy=…)`, the same check runs on every reply and a
match replaces the reply with a refusal:

{{ locus_source examples/notebook_70_guardrails_advanced.py::example_topic_policy }}

`ContentPolicy` provides its own keyword defaults per category, so enabling
`violence` and `illegal_activity` is enough; `check()` again returns the
category or `None`:

{{ locus_source examples/notebook_70_guardrails_advanced.py::example_content_safety }}

## Run it

```bash
python examples/notebook_70_guardrails_advanced.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## See also

- [Topic and content policies](../concepts/safety.md#topic-and-content-policies-domain-restriction)
  — the same two policies attached to an `OutputFilterHook` on a live agent,
  and what to put behind them in production.
- [The hook lifecycle phases](../concepts/hooks.md#the-lifecycle-phases) — where
  `on_after_model_call` sits among the points a hook can subscribe to.

## Source

??? note "Full source — `examples/notebook_70_guardrails_advanced.py`"

    ````python
    --8<-- "examples/notebook_70_guardrails_advanced.py"
    ````
