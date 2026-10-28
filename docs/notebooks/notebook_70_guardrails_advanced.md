# Advanced Guardrails

Three controls that work alongside the basic `GuardrailsHook` from
[Guardrails and Security](notebook_69_guardrails_security.md). The two
policies match against declared topic and category word lists — you supply
`TopicPolicy`'s keywords (a blocked topic with no entry in `keywords` is matched on its own name; an empty list matches nothing),
`ContentPolicy` ships defaults per category — so
they act on the subject of a message rather than on its length or shape.
The matching is literal, case-insensitive substring matching, not
classification: treat these as a cheap first filter and put a moderation
API or a classifier behind them for anything adversarial.

- `OutputFilterHook`: after each model call in the agent loop, redact PII in
  the agent's reply, or replace the reply when it trips a topic or content
  policy; the filtered reply is what the loop records and returns. It does
  not cover raw stream chunks (with token streaming on they reach the
  consumer before the hook runs) or the separate no-tools calls a run can
  make: the summary at the iteration limit, the follow-up after an empty
  reply, and structured-output repair.
- `TopicPolicy`: declarative topic blocking with keyword maps.
- `ContentPolicy`: harmful-content categories (violence, illegal activity).

Run it (OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider):

    python examples/notebook_70_guardrails_advanced.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_70_guardrails_advanced.py

## Source

````python
--8<-- "examples/notebook_70_guardrails_advanced.py"
````
