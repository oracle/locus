# Guardrails and Security

Block dangerous calls before the model sees them. Each part wires a
guardrail into a real agent run and prints the model round-trip cost,
so the safety policy is exercised live, not described in the abstract.

- `GuardrailsHook` with a typed `GuardrailConfig` (block list, length caps,
  default action).
- PII detection and redaction on user input.
- Content pattern blocking (SQL injection, path traversal, shell escapes).
- Tool allowlist vs denylist.
- Stacked hooks via `HookRegistry` plus a separate `ContentFilterHook`.

Run it (OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider):

    python examples/notebook_69_guardrails_security.py

Offline / no credentials:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_69_guardrails_security.py

Pin a specific model:

    LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_69_guardrails_security.py

## Source

````python
--8<-- "examples/notebook_69_guardrails_security.py"
````
