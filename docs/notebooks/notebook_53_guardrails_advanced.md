# Advanced Guardrails

Three policy types that work on top of the basic `GuardrailsHook` from
notebook 46. They focus on what the agent talks about, not just what
characters appear in the prompt.

- `TopicPolicy`: declarative topic blocking with keyword maps.
- `ContentPolicy`: harmful-content categories (violence, illegal activity).
- `OutputFilterHook`: redact PII or block topics in the agent's reply
  before it leaves the process.

Run it (OCI Generative AI is the default; auto-detected from `~/.oci/config`):

    python examples/notebook_53_guardrails_advanced.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_53_guardrails_advanced.py

## Source

```python
--8<-- "examples/notebook_53_guardrails_advanced.py"
```
