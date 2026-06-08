# Advanced Guardrails

Three policy types that work on top of the basic `GuardrailsHook` from
notebook 46. They focus on what the agent talks about, not just what
characters appear in the prompt.

- `TopicPolicy`: declarative topic blocking with keyword maps.
- `ContentPolicy`: harmful-content categories (violence, illegal activity).
- `OutputFilterHook`: redact PII or block topics in the agent's reply
  before it leaves the process.

Run it (defaults to the bundled mock model; set `TULIP_MODEL_PROVIDER` to `openai` / `anthropic` for a live model):

    python examples/notebook_51_guardrails_advanced.py

Offline:

    TULIP_MODEL_PROVIDER=mock python examples/notebook_51_guardrails_advanced.py

## Source

```python
--8<-- "examples/notebook_51_guardrails_advanced.py"
```
