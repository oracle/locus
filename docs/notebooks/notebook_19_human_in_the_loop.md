# Human-in-the-Loop

Pause a graph mid-execution, ask a human, then resume with their
answer. `interrupt(payload)` pauses the running node and returns
control to the caller with `result.is_interrupted = True`. The caller
inspects the payload, gets a response (web form, CLI prompt, Slack
reply — whatever makes sense), then calls
`graph.execute(Command(update=..., resume=...))` to continue. The
same node restarts; its `interrupt()` call now returns the supplied
response.

What you'll see:

- `interrupt(payload)` — surface a payload to the caller.
- `Command(update=..., resume=...)` — resume execution.
- Multiple interrupts in one workflow.
- Conditional interrupts (only ask for higher-risk cases).
- `graph.config.interrupt_before = [...]` — pause before specific nodes
  without modifying them.

This notebook doesn't call any LLM, so the model provider doesn't
matter:

```bash
TULIP_MODEL_PROVIDER=mock python examples/notebook_19_human_in_the_loop.py
```

## Source

```python
--8<-- "examples/notebook_19_human_in_the_loop.py"
```
