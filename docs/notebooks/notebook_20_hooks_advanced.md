# Advanced Hooks

Notebook 12 covered hook basics. This one focuses on the safety
properties Locus enforces on the event objects hooks see, and on the
control levers a hook can pull mid-flight: `event.cancel` to skip a tool
call, and `event.retry` to re-issue a model call.

What you'll learn:

- Most fields on hook event objects are read-only. Mutating
  `event.tool_name` raises `AttributeError` — that's the framework
  protecting the agent's invariants.
- `event.arguments` and `event.cancel` *are* writable.
- Setting `event.cancel = "<reason>"` in `on_before_tool_call` skips the
  call and feeds the reason back as the tool's result.
- Priority ordering is reversed on "after" callbacks so cleanup unwinds
  LIFO.

Run it:

```
.venv/bin/python examples/notebook_20_hooks_advanced.py
```

Uses the OCI Generative AI default provider (canonical id:
`openai.gpt-4.1` or `meta.llama-3.3-70b-instruct`). For offline runs set
`LOCUS_MODEL_PROVIDER=mock`; OpenAI, Anthropic and Ollama also work.

Prerequisite: notebook 12.

## Source

```python
--8<-- "examples/notebook_20_hooks_advanced.py"
```
