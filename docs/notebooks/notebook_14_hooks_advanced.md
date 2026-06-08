# Advanced Hooks

Notebook 12 covered hook basics. This one focuses on the safety
properties Tulip enforces on the event objects hooks see, and on the
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
.venv/bin/python examples/notebook_14_hooks_advanced.py
```

Uses the bundled mock model by default. Set `TULIP_MODEL_PROVIDER` to
openai / anthropic for a live model; keep `TULIP_MODEL_PROVIDER=mock`
for offline runs.

Prerequisite: notebook 12.

## Source

```python
--8<-- "examples/notebook_14_hooks_advanced.py"
```
