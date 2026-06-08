# Agent Hooks

Hooks are middleware for agents. Subclass `HookProvider`, override the
callbacks you need, and Tulip invokes them at four lifecycle points:
before/after the invocation, and before/after each tool call. Use them
to add logging, timing, validation, guardrails, or any cross-cutting
concern without touching the agent or its tools.

What you'll learn:

- Writing a `HookProvider` and registering it on an `Agent`.
- The four callback points and what they receive.
- Using `HookPriority` to control execution order.
- Mutating `event.arguments` from `on_before_tool_call` to rewrite the
  call before the tool runs.
- Composing several hooks on one agent.

Run it:

```
.venv/bin/python examples/notebook_12_agent_hooks.py
```

Uses the bundled mock model by default. Set `TULIP_MODEL_PROVIDER` to
openai / anthropic for a live model; keep `TULIP_MODEL_PROVIDER=mock`
for offline runs.

Prerequisite: notebook 11.

## Source

```python
--8<-- "examples/notebook_12_agent_hooks.py"
```
