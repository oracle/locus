# Termination

When does an agent stop? Tulip
answers that with a typed, composable **algebra of stop conditions** —
small classes that each return `True` when the run should end, combined
with `&` (and) and `|` (or).

```python
from tulip.core.termination import (
    MaxIterations, ToolCalled, ConfidenceMet, TextMention,
)

termination = (
    (ToolCalled("send_summary") & ConfidenceMet(0.9))
    | TextMention(r"\bDONE\b")
    | MaxIterations(10)
)
```

Read it left to right: *stop when we sent the summary and we're
confident, **or** the model said "DONE", **or** we hit ten iterations*.

This is one of the SDK's signature primitives. Every stop condition is
inspectable, unit-testable, and serialisable — no hand-rolled `if`
ladders sprinkled through the loop.

## When to pick which condition

| Situation | Use |
|---|---|
| Hard cap on cost / runaway protection | `MaxIterations`, `TokenLimit`, `TimeLimit` |
| The work is "done" when one specific tool fires | `ToolCalled("submit_order")` |
| The model is confident and Reflexion agrees | `ConfidenceMet(0.85)` (requires `reflexion=True`) |
| The agent is supposed to write text, not call more tools | `NoToolCalls()` |
| The run ends when the model emits a sentinel | `TextMention(r"\bSHIP\b")` |
| Custom predicate over `AgentState` | `CustomCondition(fn)` |

## Getting started

### 1. Pick one condition

```python
from tulip.agent import Agent
from tulip.core.termination import MaxIterations

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, summarise],
    termination=MaxIterations(8),
)
```

A single condition is a perfectly fine starting point. `MaxIterations`
is the safety net every production agent should have.

### 2. Combine with `&` and `|`

```python
from tulip.core.termination import (
    MaxIterations, ToolCalled, ConfidenceMet,
)

termination = (
    ToolCalled("send_summary")        # the work happened
    & ConfidenceMet(0.85)             # we believe the result
) | MaxIterations(8)                  # …or the safety cap
```

`&` and `|` are real Python operator overloads (`__and__` / `__or__`)
on `TerminationCondition`, so the result is a typed
`AndCondition` / `OrCondition` you can keep composing, log, or pass
through tests.

### 3. Inspect what stopped the run

```python
result = agent.run_sync(prompt)
print(result.stop_reason)
# → "ToolCalled('send_summary') and ConfidenceMet(0.85)"
```

Each condition has a `__repr__` that round-trips to its constructor,
so logs and traces tell you *exactly* which branch of the algebra
fired.

## Built-in conditions

| Condition | Triggers when |
|---|---|
| `MaxIterations(n)` | The ReAct loop has run `n` turns. |
| `TokenLimit(n)` | Cumulative model tokens exceed `n`. |
| `TimeLimit(seconds)` | Wall-clock budget exceeded. |
| `NoToolCalls()` | The most recent turn produced text and zero tool calls. |
| `ToolCalled(name, args=None)` | A specific tool fired (with optional args predicate). |
| `ConfidenceMet(threshold)` | Reflexion confidence ≥ threshold. |
| `TextMention(pattern)` | Final message contains a regex match. |
| `CustomCondition(fn)` | `fn(state) -> bool` — anything you can write in Python. |

Every condition takes `AgentState` and returns `bool`. They run after
each iteration; the first `True` wins.

## Custom conditions

Write any predicate over `AgentState`:

```python
from tulip.core.termination import CustomCondition

def revenue_extracted(state) -> bool:
    return any(
        "revenue_usd" in (e.result or {})
        for e in state.tool_executions
    )

termination = CustomCondition(revenue_extracted) | MaxIterations(15)
```

Custom conditions compose with built-ins exactly the same way — `&`
and `|` work across the whole hierarchy.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Agent always stops at `MaxIterations` | The "happy-path" condition never fires — model isn't calling the tool you keyed on, or confidence never reaches the threshold. Lower the threshold or check the tool name. |
| `&` / `\|` precedence surprises | Python's normal precedence applies: `&` binds tighter than `\|`. Add parentheses when in doubt — `(A & B) \| C` reads cleaner anyway. |
| `ConfidenceMet` never trips | `reflexion=True` is required — without it, confidence stays at the default. |
| `ToolCalled("x")` fires before the tool finishes | It checks the *call*, not the *result*. Pair with `ConfidenceMet` or a `CustomCondition` that inspects `tool_executions`. |

## Source and notebook

- [`notebook_15_termination.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_15_termination.py) — runnable algebra examples.
- [`tulip.core.termination`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/core/termination.py) — every condition class, plus `__or__` / `__and__`.

## See also

- [Reasoning](reasoning.md) — pair `ConfidenceMet` with `reflexion=True`.
- [Events](events.md) — `TerminateEvent.reason` carries the condition's `repr`.
- [Agent loop](agent-loop.md) — where conditions evaluate inside the ReAct cycle.
