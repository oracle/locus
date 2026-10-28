# Termination

When does an agent stop — the refund resolved, the deploy verified,
the incident contained? Locus
answers that with a typed, composable **algebra of stop conditions** —
small classes that each return `True` when the run should end, combined
with `&` (and) and `|` (or).

```python
from locus.core.termination import (
    MaxIterations, ToolCalled, ConfidenceMet, TextMention,
)

termination = (
    (ToolCalled("issue_refund") & ConfidenceMet(0.9))
    | TextMention("ESCALATE")
    | MaxIterations(10)
)
```

Read it left to right: *stop when the refund is issued and we're
confident in the call, **or** the model flagged it for human review
("ESCALATE"), **or** we hit ten iterations* — the last branch caps
runaway tool calls on a live account. A containment flow reads the
same way — `ToolCalled("isolate_host") & ConfidenceMet(0.9)` — the
algebra doesn't care which domain the tools come from.

This is one of the SDK's signature primitives. Every stop condition is
inspectable, unit-testable, and serialisable — no hand-rolled `if`
ladders sprinkled through the response loop.

## When to pick which condition

| Situation | Use |
|---|---|
| Hard cap on tool calls against live systems / runaway protection | `MaxIterations`, `TokenLimit`, `TimeLimit` |
| The run is "done" when one specific tool fires | `ToolCalled("issue_refund")` |
| The model is confident and Reflexion agrees | `ConfidenceMet(0.85)` (requires `reflexion=True`) |
| The agent should write a summary, not call more tools | `NoToolCalls()` |
| The run ends when the model flags for human review | `TextMention("ESCALATE")` |
| Custom predicate over `AgentState` | `CustomCondition(fn)` |

## Getting started

### 1. Pick one condition

```python
from locus.agent import Agent
from locus.core.termination import MaxIterations

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_order, issue_refund],
    termination=MaxIterations(8),
)
```

A single condition is a perfectly fine starting point. `MaxIterations`
is the safety net every production agent should have — it stops the
loop from grinding your APIs forever on a request it can't resolve.

### 2. Combine with `&` and `|`

```python
from locus.core.termination import (
    MaxIterations, ToolCalled, ConfidenceMet,
)

termination = (
    ToolCalled("issue_refund")        # the refund went out
    & ConfidenceMet(0.85)             # we believe the call
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
# → "terminal_tool"   (a fixed StopReason literal)
```

`result.stop_reason` is **normalized** to one of a fixed set of
`StopReason` literals (`complete`, `terminal_tool`, `confidence_met`,
`max_iterations`, `tool_loop`, `no_tools`, `grounding_failed`,
`token_budget`, `cost_budget`, `time_budget`, `context_exhausted`,
`interrupted`, `error`, `cancelled`).
The fuller, branch-level string is on the `TerminateEvent.reason` event
field — each condition's `check()` returns a short reason token
(e.g. `tool_called:isolate_host`, `confidence_met`), and a composite
joins them with `AND`. So watch `TerminateEvent.reason` in your event
stream when you need to know *exactly* which branch fired; key metrics
and control flow on the normalized `result.stop_reason`.

## Built-in conditions

| Condition | Triggers when |
|---|---|
| `MaxIterations(n)` | The ReAct loop has run `n` turns. |
| `TokenLimit(n)` | Cumulative model tokens exceed `n`. |
| `TimeLimit(seconds)` | Wall-clock budget exceeded. |
| `NoToolCalls()` | The most recent turn produced a summary and zero tool calls. |
| `ToolCalled(name, *, require_success=False)` | A specific tool was called — e.g. `ToolCalled("issue_refund")`. With `require_success=True`, a call that raised doesn't count; a call a before-tool hook cancelled is recorded without an error, so it still counts. |
| `ConfidenceMet(threshold)` | Reflexion confidence ≥ threshold. |
| `TextMention(text, case_sensitive=False)` | The last assistant message contains `text` as a substring; case-insensitive by default. |
| `CustomCondition(fn)` | `fn(state, **ctx) -> (stop, reason)` — anything you can write in Python. |

Every condition takes `AgentState` and its `check()` returns a
`(stop: bool, reason: str | None)` tuple. The loop evaluates the
algebra once per iteration, at the start, before Think; the first one
that stops wins.

## Custom conditions

Write any predicate over `AgentState`:

```python
from locus.core.termination import CustomCondition, MaxIterations

def refund_settled(state, **ctx) -> tuple[bool, str | None]:
    # Stop only once issue_refund actually returned a receipt —
    # not just when the call was emitted. `e.result` is a string (the
    # tool's output, or the cancel message if a hook blocked the call),
    # or None if the call raised.
    settled = any(
        e.tool_name == "issue_refund" and "receipt" in (e.result or "")
        for e in state.tool_executions
    )
    return settled, ("refund_settled" if settled else None)

termination = CustomCondition(refund_settled) | MaxIterations(15)
```

The loop calls `fn(state, **ctx)` and always passes `last_message` and
`no_tool_calls` as keywords, so the `**ctx` parameter is required even
if you ignore it. Return the `(stop, reason)` tuple — the loop unpacks
it directly, so a bare `bool` raises `TypeError: cannot unpack
non-iterable bool object`. The reason string is what surfaces on
`TerminateEvent.reason`.

Custom conditions compose with built-ins exactly the same way — `&`
and `|` work across the whole hierarchy.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Agent always stops at `MaxIterations` | The "done" condition never fires — model isn't calling `issue_refund`, or confidence never reaches the threshold. Lower the threshold or check the tool name. |
| `&` / `\|` precedence surprises | Python's normal precedence applies: `&` binds tighter than `\|`. Add parentheses when in doubt — `(A & B) \| C` reads cleaner anyway. |
| `ConfidenceMet` never trips | `reflexion=True` is required — without it, confidence stays at the default, so the agent never early-stops on a high-confidence resolution. |
| `ToolCalled("issue_refund")` fires before the refund settles | It checks the *call*, not the *result* — `require_success=True` additionally skips a call that raised, so a terminal tool can reject a submission by raising and the loop continues. For a check on the returned content, pair with `ConfidenceMet` or a `CustomCondition` that inspects `tool_executions`. |

## Source and notebook

- [`notebook_08_termination.py`](https://github.com/oracle/locus/blob/main/examples/notebook_08_termination.py) — runnable algebra examples.
- [`locus.core.termination`](https://github.com/oracle/locus/blob/main/src/locus/core/termination.py) — every condition class, plus `__or__` / `__and__`.

## See also

- [Reasoning](reasoning.md) — pair `ConfidenceMet` with `reflexion=True`.
- [Events](events.md) — `TerminateEvent.reason` carries the branch-level reason token(s).
- [Agent loop](agent-loop.md) — where conditions evaluate inside the ReAct cycle.
