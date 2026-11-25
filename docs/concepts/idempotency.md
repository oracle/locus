# Idempotency

> The SDK deduplicates identical tool calls within a documented execution scope.

The model may retry. With `idempotent=True`, an identical call later in the
same `agent.run()` reuses the recorded result instead of invoking the tool body
again. When `agent.run(..., thread_id=...)` reloads a checkpointed thread, the
new turn starts with an empty record, so an earlier turn's result does not
prevent a replay. A run continued with
`agent.resume()` does not check for a prior match; see
[How it works](#how-it-works).

The process may die. The [tool-effect ledger](#the-tool-effect-ledger) is
the second mechanism on this page: every executed call is recorded in the
thread's checkpointer as it returns, and a run that continues the turn after
a crash replays the record instead of running the tool again.

Read this page before running an agent whose tools refund, deploy, page,
email, or write.

## When to use `idempotent=True`

| Situation | `idempotent=True`? |
|---|---|
| Side-effecting tool with real-world cost (issue a refund, ship a deploy, page on-call) | Usually yes, plus a downstream idempotency key |
| Database write you can't trivially roll back | **yes** |
| External service that's already idempotent on its end | yes — the SDK dedupes the round-trip too |
| Read-only order lookup | no — re-reads are cheap, leave it to the model |
| Tool that *intentionally* generates a new entity each call (e.g. `mint_case_id`) | no — that breaks the contract |

## How it works

Inside a single agent run, for a tool declared `idempotent=True`, the SDK
takes the tool's `(name, arguments)` tuple as the key for each call the
model emits. **The first call with a given key hits the function body** and
the result is recorded. **A later call with the same key in the `run()` loop
short-circuits to the cached response** without invoking the body
(exceptions below).

The drawing follows one call through the loop's Execute
step: the key, the check against the run's record, and the result's way back
to the model.

{{ locus_diagram idempotent-execution }}

Declaring the tool:

```python
from locus.tools import tool

@tool(idempotent=True)
def issue_refund(order_id: str, amount: float) -> dict:
    """Refund an order. Re-fires within a run return the cached receipt."""
    return billing.refund(order_id, amount)
```

The argument match is the trust boundary:

- **Same call**: the model re-emits `issue_refund("ord-4821", 120.0)` after
  seeing the receipt → cache hit, body skipped.
- **Different call**: the model emits `issue_refund("ord-4822", 120.0)` →
  different key, body runs.

Dedup compares the arguments **as they reach the tool** — the dict the
model emitted, after any [`on_before_tool_call`](hooks.md) hooks have run.
The rules, one at a time:

- **Across iterations of one run.** A call from an earlier iteration of the
  same run is matched with a plain `dict == dict` equality on
  `(tool_name, arguments)`.
- **Within one model response.** A duplicate inside the same model response
  is matched on a sorted-key JSON serialisation of the arguments, before
  either call is recorded. The two checks agree for ordinary arguments, but
  not on number type: `120` and `120.0` match across iterations of one run,
  not within one response.
- **A new turn starts empty.** Both checks run in the `run()` loop and see
  only the current turn. A run that reloads a checkpointed thread with
  `thread_id` starts a new turn with an empty record, so a call from an
  earlier turn is never matched.
- **`resume()` does not check.** Neither check runs after `agent.resume()`
  answers an interrupt: the continued run executes each tool call without
  looking for a prior match, so an identical idempotent call runs the body
  again. A held call that `resume(perform_dangling=True)` runs is not
  recorded in `tool_executions` at all, so no later call can match it.
- **Both sides are post-hook.** Each call the loop runs, or serves from the
  cache, is recorded with the hook-modified arguments, so a before-hook that
  normalises arguments (for example, upper-casing an order id) makes a new
  call and its earlier twin agree. A call a hook cancels is recorded with the
  arguments the model emitted, so for an idempotent tool a later identical
  call can match that cancelled record and return its cancel message without
  running the body. A hook only affects calls made after it is installed; it
  does not retroactively match executions already recorded with
  un-normalised arguments.
- **No schema normalization.** Without such a hook, defaults are *not*
  filled in before the comparison, so a call that omits an optional argument
  and a call that passes that argument's default value are treated as
  **different keys** and both fire the body. (Dict equality is itself
  order-independent, so key order alone will not break a match.)

## Why this matters

### Actions with real-world cost

A model that calls `issue_refund` twice in one run is a common failure
mode. Sometimes it sees an ambiguous tool result and tries
again "to be sure". Sometimes the network glitches and the model
believes the call failed. Without idempotency, the customer is refunded
twice and finance opens a ticket for the duplicate.

Within the active run, an identical second call reuses the first result. The
same mechanism applies to a deploy (`deploy_service`), a page (`page_oncall`),
and the security-operations variant (`isolate_host`).

### Outbound side-effects

`page_oncall`, `open_case`, `slack_alert`, `block_indicator` — anything
that touches a human or a downstream system. With `idempotent=True`, an
identical call fires once within the run.

### Database writes you can't roll back

Insert into an audit table, append to a Kafka topic, sign a JWT —
operations where retrying isn't free. Idempotent tools reduce duplicate calls
inside the agent runtime; they do not create distributed exactly-once delivery.

## What it is *not*

| Concept | Idempotency is… | Idempotency is *not*… |
|---|---|---|
| Scope | within a single agent run | cross-run — a new run starts with an empty record, even on a checkpointed thread (a run *continued* after a crash is the same run; the [ledger](#the-tool-effect-ledger) covers it) |
| Failure | one fire per identical call in the `run()` loop | retry — if the body raises, the exception propagates as the cached "result" |
| Boundary | per-agent | network — two different agents both calling `issue_refund(o, a)` each fire once |

If you need cross-run idempotency, use an idempotent server-side endpoint.
Generate a stable operation key from the business
operation (for example, `refund:ORD-4821:v1`) and send it to that endpoint on
every attempt.

### Replays after a checkpoint reload

When `agent.run(..., thread_id=...)` reloads a checkpointed thread, the model
may decide to re-issue tool calls it has already seen. The new turn starts
with an empty `tool_executions` record, so an idempotent tool fires again.
Across turns, only a stable idempotency key sent to the downstream system
keeps the side effect from repeating.

## The external-success crash window

There is an unavoidable boundary between two systems:

1. the payment, deploy, or message succeeds in the external service;
2. the process stops before the SDK durably records the returned receipt;
3. the run resumes and cannot prove locally that the external call succeeded;
4. the call may be attempted again.

An in-memory cache cannot close that window. A checkpointer narrows it to the
time between the call returning and the next save; the tool-effect ledger
below narrows it to the ledger write itself, a single small record made the
moment the call returns.

What remains — the process dying between the
external success and that record — only the downstream system can close, by
recognizing the same stable idempotency key and returning the original result
rather than repeating the operation. Treat `idempotent=True` as agent-loop
deduplication, the ledger as crash-replay safety, and downstream keys as
distributed-operation safety.

## The tool-effect ledger

A checkpoint holds a tool call's result only once the iteration that made it
is saved. With the default `durability="sync"` that save follows each tool
batch; with `"async"` it lands in the background and with `"exit"` only at the
end of the run (see [Durability modes](checkpointers.md#durability-modes)). A
process killed in between has issued the refund and kept no record of it, and
a run that picks the turn up would ask the model again, get the same call
back, and issue it twice.

The ledger is the record. When it is on, the executor writes one entry per
executed call **the moment the call returns**, before the result is folded
into state or seen by a hook, keyed by

```
(thread id, step, tool name, canonical arguments)
```

where *step* is the iteration and *canonical arguments* is the call's
argument dict serialised with sorted keys, so argument order and whitespace do
not make a new key but a changed value does. The entry carries the full
`ToolResult` (content, error, structured content), the id of the call that
produced it, and the invocation (the run id) it belongs to.

Entries live in
the thread's own checkpointer — as checkpoints of the sidecar thread
`<thread_id>~ledger`, one per call under its key, plus a `manifest` naming the
run, its prompt and metadata — so every backend has the ledger without code of
its own.

The turn's final save deletes the sidecar once that checkpoint
holds every result. A sidecar found later is the trace of a turn that never
finished.

### When it is on

| `Agent(tool_ledger=...)` | `durability="sync"` | `"async"` / `"exit"` |
|---|---|---|
| `None` (default) | off | **on** |
| `True` | on | on |
| `False` | off | off |

It needs a checkpointer and a `thread_id`. The cost is one small write per
executed tool call, a manifest write once per turn and one delete at the end;
the [overhead table](checkpointers.md#overhead) puts numbers on it.

### Replay

Replay is on for the runs that *continue* a turn, and off for a fresh turn,
which can have no records yet:

- `agent.continue_turn(thread_id)` — from the thread's last checkpoint; when
  none of the turn's checkpoints landed (`"exit"`), from the turn's start as
  the manifest recorded it (the prompt, the run id).
- `agent.resume(invocation_id=..., thread_id=...)` — the same, naming the
  run that must be in flight; it takes no `response`.
- `agent.run(prompt, thread_id=..., invocation_id=...)` with the id the
  crashed run was started with. The caller keeps the id (a job id, a queue
  message id); a retry under the same id is the same invocation.

The continued run calls the model again from the last checkpoint. When the
model emits a call the ledger has — same step, same tool, same canonical
arguments, same invocation — the executor returns the recorded result and the
tool body does not run; `ToolResult.replayed` is `True` and the
`ToolCompleteEvent` carries the recorded output.

A call with no record runs
and is recorded. A record from another invocation (an earlier turn on the
thread that never finished) is never served: a new turn on the thread starts
with a fresh run id.

### What each annotation guarantees

| Tool | After a crash and `continue_turn` / `resume(invocation_id=...)` |
|---|---|
| no annotation | **At least once.** A call whose record landed is served from the ledger and does not run again. A process killed between the body returning and the record landing leaves no record, and the replayed call runs the body again — the model is not told it may be a repeat, so give the downstream system a stable idempotency key as well. |
| `idempotent=True` | The same, and the developer may prefer a re-run: `Agent(rerun_idempotent_on_replay=True)` runs an idempotent tool again on replay (which the annotation declares safe) and refreshes the record with the new result, for tools whose output should be current rather than the one the crashed run saw. Tools not marked idempotent are always served from the ledger. |

A hook's `retry` on `AfterToolCallEvent` runs the call again regardless — a
retry must run — and its result replaces the record. A tool that returns the
interrupt marker (`ask_user`, a gated tool holding a call) is recorded like any
other, so a replay after a crash at that point pauses again, on the same
hold.

```python
agent = Agent(
    model=...,
    tools=[issue_refund, lookup_order],
    checkpointer=postgresql_checkpointer(dsn=...),
    durability="exit",          # one checkpoint per turn; the ledger covers the run
)

# A worker picks the job up and runs it under the job's id.
agent.run_sync("Refund order 4821.", thread_id="case-4821", invocation_id=job.id)

# The worker died. Another one continues the same turn: issue_refund, already
# recorded, is replayed; anything the crashed run never reached runs now.
async for event in agent.resume(invocation_id=job.id, thread_id="case-4821"):
    ...
```

Temporal and DBOS run an agent in segments through the same runtime, so a
segment's calls are in the ledger too; a segment those engines will not retry
can be finished with `run_agent_segment(..., kind="continue")`, or
`agent.continue_turn(thread_id)` by hand. See
[`locus.durable`](https://github.com/oracle/locus/tree/main/src/locus/durable).

An engine that replays the loop itself, instead of running it in segments,
needs the loop to read the same times and ids on replay. It installs its own
clock, ids and random generator with
[`locus.core.sources`](../api/core.md#clock-ids-and-randomness).

## Practical recipe — refund approval

A canonical multi-agent idempotency shape: an agent, or several agents
debating, loops over a refund decision, then writes once.

```python
@tool(idempotent=True)
def issue_refund(order_id: str, amount: float) -> dict:
    return billing.refund(order_id, amount)

@tool(idempotent=True)
def page_oncall(case_id: str, summary: str) -> str:
    return pager.notify(team="support", subject=f"CASE {case_id}", body=summary)
```

The agent can iterate while reasoning about whether to refund. Repeated calls
with identical arguments reuse the cached result in the supported scope.
For a real payment or page, the tool body should also pass a stable
idempotency key to the provider.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Tool re-fires despite `idempotent=True` | Argument changed between calls. Check that the model isn't mutating order ids / amounts between iterations. |
| Idempotent cache survives across runs unexpectedly | It cannot — the record is the run's own `state.tool_executions`, and a new turn starts it empty even on a checkpointed thread. |
| Body raised first time, cache returns the exception | This is by design — the failure is part of the "result" of the first call. The model sees the failure and can react. To re-attempt, the model must change an argument. |
| Read-only lookup tagged `idempotent=True` | Harmless but wasteful — the cache hit savings are negligible vs the read itself. Leave it off. |

## Source and notebook

- [`@tool` decorator with idempotency hook](https://github.com/oracle/locus/blob/main/src/locus/tools/decorator.py)
- [`find_matching_execution`](https://github.com/oracle/locus/blob/main/src/locus/tools/executor.py) — the argument match itself.
- [`_maybe_cached_idempotent_result`](https://github.com/oracle/locus/blob/main/src/locus/agent/runtime_loop.py) — where the `Agent` runtime checks for a prior matching call before invoking the tool body.
- [`locus.memory.ledger`](https://github.com/oracle/locus/blob/main/src/locus/memory/ledger.py) — the tool-effect ledger: keys, records, the manifest, and the replay rules; the executor consults it in [`locus.tools.executor`](https://github.com/oracle/locus/blob/main/src/locus/tools/executor.py).
- [`notebook_03_agent_with_tools.py`](https://github.com/oracle/locus/blob/main/examples/notebook_03_agent_with_tools.py) — walks through the `@tool` decorator end-to-end (typed arguments, the tool loop, the start/complete event pair per call); it does not set `idempotent=True`.

## See also

- [Tools](tools.md) — the full `@tool` decorator surface.
- [Checkpointers](checkpointers.md) — reloading a thread by `thread_id`; the new turn starts with an empty record, so idempotency does not carry across it. Its [durability modes](checkpointers.md#durability-modes) section says when the ledger is on and what it costs.
