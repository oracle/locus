# The agent loop

Every Locus agent runs the same
loop. Four named nodes (`Think → Execute → Reflect → Terminate`), one
router that decides what runs next, one typed event stream, one piece of
immutable state that flows through. This page is the architectural
reference — what each node does, why it exists, what it emits, and how
to extend it.

{{ locus_diagram agent-loop }}

The loop is implemented in
[`src/locus/agent/runtime_loop.py`](https://github.com/oracle/locus/blob/main/src/locus/agent/runtime_loop.py):
`AgentRuntimeMixin.run`, which `Agent` mixes in. The separate
[`src/locus/loop/`](https://github.com/oracle/locus/tree/main/src/locus/loop)
package is a deprecated second ReAct implementation that `Agent` has
never used.

## Origin: ReAct, then refinement

The base pattern is **ReAct** ([Yao et al., 2022](https://arxiv.org/abs/2210.03629)) —
*Thought → Action → Observation*, repeated until the model decides
to stop. ReAct is now the default loop in most agentic SDKs.

The SDK keeps the spirit and adds three things:

- **Action becomes Execute** — a real node in the graph that owns
  tool dispatch *and* idempotency dedup, not a callback.
- **Reflect becomes its own node** — a structured self-evaluation step
  that runs *between* Execute and the next Think, when Reflexion is
  configured and its interval is due.
- **Terminate becomes algebra** — stopping is a tree of typed conditions
  composed with `&` and `|`, evaluated at the start of every iteration,
  before Think.

## State

Every node receives an `AgentState` and returns a new `AgentState`.
State is **immutable** — updates produce a new instance via
`state.with_message(...)`, `state.with_tool_execution(...)`,
`state.with_metadata(...)`. Hooks see frozen events; nodes see
frozen state.

The state value object carries:

- **`messages`** — the conversation in chat format, including the
  system prompt, the user's prompt, every model message, and every
  tool result.
- **`tool_executions`** — a chronological list of every tool call,
  its arguments and its result (or error), which Execute searches for
  idempotent dedup.
- **`iteration`** — the running iteration counter, consumed by
  termination conditions.
- **`metadata`** — a free-form dict for hooks and applications to
  thread their own data.

## Think

The Think node calls the configured model with the current message
list and gets back either a final answer or a set of tool calls to
fire. It emits a `ThinkEvent` with the model's reasoning content (when
the provider exposes it — extended-thinking models do; older models
don't) and a `ModelChunkEvent` per streamed token.

If the model made no tool calls, the run ends, after the grounding
pass when grounding is on (an answer that fails it can send the loop
round again). That is the default `completion_mode="auto"`;
`"explicit"` keeps looping instead. If the model made tool calls, the
loop goes to Execute.

## Execute

The Execute node fires the tool calls returned by Think. Two
behaviours make it different from a "just run the function" callback:

1. **Idempotent dedup.** For tools tagged `@tool(idempotent=True)`,
   Execute walks `state.tool_executions` and looks for a previous
   call with the same `(tool_name, arguments)` tuple. If found, the
   cached result is returned; the body never runs. The model can
   retry, loop, or panic without firing the tool a second time.
   Implementation:
   [`find_matching_execution()` `tools/executor.py:21`](https://github.com/oracle/locus/blob/main/src/locus/tools/executor.py#L21) — called from `_maybe_cached_idempotent_result` in [`agent/runtime_loop.py`](https://github.com/oracle/locus/blob/main/src/locus/agent/runtime_loop.py).

2. **Parallel dispatch.** Tool calls returned in the same model
   response fire concurrently. Execute awaits them all before the
   loop moves on. Errors in one tool don't cancel the
   others; each tool's error becomes a tool-error message in the
   state.

Execute emits `ToolStartEvent` and `ToolCompleteEvent` per call. A
cached short-circuit still emits a plain `ToolCompleteEvent` (there is
no separate cache-hit event type) — the dedup is recorded on state
instead: the `ToolExecution` is flagged `idempotent_cache_hit=True`, so
the run-trace can still tell a re-fired call from a fresh one.

## Reflect

The Reflect node runs a structured self-evaluation between Execute
and the next Think. It's gated on a fixed cadence, and otherwise
control goes straight back to Think:

- **Fixed cadence.** `reflexion=ReflexionConfig(evaluate_every_n_iterations=N)`
  reflects on every iteration whose number is a multiple of N;
  `reflexion=True` uses N=1, every iteration. Reflexion is off by
  default.

That interval is the only trigger. Tool errors and repeating tool
patterns are what the Reflector looks at once it runs, not reasons to
run it. Separately, in the default `completion_mode="auto"`, the loop
ends the run with `tool_loop` when the same tool calls with the same
arguments repeat across `tool_loop_threshold` (default 3) consecutive
iterations, whether or not Reflexion is on.

The Reflector itself
([`Reflector` class — `reasoning/reflexion.py:69`](https://github.com/oracle/locus/blob/main/src/locus/reasoning/reflexion.py#L69))
scores the iteration without calling a model: it checks recent
iterations for a repeating tool pattern, counts the iteration's
successful and failed tool calls, and adjusts a confidence score. The
loop then emits a `ReflectEvent` carrying the `assessment`,
`guidance`, and `new_confidence`. When the reflection carries
guidance, the loop adds it to the messages as a system message
(unless `include_guidance=False`), so the next Think sees it.

Two complementary reasoning add-ons:

- **Grounding** — runs on the final answer, before it is returned
  (default `completion_mode="auto"` only, only when the run actually
  called tools, and not on the part of a run continued with
  `agent.resume()` after an interrupt or hold): a judge model scores
  each claim extracted from the answer against the tool results, and a
  failing score injects replan guidance and re-enters the loop, up to
  `max_replans` times (default 2). Off by default; switch on via `Agent(grounding=True)`.
- **Causal** — a standalone `build_causal_chain()` builder that turns
  the events your agent surfaced into a cause-effect graph and flags
  cycles or contradictions. Run it over a finished trace; it isn't an
  `Agent(...)` flag.

Source:
[`GroundingEvaluator.evaluate_with_llm` `reasoning/grounding.py:420`](https://github.com/oracle/locus/blob/main/src/locus/reasoning/grounding.py#L420) ·
[`build_causal_chain` `reasoning/causal.py`](https://github.com/oracle/locus/blob/main/src/locus/reasoning/causal.py).

## Terminate

At the start of every iteration, before Think, the loop checks the
agent's `termination` condition. The condition is a typed object —
`MaxIterations`, `TokenLimit`, `TimeLimit`, `NoToolCalls`, `ToolCalled`,
`ConfidenceMet`, `TextMention`, or `CustomCondition` — composable
with `&` (And) and `|` (Or):

```python
from locus.core.termination import (
    MaxIterations, ToolCalled, ConfidenceMet, TokenLimit,
)

terminate = (
    ToolCalled("issue_refund") & ConfidenceMet(0.9)
) | MaxIterations(10) | TokenLimit(15_000)
```

The composite itself is a `TerminationCondition` whose `check()`
walks the tree and short-circuits on the first satisfied branch. The
loop emits a `TerminateEvent` carrying the satisfied condition's
reason, then exits.

Source:
[`src/locus/core/termination.py`](https://github.com/oracle/locus/blob/main/src/locus/core/termination.py).

## The router

There is no separate router object: transitions between nodes are
the control flow of `AgentRuntimeMixin.run` in
[`agent/runtime_loop.py`](https://github.com/oracle/locus/blob/main/src/locus/agent/runtime_loop.py).
(The `Router` class in the deprecated `locus/loop/` package is not on
this path.) The rules:

| From | To | When |
|---|---|---|
| iteration start | Terminate | a termination check fires (see [Terminate](#terminate)) |
| iteration start | Think | otherwise |
| Think | Execute | the model made tool calls |
| Think | Terminate | no tool calls, in `completion_mode="auto"` — through the grounding pass first when grounding is on and the run has called tools |
| grounding pass | next iteration | the answer fails grounding and replans remain |
| Execute | Reflect | Reflexion is configured and the iteration number is a multiple of `evaluate_every_n_iterations` |
| Execute | next iteration | otherwise |
| Reflect | next iteration | always (Reflect feeds back into the next Think) |

Termination is checked once per iteration, at the top of the loop
before Think, not after every node. A condition that becomes true
during Execute or Reflect ends the run at the next iteration boundary,
before the next model call.

## Events

Each node emits typed, **write-protected** events that hooks can
observe but never mutate:

| Event | Emitted by |
|---|---|
| `ThinkEvent` | Think, once per iteration |
| `ModelChunkEvent` | Think, per streamed chunk |
| `ToolStartEvent` | Execute, before each tool fires |
| `ToolCompleteEvent` | Execute, after each tool returns (sets `error` on failure) |
| `ReflectEvent` | Reflect, after each self-evaluation |
| `GroundingEvent` | the final-answer grounding pass |
| `InterruptEvent` | Execute, when a tool requests human input |
| `TerminateEvent` | Terminate, when the loop exits |

Events are Pydantic models with `model_config = {"frozen": True}`.
They serialise cleanly to JSON for SSE, telemetry, and structured
logging.

## Hooks

Hooks are how you observe and *steer* the loop without forking it. A
hook subclasses `HookProvider` (`locus.hooks.provider.HookProvider`)
and implements the lifecycle callbacks it cares about —
`on_before_invocation` / `on_after_invocation`,
`on_iteration_start` / `on_iteration_end`,
`on_before_tool_call` / `on_after_tool_call`,
`on_before_model_call` / `on_after_model_call`. There is no
`Continue`/`Cancel`/`Retry` directive return type. Instead a hook
steers the loop two ways:

- **Mutate the event in place.** The after-model-call and
  after-tool-call events carry a `retry` flag — set `event.retry = True`
  and the runtime re-runs that call (this is exactly how
  `ModelRetryHook` works).
- **Raise to abort.** Raising from any callback short-circuits the run
  (useful for budget guards).

Built-in hooks:
[`LoggingHook`](hooks.md), `StructuredLoggingHook`,
`TelemetryHook` (OpenTelemetry-compatible),
`ModelRetryHook`, `GuardrailsHook` (topic policy + PII redaction),
and `SteeringHook` (LLM-as-judge tool approval). Source:
[`hooks/builtin/__init__.py`](https://github.com/oracle/locus/blob/main/src/locus/hooks/builtin/__init__.py) (re-exports the four most-used hooks) ·
[`hooks/builtin/steering.py`](https://github.com/oracle/locus/blob/main/src/locus/hooks/builtin/steering.py) ·
[`hooks/builtin/retry.py`](https://github.com/oracle/locus/blob/main/src/locus/hooks/builtin/retry.py).

## A concrete example

Consider this prompt against a support agent with `lookup_order` +
`issue_refund`:

> *"Process refund request R-42: verify order ord-4821, refund if eligible."*

Iteration by iteration:

| # | Node | What happens |
|---|---|---|
| 1 | Think | Model emits a tool call: `lookup_order(order_id="ord-4821")`. Streams `ThinkEvent` + `ModelChunkEvent`s. |
| 1 | Execute | Runs `lookup_order`. Tool is **not** marked idempotent (read-only) so no dedup. Result added to `state.tool_executions`. Emits `ToolStartEvent`, `ToolCompleteEvent`. |
| 1 | Reflect | Skipped — this agent reflects every second iteration (`evaluate_every_n_iterations=2`), so reflection is not due. Control goes back to Think. |
| 1 | Terminate? | Checked at the start of the next iteration: `MaxIterations(8)` not yet hit. `ToolCalled("issue_refund")` not satisfied. Continue. |
| 2 | Think | The order record confirms an eligible return, so the model emits `issue_refund(order_id="ord-4821", request_id="R-42")`. |
| 2 | Execute | Tool is `idempotent=True`. Execute walks `state.tool_executions` for an earlier `issue_refund` call with equal arguments (`{order_id: "ord-4821", request_id: "R-42"}`). None — so the body fires. Refund receipt `RF-58291` returned. |
| 2 | Reflect | Reflexion runs (iteration 2 is due). The iteration's one successful call raises confidence from 0.0 to 0.15. `ReflectEvent` emitted. |
| 2 | Terminate? | Checked at the start of iteration 3: `ToolCalled("issue_refund")` ✓ but `ConfidenceMet(0.9)` ✗ (0.15), so the AND branch is false, and `MaxIterations(8)` is not hit. Continue. |
| 3 | Think | The refund is done, so the model answers in text with no tool calls. In the default `completion_mode="auto"` the run ends with `TerminateEvent(reason="complete")`; `result.stop_reason == "complete"`. |

Total: **three iterations, two tool calls, one Reflect, one Terminate**.

If the model had hallucinated and re-emitted `issue_refund` with the
same args on iteration 3 (it didn't, but it could), Execute would
have caught the duplicate `(name, arguments)` match, returned the cached
`RF-58291`, and flagged that `ToolExecution` with
`idempotent_cache_hit=True` so the trace shows the short-circuit
clearly. Refunding an already-refunded order is a no-op — exactly
what idempotency guarantees.

## Stop reasons

Every run ends with a `TerminateEvent` whose `reason` field names the
satisfied condition. The named reasons you'll see:

| Reason | When |
|---|---|
| **`MaxIterations`** | the iteration counter hit the configured ceiling |
| **`TokenLimit`** | cumulative model tokens exceeded the budget |
| **`TimeLimit`** | wall-clock budget exceeded |
| **`NoToolCalls`** | the model emitted text and no tool calls (the natural "I'm done" signal) |
| **`ToolCalled`** | a specific tool fired (with `require_success=True`, only a call that did not error counts) |
| **`ConfidenceMet`** | the Reflexion confidence score cleared the threshold |
| **`TextMention`** | the final message contained the configured text (case-insensitive substring by default) |
| **`CustomCondition`** | a user-supplied `fn(state, **ctx)` returned `(True, reason)` |
| **`cancelled`** | the caller called `agent.cancel()` |
| **`error`** | the model (or a node, or a hook inside the loop) raised; the loop emits `TerminateEvent(reason="error")` and re-raises the exception, so `arun()` / `run_sync()` raise it instead of returning a result |

Note that `result.stop_reason` is **normalized** to a fixed set of
literals (`complete`, `terminal_tool`, `confidence_met`,
`max_iterations`, `tool_loop`, `no_tools`, `grounding_failed`,
`token_budget`, `cost_budget`, `time_budget`, `context_exhausted`,
`interrupted`, `error`, `cancelled`),
while `TerminateEvent.reason` carries the finer branch-level string.
Composite conditions (`OrCondition`, `AndCondition`) report the
underlying satisfied leaf token(s), so the event reason always points
at the branch that fired.

## Cancellation

Three ways to stop a running agent without waiting for the natural
terminate condition:

### From a hook

Any hook can **raise** from a callback to abort the run. The current
node finishes (so a tool call is not torn out mid-flight), then the
loop unwinds. Useful for budget guards; the `SteeringHook` uses the
same family of callbacks to vote on each tool call before it fires.

```python
from locus.hooks.provider import HookProvider

class BudgetGuard(HookProvider):
    async def on_iteration_start(self, iteration: int, state) -> None:
        if state.total_tokens_used > 100_000:
            raise RuntimeError(f"token budget exceeded at {state.total_tokens_used} tokens")
```

### From the caller

```python
import asyncio

run = asyncio.create_task(agent.arun(prompt))
# … later, on a timeout, on a user click, on whatever:
run.cancel()
```

The cancellation lands at whatever the run is awaiting — typically a
model call or a tool call — rather than between nodes. In-flight tool
calls running on asyncio see the standard `CancelledError` propagate
through their await points; cooperative tools can catch it to release
resources before re-raising.

### Via `agent.cancel()`

`agent.cancel()` sets a flag the loop checks at the start of each
iteration. The loop exits at the next iteration boundary with a
`TerminateEvent(reason="cancelled")` (`result.stop_reason ==
"cancelled"`). For thread-bound runs, the
state still flushes to the checkpointer before exit, so the
conversation can resume cleanly later.

## Common problems

### Context window exhaustion

Long-running agents accumulate every model message and every tool
result in `state.messages`. Eventually the next Think exceeds the
provider's context window and fails. Three remedies:

1. **Tune the conversation manager** — an agent with no
   `conversation_manager=` already gets one: `LLMCompactor(context_length=...)`
   when the model has registered metadata (it prunes stale tool output
   and keeps a token-budgeted tail, with no extra model calls), and a
   `SlidingWindowManager` otherwise. Override that default with
   `Agent(conversation_manager=LLMCompactor(summarize_fn=..., context_length=...))`
   to protect the system prompt and the most recent turns, then summarise
   the middle on demand. Pass the model's window as `context_length=`:
   a compactor you construct assumes 128,000 tokens and does not inherit
   the window the default looked up. Source:
   [`src/locus/memory/compactor.py`](https://github.com/oracle/locus/blob/main/src/locus/memory/compactor.py).
2. **Tighten tool result size** — return concise structured data,
   not blobs of HTML. The model rarely needs the full source.
3. **Decompose with multi-agent** — let an orchestrator delegate
   long sub-tasks to specialists with their own short context.

### Tool selection mistakes

If the model picks the wrong tool, it's almost always the tool's
description that's the bug. Tool docstrings are part of the
contract the model sees. Be explicit: *when to use this tool, when
not to, what the inputs mean.*

### Loops that never converge

Symptom: the agent calls the same tool with slightly-different args,
five times, then hits `MaxIterations` and gives up. Two fixes:

- **Reflect on cadence** — `reflexion=True` (with the default
  cadence) catches loops via the Reflector's pattern detector.
- **Terminate on no-progress** — compose a `CustomCondition` that
  fires when the latest tool result equals the previous one.

### Idempotency key collisions

If two semantically-different calls happen to carry the same tool
name and equal arguments, Execute will dedup the second one and the
agent will get a stale receipt. Fix by including a per-request
identifier in the args (e.g., `request_id`) so distinct calls have
distinct arguments.

## Putting it together

```python
from locus.agent import Agent, ReflexionConfig
from locus.tools import tool
from locus.memory.backends import S3Backend
from locus.core.termination import (
    MaxIterations, ToolCalled, ConfidenceMet,
)
from locus.hooks.builtin import StructuredLoggingHook

@tool(idempotent=True)
def issue_refund(order_id: str, request_id: str) -> dict:
    return billing.refund(order_id, request_id)

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_order, issue_refund],
    system_prompt="You are a customer-support agent.",
    reflexion=ReflexionConfig(evaluate_every_n_iterations=2),  # Reflect every 2nd iteration, as in the walkthrough
    grounding=True,                    # claim verification
    checkpointer=S3Backend(...),
    hooks=[StructuredLoggingHook()],     # logs at INFO by default
    termination=(
        ToolCalled("issue_refund") & ConfidenceMet(0.9)
    ) | MaxIterations(8),
)

async for event in agent.run("Process refund request R-42: verify the order, refund if eligible.",
                             thread_id="th-q3-2026"):
    match event:
        case ThinkEvent(reasoning=r) if r:    print(f"💭 {r}")
        case ToolStartEvent(tool_name=n):     print(f"🔧 {n}")
        case ReflectEvent(guidance=g) if g:   print(f"🪞 {g}")
        case TerminateEvent(reason=why):      print(f"✅ {why}")
```

## What you can configure

| `Agent(...)` argument | Loop effect |
|---|---|
| `model=` | which provider Think calls |
| `tools=` | what Execute can dispatch |
| `system_prompt=` | prepended to the message list before the first Think |
| `reflexion=True` | enables Reflect on every iteration; `ReflexionConfig(evaluate_every_n_iterations=N)` for every Nth |
| `grounding=True` | checks the final answer's claims against tool results before it returns |
| `checkpointer=` | persists state at the end of a run and before an interrupt (for a run with a `thread_id`), so the run can resume after restart; pair with `checkpoint_every_n_iterations=` to also save inside the loop |
| `conversation_manager=` | summarises / prunes long histories before they exceed the context window |
| `hooks=` | observe and steer every event |
| `termination=` | the algebra the loop checks at the start of each iteration |
| `max_iterations=` | the built-in iteration cap (default 20); reaching it triggers one last model call, without tools, for a final summary |
| `tool_execution=` | `"concurrent"` (default) or `"sequential"` |

## Where to next

- [Tools](tools.md) — how to write the things Execute calls.
- [Idempotency](idempotency.md) — why and when to mark a tool idempotent.
- [Reasoning](reasoning.md) — Reflexion / Grounding / Causal in detail.
- [Termination](termination.md) — every built-in condition + composition.
- [Events & Streaming](events.md) — the typed event taxonomy.
- [Hooks](hooks.md) — observe + steer.
- [Multi-agent](multi-agent.md) — the loop runs inside seven coordination patterns plus A2A.
