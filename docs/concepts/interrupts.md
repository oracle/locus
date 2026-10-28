---
title: Interrupts
---

# Interrupts & human-in-the-loop

Sometimes the agent shouldn't decide alone. A human approves the
$4,000 refund. A release manager signs off before the deploy goes
out. Policy requires a named person between recommendation and
action.

Locus treats human approval as
**a tool the model can call** — same shape as any other tool, except
it surfaces a question to your app and resumes when the human
responds.

For consequential actions the check is enforced by the runtime, not
offered to the model — a policy that answers hold (`require_human`) stops the
action at the [admission gate](control-layer.md#the-enforcement-boundary),
and with `on_refusal="interrupt"` and an approval store the run itself
pauses until a named person decides ([below](#a-hold-that-pauses-the-run)).

## The shape

The SDK ships a built-in `ask_user` tool. The model calls it like any
other tool; instead of returning a value, the run **pauses** — the
agent yields an `InterruptEvent` and stops. Your app reads the question
off the event, gets an answer, and calls `agent.resume(answer)` to
continue.

```python
from locus.agent import Agent
from locus.core.events import InterruptEvent
from locus.tools import tool

@tool(idempotent=True)
def issue_refund(order_id: str, amount: float) -> dict:
    return billing.refund(order_id, amount)

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_order, issue_refund],
    # ``ask_user`` is auto-registered only in explicit-completion mode; the
    # default is "auto", where it is absent and the prompt below would ask
    # for a tool the agent does not have.
    completion_mode="explicit",
    system_prompt=(
        "You are a customer-support agent. "
        "Always call ask_user for approval before issue_refund."
    ),
)

async for event in agent.run("Refund order ord-4821 if the return is eligible."):
    if isinstance(event, InterruptEvent):
        answer = input(f"{event.question} ")   # or Slack / web / email
        async for resumed in agent.resume(answer):
            print(resumed)
```

When the model calls `ask_user`, the runtime captures the question and
yields an `InterruptEvent(question=..., options=..., interrupt_id=...)`,
then pauses. Your app surfaces the question to a human and threads the
answer back via `agent.resume(...)`. You can also expose your own
pause-for-input by calling
[`interrupt(...)`](https://github.com/oracle/locus/blob/main/src/locus/core/interrupt.py)
from inside a tool body — `ask_user` is just the built-in wrapper.

## Three ways the human responds

### Synchronous — read from stdin

The simplest case for CLI agents and demos: write your tool to call
`input("[y/N] ")` directly. The thread blocks until the human types.

```python
@tool
def cli_approval(reason: str) -> dict:
    answer = input(f"{reason}\nApprove? [y/N] ").strip().lower()
    return {"approved": answer == "y", "reason": reason}
```

### Async — checkpointer-mediated

For long-running workflows, the agent yields an `InterruptEvent` and
pauses when the model calls `ask_user`. A separate process (browser,
Slack action, email link) eventually resumes. `agent.resume(...)` is an
**async generator**, so you iterate it — you don't `await` it:

```python
async for event in agent.resume("approved"):
    handle(event)
```

The loop threads the response into the next Think and continues
streaming events.

### Steering — a second model votes

Not strictly human-in-the-loop, but lives in the same family. The
`SteeringHook` runs an LLM-as-judge on every tool call before it
fires:

```python
from locus.hooks.builtin import SteeringHook
from locus.models import get_model

agent = Agent(
    ...,
    hooks=[SteeringHook(
        # A model instance, not a provider string — the hook calls
        # ``.complete()`` on whatever it is given.
        model=get_model("{{ locus_example_model }}"),
        policy="Reject any tool call that doesn't match the user's stated request.",
    )],
)
```

When the judge votes "no", the call is rejected and the agent
re-plans. This is policy enforcement, not human review — but it's
the same shape: a checkpoint between Think and Execute.

## A hold that pauses the run

`ask_user` pauses when the *model* asks. For a gated tool the runtime
decides: wrap it with `gate_tool(..., on_refusal="interrupt")` and an
approval store, and a hold (`require_human`) parks the run at the held call
instead of handing the model a refusal. The checkpointer keeps the
conversation, the store keeps the pending approval, and the run waits for a
named person to decide.

```python
from locus.agent import Agent
from locus.control import AuditTrail, ControlPolicy, FileApprovals, gate_tool
from locus.core.events import InterruptEvent
from locus.memory.backends.file import FileCheckpointer

store = FileApprovals("approvals.json")
refund = gate_tool(
    issue_refund,
    policy=ControlPolicy(),   # no verification supplied, so the call is held
    approval=store,
    on_refusal="interrupt",
    trail=AuditTrail(),
)
agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_order, refund],
    checkpointer=FileCheckpointer("checkpoints"),
)

async for event in agent.run("Refund order ord-4821.", thread_id="t1"):
    if isinstance(event, InterruptEvent):
        approval_id = event.metadata["approval_id"]   # the run is parked

# Later, even in a new process that rebuilds the same agent and store:
store.decide(approval_id, "approved", by="alice@example.com")
async for event in agent.resume("approved", thread_id="t1", perform_dangling=True):
    print(event)
```

What it does, and where it stops:

- **A restart is survived only by durable parts.** `FileApprovals` keeps
  approvals in one JSON file; `InMemoryApprovals` lives in this process and
  is gone on restart, so use it for tests and demos. Resuming in a new
  process also needs the checkpointer and the same `thread_id`.
- **`perform_dangling=True` is what performs the action.** It re-invokes the
  held call, which finds the decision. Without it, `resume` folds
  `"approved"` in as the call's text result. Nothing re-invokes the call,
  and the model usually treats it as done, so the approved action does not
  happen.
- **An approved call runs at most once; a policy deny never pauses.** The
  approved call is weighed against the policy again when it is redeemed, so
  a deny such as a spend limit still refuses it, and the approval is
  consumed immediately before the side effect, so a repeated call holds
  again instead of riding an old yes. A call the person denies returns a
  refusal the model reads. A policy deny is refused on the spot — no one
  approves past it.
- **An approval names one call.** Its id is derived from the principal, the
  tool and the arguments, plus `ControlPolicy(version=...)` and
  `gate_tool(approval_context=...)` when set, so a call with different
  arguments waits for its own decision.
- **Anyone named can decide unless you say otherwise.** Pass an
  `ApprovalAuthority` to the store to require roles, a quorum or
  delegations. `FileApprovals` serialises writers within one process only.

This is the pausing alternative to the
[bridge-polling hold](../api/control.md#holding-an-action-for-a-human), where
the held call returns an `approval_id` the agent polls and the run carries on.

## Cancelling a run mid-flight

Three ways to stop a running agent without waiting for the
termination algebra to fire:

1. **Hook raises to short-circuit the loop.** Any hook callback can
   raise to abort the run. Useful for budget guards.

   ```python
   class BudgetGuard(HookProvider):
       async def on_iteration_start(self, iteration: int, state: AgentState) -> None:
           if state.total_tokens_used > 100_000:
               raise RuntimeError(f"token budget exceeded at {state.total_tokens_used} tokens")
   ```

2. **Caller cancels the task.** Standard `asyncio` cancellation:

   ```python
   run = asyncio.create_task(agent.run(prompt))
   # ... later
   run.cancel()
   ```

3. **`agent.cancel()`.** Sets a flag the runner polls between nodes;
   the loop exits at the next safe point with
   `TerminateEvent(reason="cancelled")` (and `result.stop_reason ==
   "cancelled"`). State still flushes to the checkpointer first, so the
   conversation can resume cleanly later.

In the `agent.cancel()` case the loop emits a final
`TerminateEvent(reason="cancelled")` so your downstream observability
gets a clean signal.

## What you don't lose on cancel

Cancelled runs **still persist state** to the checkpointer. The
`thread_id` retains the conversation up to the moment of cancel.
You can resume later with the same thread, inspect the state for
debugging, or branch off a new thread from the partial conversation.

## See also

- [Agent Loop](agent-loop.md) — where cancellation is observed in the
  runner.
- [Hooks](hooks.md) — write custom hooks that raise to abort the run.
- [Conversation Management](conversation-management.md) — how
  `thread_id` resumption works.
- [Human in the loop](https://github.com/oracle/locus/blob/main/examples/notebook_15_human_in_the_loop.py)
  — a full runnable example.
- [Multi-agent + HITL](https://github.com/oracle/locus/blob/main/examples/notebook_27_multiagent_human_in_loop.py)
  — three HITL (human-in-the-loop) patterns in one file (approval gate, human-as-tool,
  long-pause snapshot/resume).
- [Incident response](https://github.com/oracle/locus/blob/main/examples/notebook_77_incident_response.py)
  — `interrupt()` as the page-the-human gate after severity
  classification.
- [Support concession approval](https://github.com/oracle/locus/blob/main/examples/notebook_78_concession_approval.py)
  — three stacked `interrupt()` gates on the top tier.
- [Contract review](https://github.com/oracle/locus/blob/main/examples/notebook_79_contract_review.py)
  — `interrupt()` for human counsel inside a refinement loop.
