# Interrupts & human-in-the-loop

Sometimes the agent shouldn't decide alone. A human approves the
$2M PO. A reviewer signs off on the customer refund. A regulator
requires an audit checkpoint between research and submission.

Tulip treats human approval as
**a tool the model can call** — same shape as any other tool, except
it surfaces a question to your app and resumes when the human
responds.

## The shape

```python
from tulip.agent import Agent
from tulip.tools.decorator import tool

@tool
def request_human_approval(reason: str, action: str) -> dict:
    """Pause the run for human approval. The runner pauses until
    your app calls agent.resume(response=...)."""
    raise PendingApproval(reason=reason, action=action)

@tool(idempotent=True)
def submit_po(vendor_id: str, amount_usd: float) -> dict:
    return finance.submit(vendor_id, amount_usd)

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_vendors, request_human_approval, submit_po],
    system_prompt=(
        "You are a procurement officer. "
        "Always call request_human_approval before submit_po."
    ),
)
```

`PendingApproval` is your own sentinel exception. When the agent
calls the tool, the SDK catches the exception, persists state to the
checkpointer, and exits with `TerminateEvent(reason="PendingApproval")`.
Your app reads the reason out of `state.metadata` and asks the human.

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

For long-running workflows, the agent persists state and exits when
the approval tool raises `PendingApproval`. A separate process
(browser, Slack action, email link) eventually calls:

```python
await agent.resume(response="approved")
```

The loop rehydrates from the checkpointer, threads the response into
the next Think, and continues.

### Steering — a second model votes

Not strictly human-in-the-loop, but lives in the same family. The
`SteeringHook` runs an LLM-as-judge on every tool call before it
fires:

```python
from tulip.hooks.builtin.steering import SteeringHook

agent = Agent(
    ...,
    hooks=[SteeringHook(
        judge_model="anthropic:claude-sonnet-4-6",
        policy="Reject any tool call that doesn't match the user's stated request.",
    )],
)
```

When the judge votes "no", the call is rejected and the agent
re-plans. This is policy enforcement, not human review — but it's
the same shape: a checkpoint between Think and Execute.

## Cancelling a run mid-flight

Three ways to stop a running agent without waiting for the
termination algebra to fire:

1. **Hook raises to short-circuit the loop.** Any hook callback can
   raise to abort the run. Useful for budget guards.

   ```python
   class BudgetGuard(HookProvider):
       async def on_iteration_start(
           self, iteration: int, state: AgentState
       ) -> None:
           if state.total_tokens_used > 100_000:
               raise RuntimeError("token budget exceeded")
   ```

2. **Caller cancels the task.** Standard `asyncio` cancellation:

   ```python
   run = asyncio.create_task(agent.run(prompt))
   # ... later
   run.cancel()
   ```

3. **`agent.cancel()`.** Sets a flag the runner polls between nodes;
   the loop exits at the next safe point with
   `TerminateEvent(reason="Cancelled")`. State still flushes to the
   checkpointer first, so the conversation can resume cleanly later.

In all three cases the loop emits a final
`TerminateEvent(reason="Cancelled: …")` so your downstream
observability gets a clean signal.

## What you don't lose on cancel

Cancelled runs **still persist state** to the checkpointer. The
`thread_id` retains the conversation up to the moment of cancel.
You can resume later with the same thread, inspect the state for
debugging, or branch off a new thread from the partial conversation.

## See also

- [Agent Loop](agent-loop.md) — where Cancel directives are
  observed in the runner.
- [Hooks](hooks.md) — write custom hooks that return `Cancel`.
- [Conversation Management](conversation-management.md) — how
  `thread_id` resumption works.
- [Notebook 09 — human in the loop](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_19_human_in_the_loop.py)
  — a full runnable example.
- [Notebook 46 — multi-agent + HITL](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_33_multiagent_human_in_loop.py)
  — three HITL patterns in one file (approval gate, human-as-tool,
  long-pause snapshot/resume).
- [Notebook 47 — incident response](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_63_incident_response.py)
  — `interrupt()` as the page-the-human gate after severity
  classification.
- [Notebook 48 — procurement approval](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_64_procurement_approval.py)
  — three stacked `interrupt()` gates on the top tier.
- [Notebook 49 — contract review](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_65_contract_review.py)
  — `interrupt()` for human counsel inside a refinement loop.
