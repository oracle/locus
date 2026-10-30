# Conversation management

A Locus agent holds one user's
conversation in `state.messages`. To make that conversation **survive
across requests** — across deploys, restarts, and "I'll come back
tomorrow" gaps — you wire a checkpointer and a `thread_id`. Within a
run, a conversation manager keeps that list inside the model's context
window; the drawing shows where the loop runs it and what each strategy
keeps.

{{ locus_diagram context-strategies }}

## The minimum

{{ locus_model_tabs }}

```python
from locus.agent import Agent
from locus.memory.backends import S3Backend

agent = Agent(
    model="{{ locus_model }}",
    tools=[...],
    checkpointer=S3Backend(
        bucket="locus-threads",
        prefix="<your-prefix>/",
    ),
)

# Day 1
agent.run_sync("Look into the refund request on order ord-4821.", thread_id="ticket-8912-refund")

# Day 2 — same thread_id, conversation continues
agent.run_sync("What did we conclude?",                           thread_id="ticket-8912-refund")
# → "The return on ord-4821 is eligible. Want me to draft the refund for manager approval?"
```

The `thread_id` is the unit of conversation. With `checkpointer=`
alone, state is saved when the run finishes or raises, before the
agent pauses on an interrupt and — for a run with a `thread_id` on a
checkpointer that can delete a single checkpoint, as every backend on
the [Checkpointers](checkpointers.md) page can — at the end of each
loop iteration that executes tool calls, so a process that dies
mid-run keeps its progress up to the last such save; pass
`checkpoint_every_n_iterations=0` to keep only the end-of-turn and
interrupt saves. A run continued with `agent.resume(...)` or
`agent.continue_turn(...)` saves on the same schedule as a fresh run:
after each tool-calling iteration, and when it finishes, raises, or
pauses again. Every fresh `agent.run_sync(..., thread_id=...)` call
rehydrates state before the first Think.

## Threads, not sessions

The SDK uses **thread** as the term — borrowing from chat UIs and
issue trackers — because a single piece of work can have many
simultaneous conversations:

| Thread | Use |
|---|---|
| `ticket-8912-refund` | an open support conversation about one refund request |
| `deploy-2026-07-rollout` | a release thread the deploy operator drives across the week |
| `case-7731-triage` | a security-operations triage conversation for one alert |

A thread is a string. Pick the convention that matches your domain.

## What gets persisted

The checkpointer saves the full `AgentState`:

- **`messages`** — system prompt, every user message, every model
  message, every tool result.
- **`tool_executions`** — the dedup history Execute walks for
  idempotent calls.
- **`iteration`** — the running counter (so termination conditions
  resume correctly).
- **`metadata`** — your application's per-thread state.

Hooks are not called on save or load; saves and loads are published
on the event bus as `memory.checkpoint.saved` /
`memory.checkpoint.loaded` instead (see
[Observability](observability.md#in-process-sse-eventbus)). Custom
application data goes in `metadata`.

## Thread lifecycle

```python
# List all threads in a bucket
threads = await checkpointer.list_threads()
# → ["ticket-8912-refund", "case-7731-triage", ...]

# Inspect one
state = await checkpointer.load("case-7731-triage")
print(len(state.messages), "messages")

# Branch — new thread, copy of an existing one
await checkpointer.copy_thread(
    source_thread_id="case-7731-triage",
    dest_thread_id="case-7731-triage-replay",
)

# Drop
await checkpointer.delete("case-7731-triage-replay")

# Vacuum old threads via lifecycle policy (per backend)
```

For S3-compatible object storage, retention is enforced by the bucket's
lifecycle policy — *not* by locus. Configure an S3 lifecycle rule
(transition/expiration) once at the bucket level and the store handles
the cleanup; `OCIBucketBackend` relies on the equivalent OCI Object
Storage lifecycle policy.

## Concurrent updates to the same thread

Two `agent.run(...)` calls against the same `thread_id` are usually a
bug — you'll race on the checkpoint. Three patterns to avoid that:

1. **Per-case lock at the application layer.** Most chat UIs and
   ticketing systems already serialise messages per session.
2. **Distinct sub-threads.** If the user asks two things in
   parallel, give them two thread ids.
3. **Last-write-wins is the default.** The SDK's checkpointers do not
   currently expose a conflict exception — if you need optimistic
   concurrency, layer it at the application or database level.

## Compaction — keep long threads in budget

After dozens of turns, even the most disciplined conversation
exceeds the model's context window, so an agent manages its context
by default. With no `conversation_manager`, an agent whose model's
window is known (`context_window=`, registered metadata — a built-in
entry, or one you add with `locus.models.metadata.register_metadata` —
or the model itself) gets a `ContextCompactor`: old tool output is
cleared first and, when that is not enough, older turns are summarised
by the agent's own model (or `compaction.summary_model`) and the run
carries on. With `compaction=False` it gets
`LLMCompactor(context_length=<model window>)` with no summariser
instead: stale tool output is pruned and a token-budgeted tail is
kept, with no extra model calls. When the window is unknown, the agent
falls back to `SlidingWindowManager(window_size=max(20, max_iterations * 2))`.
Pass `NullManager()` to turn management off.

`LLMCompactor` is the token-aware built-in `ConversationManager`: it
compacts old turns (and summarises them if you pass `summarize_fn`)
while protecting:

- The system prompt.
- The first N user/assistant turns (the "anchor" of the
  conversation).
- A trailing fraction of recent turns (the context the model needs).

```python
from locus.memory.compactor import LLMCompactor

async def summarise(messages: list, previous_summary: str | None) -> str:
    """Your summarise function — typically a small-model call."""
    ...

# Every argument has a default. Set context_length to your model's window:
# an LLMCompactor you pass yourself does not look it up (default 128_000).
agent = Agent(
    ...,
    conversation_manager=LLMCompactor(
        context_length=128_000,        # the model's context window
        trigger_fraction=0.85,         # compact when usage hits 85% (default 0.8)
        head_turns=2,                  # first 2 turns kept verbatim (default)
        tail_token_fraction=0.4,       # ~40% of budget for recent turns (default 0.5)
        summarize_fn=summarise,        # optional; without it the middle is dropped
    ),
)
```

The compactor runs on the way **into** Think — only when estimated
token usage exceeds `trigger_fraction * context_length`. In short
threads it never fires.

## Retrieving a single thread for a UI

The reference `AgentServer` (`POST /invoke`, `POST /stream`,
`GET /threads/{id}`) reads the thread directly from the checkpointer
and returns the message list — useful for rendering chat history on
page load.

```http
GET /threads/case-7731-triage
# → { "thread_id": "case-7731-triage", "iteration": 9, "messages": [...],
#     "tool_executions": [...], "metadata": {...} }
```

## See also

- [Checkpointers](checkpointers.md) — every backend (the native
  checkpointers, and the rest through `*_checkpointer()` factories)
  and their tradeoffs.
- [Streaming & Server](server.md) — `AgentServer` and SSE.
- [Hooks](hooks.md) — observe and steer the loop between saves; the
  saves and loads themselves reach the
  [event bus](observability.md#in-process-sse-eventbus) as
  `memory.checkpoint.saved` / `memory.checkpoint.loaded`.
