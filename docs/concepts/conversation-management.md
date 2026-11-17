# Conversation management

An agent holds one user's
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
entry, one an OCI id such as `oci:openai.gpt-4.1` resolves through the
model it serves, or one you add with
`locus.models.metadata.register_metadata` — or the model itself) gets a
`ContextCompactor`: old tool output is
cleared first and, when that is not enough, older turns are summarised
by the agent's own model (or `compaction.summary_model`) and the run
carries on. With `compaction=False` it gets
`LLMCompactor(context_length=<model window>)` with no summariser
instead: stale tool output is pruned and a token-budgeted tail is
kept, with no extra model calls. When the window is unknown, the agent
falls back to `SlidingWindowManager(window_size=max(20, max_iterations * 2))`.
Pass `NullManager()` to turn management off. For a run that needs more
stages than clear-then-summarise, see [Tiered compaction](#tiered-compaction)
below.

A model whose provider compacts threads itself can take the summary
stage over: `CompactionConfig(native=True)` (or
`ContextCompactor(native_compaction=True)`) hands the older history to
the model's `compact_history()` and keeps what comes back, which the
model replays on later requests; the `CompactionEvent` then has
`stage="native"`. `OCIResponsesModel` with `store=False` offers this; see
[OCI Responses API](oci-responses.md#native-compaction). A model without
it, or a failed call, summarises as usual.

### Tiered compaction

A long coding or research run does better with a pipeline than with two
stages: drop what is cheap to drop first, and spend a model call only
when that was not enough. `TieredCompaction` runs tiers in order, each
with its own **trigger**, a fraction of the usable window (the model's
window minus the room reserved for the reply). Before each model call
the context is measured; a tier runs only while the context is at or
over its trigger, the context is measured again after it, and a tier
that changes nothing is skipped, so a cheap tier that bought enough room
stops the pipeline before an expensive one runs. The window comes from
the model's metadata entry; `context_length=` names it when the model
has none, and wins when both are given.

| Tier | Default trigger | What it does |
|---|---|---|
| `ClearToolResults` | 0.5 | Tool outputs older than the newest `keep_tokens` become one-line stubs. The newest output is kept whole; an offloaded one keeps its handle; a pinned one is never cleared. |
| `SlidingWindow` | 0.7 | Turns older than the newest `keep_turns` (30) are dropped; a note says how many. |
| `Summarize` | 0.85 | Older history is folded into a model-written summary, by the agent's own model (`summary_model=AGENT_MODEL`) or the one you name; `None` drops it with a note instead. |
| `Relevance` | 0.95 | Only the `keep_turns` (8) turns most relevant to the current task stay, scored by an `embedder=` when you pass one and by recency otherwise. |

`Native` is a fifth tier for a model whose provider compacts threads
itself (`compact_history`, which `OCIResponsesModel` with `store=False`
offers): `TieredCompaction.default_tiers(native=True)` puts it before
`Summarize`, which then runs only when the model has no endpoint.

```python
from locus.agent import Agent
from locus.memory import ClearToolResults, Relevance, Summarize, TieredCompaction

agent = Agent(
    model="{{ locus_model }}",
    tools=[...],
    conversation_manager=TieredCompaction(
        [
            ClearToolResults(trigger=0.5, keep_tokens=20_000),
            Summarize(trigger=0.8),
            Relevance(trigger=0.95, keep_turns=6),
        ]
    ),
)
```

The agent binds the pipeline to its model and window when it is built.
`CompactionConfig(tiers=True)` is the same with the default tiers (and
`native=True` then adds the `Native` tier); `CompactionConfig(tiers=[...])`
names them. A pipeline is exactly the tiers you pass, so to disable a
tier, leave it out. The head of the conversation (the system prompt and
the task), the user's latest request and earlier summaries never leave,
whatever the tier.

Every tier that ran yields its own `CompactionEvent`, with `tier` set and
the tokens before and after it (`agent.context.compacted` on the
[event bus](sse-events.md)). The pipeline reports `exhausted` when the
context is still at or over the highest trigger after the last tier; the
run then ends with `context_exhausted`. A model-calling tier needed again
within `min_iterations_between_summaries` iterations of its last run is
held back for that pass (the last summary did not buy room, so another
would land in the same place) and the tiers after it still run, so a
`Summarize` tier placed before `ClearToolResults` or `SlidingWindow`
cannot end a run those cheaper tiers would have kept inside the window.

### Pinning

A message that must survive compaction — acceptance criteria, a decision
the user took, the handle of a resource — is pinned, by metadata on the
message or by a predicate on the compactor:

```python
from locus.core.messages import Message
from locus.memory import TieredCompaction, pin

criteria = pin(Message.user("ACCEPTANCE: every test green, no new dependencies"))
# the same, written out: Message.user("...", metadata={"pin": True})

compactor = TieredCompaction(pins=lambda m: (m.content or "").startswith("DECISION:"))
```

Pinned messages are never cleared, windowed out, summarised away or
dropped for relevance, in `TieredCompaction` and in `ContextCompactor`
alike. A pinned tool result keeps the call that produced it, so the
conversation stays valid for the provider.

### Offloading large tool results

A test run's log or a 2,000-line file does not need to sit in the
conversation to stay available. `ToolResultOffload` moves any tool result
over `threshold_chars` (16,000) into a `ToolResultStore` the moment it is
produced; the message the model reads keeps a handle and the first
`preview_chars` (2,000), and the `fetch_tool_result` tool, registered
with the offload, reads the stored text back a page at a time. The
handle lives in the message text, so it survives a checkpoint round trip
and every compaction tier (a cleared stub names it): an offloaded result
is referenced, never lost.

```python
from locus.memory import ToolResultOffload

agent = Agent(
    model="{{ locus_model }}",
    tools=[...],
    tool_result_offload=ToolResultOffload(threshold_chars=16_000),  # or True
)
```

With no `store=` the text lives in process memory for the life of the
offload object; pass a `ToolResultStore` over a checkpointer, a bucket or
a database for anything that must outlive the process. The
[coding harness](../harness.md) has the same tool: with a `result_store`
in its `HarnessConfig`, `bash` keeps an output too long to show under a
key and `fetch_tool_result(key)` pages it back.

### `LLMCompactor`

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
