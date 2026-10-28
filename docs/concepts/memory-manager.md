# Long-term memory

A Locus SDK agent is stateless
between sessions by default. Checkpointing preserves the full message
history for one thread, but the pricing exceptions learned handling
customer A are invisible when helping customer B — and when a thread is
deleted, everything the agent learned goes with it.

`MemoryManager` fills that gap. It runs two lifecycle hooks
on every agent invocation:

| Hook | When | What |
|---|---|---|
| `on_session_start` | Before the first model call | Retrieve stored memories → inject into system prompt |
| `on_session_end` | After the agent stops | Extract durable facts from the conversation → persist to store |

The result: a support agent accumulates standing rules, pricing
exceptions, and customer preferences across cases — without the
context window ever filling up with raw history.

## Where memories live

All memories are persisted via a
[`BaseStore`](checkpointers.md#the-built-in-store-inmemorystore) backend
— the same store abstraction used for cross-thread key-value storage.
Three `BaseStore` implementations ship. The built-in `InMemoryStore`
covers local development and tests, but it is process-local and not
durable. `HolographicStore` (SQLite + FTS5, no external infrastructure)
is durable when constructed with a file path —
`HolographicStore(path="memories.db")`; its default `path=":memory:"`
is not. `PgMemory` (Postgres + pgvector, per-tenant row-level security,
needs the `postgresql` extra) is the shared, multi-tenant option. It
treats the first namespace element as the tenant, so put the tenant id
first in `namespace_prefix`, and its `search()` ranks by meaning only
when you pass it an `embedder` — without one, ranking is
lexical/associative token matching. The built-in
`LLMMemoryManager.retrieve()` sends no query, though, so with `PgMemory`
it returns the most recently updated memories and the `embedder` does
not change what gets injected. For relevance-ranked recall, override
`retrieve()` (see [Context bloat vs. recall](#context-bloat-vs-recall))
and call `self.store.search(ns, query=..., limit=limit)`.
`HolographicStore` and `PgMemory` both live in
[`locus.memory.store_backends`](../api/memory.md#durable-store-backends).
For any other backend, subclass `BaseStore` (or use `Mem0MemoryManager`).

Storage layout inside the store:

```
(namespace_prefix..., memory_type)  →  key: memory.key  →  value: {...}
```

With the default prefix `("locus_memory",)`:

| Namespace | Key | `content` of the stored value |
|---|---|---|
| `("locus_memory", "user")` | `"role"` | Tier-2 support agent |
| `("locus_memory", "feedback")` | `"refund_approval"` | Refunds over $500 need manager approval. Why: ... |
| `("locus_memory", "project")` | `"billing_rollout"` | Driven by the new billing rollout, not backlog |
| `("locus_memory", "reference")` | `"orders_api"` | Orders tracked in the 'orders' service API |

The stored value is the whole record: `type`, `key`, `content` and
`metadata`. `save()` stamps `metadata["updated_at"]` on every write, and
`retrieve()` sorts by it, newest first.

Each memory key acts as a stable identifier: re-extracting the same
fact under the same key **updates** the record, not duplicates it.

## Memory types

| Type | What to store | Decays? |
|---|---|---|
| `user` | Operator role, tier, shift, expertise | Rarely |
| `feedback` | Standing rules, playbook steps — what to do/avoid and *why* | Rarely |
| `project` | Active cases, customer-specific exceptions, decisions taken | Fast — include a *Why* |
| `reference` | Pointers to external systems, tuned queries, data feeds, runbooks | Medium |

## Quick start

```python
import asyncio
from locus.agent import Agent
from locus.memory import LLMMemoryManager, Memory, MemoryType
from locus.memory.store import InMemoryStore


async def main():
    store = InMemoryStore()   # swap for a persistent backend in production

    agent = Agent(
        model="{{ locus_example_model }}",
        memory_manager=LLMMemoryManager(store=store),
    )

    # Session 1 — agent learns a standing rule
    async for event in agent.run("Never refund over $500 without manager approval — page the duty manager first."):
        ...

    # Session 2 — the agent already knows
    async for event in agent.run("The customer on order ord-4821 wants a $750 refund. What do you do?"):
        ...
    # → agent requests manager approval before refunding, no reminder needed


asyncio.run(main())
```

## Supplying an LLM extraction function

The built-in heuristic (pattern-matching on message text) is adequate
for demos.  For production, pass an async `extract_fn` that calls a
cheap model to identify what is worth remembering:

```python
from locus.core.messages import Message
from locus.memory import LLMMemoryManager, Memory, MemoryType

async def my_extractor(messages: list[Message]) -> list[Memory]:
    transcript = "\n".join(f"{m.role.value}: {m.content}" for m in messages)
    response = await auxiliary_model.complete(
        messages=[Message.system(EXTRACTION_PROMPT), Message.user(transcript)],
    )
    # Parse the model's JSON into Memory objects; one shown literally:
    return [
        Memory(
            type=MemoryType.FEEDBACK,
            key="refund_approval",
            content="Refunds over $500 need manager approval. Why: prior chargeback.",
        )
    ]

manager = LLMMemoryManager(store=store, extract_fn=my_extractor)
```

A minimal extraction prompt:

```
You are a memory extraction assistant. Given a conversation, identify
facts worth remembering across sessions. Return JSON:

[
  {"type": "user",      "key": "role",           "content": "..."},
  {"type": "feedback",  "key": "refund_approval","content": "... Why: ... How to apply: ..."},
  {"type": "project",   "key": "billing",        "content": "... Why: ..."},
  {"type": "reference", "key": "orders_api",     "content": "..."}
]

Only include facts that are non-obvious, durable, and actionable.
Return [] if nothing is worth remembering.
```

## Scoping per user or tenant

Pass a richer `namespace_prefix` to isolate memories per user, team,
or tenant:

```python
manager = LLMMemoryManager(
    store=shared_store,
    namespace_prefix=(tenant_id, "users", user_id),
)
```

Put the tenant id first. `PgMemory` uses the first namespace element as
its row-level-security tenant, so a fixed first element such as
`"tenants"` would put every tenant behind the same boundary.

Each combination gets its own set of memories — no cross-contamination.

## Persistent backends

The SDK exposes memory along a spectrum, from "works everywhere" to
"fully managed":

| Path | Manager class | When to pick it |
|---|---|---|
| Oracle-native | `OracleAgentMemoryManager` ([`oracleagentmemory`](https://pypi.org/project/oracleagentmemory/)) | You run on Oracle Database 26ai and want Oracle's official agent-memory client — context-card summaries, scoped retrieval by user / agent / thread, LongMemEval-benchmarked extraction |
| Portable / multi-backend | `LLMMemoryManager` + any `BaseStore` (`InMemoryStore`, `HolographicStore`, `PgMemory`, or a custom subclass) | You need backend portability, an LLM-free extractor, or a test-friendly path |
| Managed | `Mem0MemoryManager` ([`mem0ai`](https://pypi.org/project/mem0ai/)) | You want a managed memory layer with LLM-tuned recall and scoped retrieval by `user_id` without writing your own extractor |

### Managed memory — `Mem0MemoryManager`

`Mem0MemoryManager` is a thin adapter that implements the
`BaseMemoryManager` contract on top of [mem0](https://mem0.ai). You get
LLM-backed extraction, prompt-ready context, and scoped retrieval —
without changing how your `Agent` consumes memory.

```python
from locus.memory.managers import Mem0MemoryManager

manager = Mem0MemoryManager(user_id="agent-7")
agent = Agent(model="{{ locus_example_model }}", memory_manager=manager)

# Pass user_id (and optional thread_id) via metadata to scope retrieval:
agent.run_sync(
    "I'm on the night shift — I prefer concise case summaries.",
    metadata={"user_id": "agent-7", "thread_id": "t-a"},
)
```

Install the optional extra:

```bash
pip install 'locus-sdk[mem0]'
```

When you need a fully-deterministic LLM-free extractor or a custom
backend, fall back to the portable path below.

### Oracle Database 26ai: `OracleStore` and `OracleAgentMemoryManager`

Two ways to keep long-term memory in Oracle Database 26ai. `OracleStore` is a
`BaseStore`, so it slots under `LLMMemoryManager` with the portable
extractor and per-user scoping; `OracleAgentMemoryManager` hands the whole
job to Oracle's own `oracleagentmemory` client (extraction, context cards,
retrieval scoped by user, agent and thread, all inside the database). Both
default to OCI Generative AI models for embedding and extraction and take
the credentials from your OCI profile.

```python
import os

from locus.agent import Agent
from locus.memory import LLMMemoryManager
from locus.memory.store_backends import OracleStore

store = OracleStore(
    dsn=os.environ["ORACLE_DSN"],
    user=os.environ["ORACLE_USER"],
    password=os.environ["ORACLE_PASSWORD"],
    wallet_location=os.environ["ORACLE_WALLET"],
    wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
)
agent = Agent(model="{{ locus_example_model }}", memory_manager=LLMMemoryManager(store=store))
```

Notebooks [37](../notebooks/notebook_37_oracle_store.md) and
[44](../notebooks/notebook_44_oracle_agent_memory.md) run both against a
live database; [Oracle Database 26ai](oracle-26ai.md) lists the rest of the
stack.

### Portable path (any BaseStore backend)

`LLMMemoryManager` works against any `BaseStore` implementation. Use
this for `InMemoryStore`, `HolographicStore`, `PgMemory`, or a custom
`BaseStore` subclass over your own backend, or when you need a
deterministic regex-based extractor:

```python
from locus.memory.store import InMemoryStore
from locus.memory.store_backends import HolographicStore

# In-memory — tests, demos, single process
manager = LLMMemoryManager(store=InMemoryStore())

# Durable, no external infrastructure — needs a file path
# (the default path=":memory:" is not durable)
manager = LLMMemoryManager(store=HolographicStore(path="memories.db"))
```

## What gets injected

At session start, all retrieved memories are formatted as a
`[Long-term Memory]` block and inserted as a system message immediately
after the main system prompt:

```
[System Prompt]
You are a customer-support assistant.

[Memory Block — injected by MemoryManager]
[Long-term Memory]
USER [role]: Tier-2 support agent, covers the night shift.
FEEDBACK [refund_approval]: Refunds over $500 need manager approval. Why: prior chargeback from an unreviewed refund.
PROJECT [billing_rollout]: Active billing migration, prioritise invoice disputes.
REFERENCE [orders_api]: Orders tracked in the 'orders' service API.

[Conversation continues...]
```

The main system prompt stays first and intact. The memory block sits
in position 2, visible to the model on its very first call.

## NoopMemoryManager

Use `NoopMemoryManager` as a test double or placeholder:

```python
from locus.memory.manager import NoopMemoryManager

agent = Agent(
    model="{{ locus_example_model }}",
    memory_manager=NoopMemoryManager(),  # wires the hook, stores nothing
)
```

## Writing a custom MemoryManager

Subclass `BaseMemoryManager` and implement three abstract methods:

```python
from locus.memory.manager import BaseMemoryManager, Memory

class MyMemoryManager(BaseMemoryManager):

    async def extract(self, messages: list) -> list[Memory]:
        """Return memories worth keeping from this conversation."""
        ...

    async def retrieve(self, limit: int = 20) -> list[Memory]:
        """Return memories to inject at session start."""
        ...

    async def save(self, memories: list[Memory]) -> None:
        """Persist a list of memories (upsert by key)."""
        ...
```

The base class provides `on_session_start` and `on_session_end`
by default — you don't need to implement them unless you need custom
injection or extraction timing.

## Observability

Two events are emitted on the agent event bus:

| Event | When | Payload |
|---|---|---|
| `memory.manager.injected` | Session start, after memories are injected | `memory_count`, `types` |
| `memory.manager.extracted` | Session end, after memories are saved | `memory_count`, `types`, `keys` |

Subscribe via the standard hook or SSE stream:

```python
from locus.observability.emit import EV_MEMORY_MANAGER_INJECTED, EV_MEMORY_MANAGER_EXTRACTED
```

## Context bloat vs. recall

The memory manager is designed to keep injected context small.
At session start the built-in injection calls `retrieve()` with its
default `limit=20`, so at most 20 memories are injected. Each memory is
a single line in the injected block — typically 50–150 tokens total,
regardless of how many sessions have accumulated.

!!! note
    `LLMMemoryManager` accepts a `retrieve_limit` constructor argument,
    but the built-in `on_session_start` injection calls `retrieve()`
    without passing it — so today `retrieve_limit` does **not** change
    how many memories are injected (the effective cap is the `retrieve()`
    default of 20). To use a different cap, override `retrieve()` as
    shown below and pass your own `limit`.

For larger memory sets, plug in a store whose `search()` ranks by
relevance (`PgMemory` with an `embedder` ranks by meaning) and override
`retrieve` to search against the current prompt before injecting. Pass
the prompt text as `query` to `self.store.search()`: `PgMemory` embeds
it for you, and none of the shipped stores implements
`search_by_embedding()` (the `BaseStore` default raises
`StoreCapabilityError`).

```python
async def retrieve(self, limit: int = 20) -> list[Memory]:
    # self._current_prompt is yours to set, e.g. from the last user message
    # in an on_session_start override before it calls super().
    items = await self.store.search(
        self._ns(MemoryType.FEEDBACK), query=self._current_prompt, limit=limit
    )
    return [Memory.from_store_value(item.value) for item in items]
```

## See also

- [Conversation management](conversation-management.md) — in-session
  context-window management (`SlidingWindowManager`, `LLMCompactor`).
- [Checkpointers](checkpointers.md) — thread-level state persistence
  and the native checkpointer backends.
- [Cross-thread store](checkpointers.md#cross-thread-store) — the
  `BaseStore` interface. Note: the checkpointer backends are a separate
  KV interface and do **not** implement `BaseStore`; for durable
  cross-thread memory use `HolographicStore` or `PgMemory` (see
  [Where memories live](#where-memories-live)), your own `BaseStore`, or
  `Mem0MemoryManager`.
- [Hooks](hooks.md) — intercept `memory.manager.*` events for custom
  logging or routing.
