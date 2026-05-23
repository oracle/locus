# Long-term memory

An Oracle Generative AI – Multi-Agent Locus SDK agent is stateless
between sessions by default. Checkpointing preserves the full message
history for one conversation thread, but facts learned in thread A are
invisible in thread B — and when a thread is deleted, everything in it
is gone.

`MemoryManager` fills that gap. It runs two lifecycle hooks
on every agent invocation:

| Hook | When | What |
|---|---|---|
| `on_session_start` | Before the first model call | Retrieve stored memories → inject into system prompt |
| `on_session_end` | After the agent stops | Extract durable facts from the conversation → persist to store |

The result: the agent accumulates knowledge across sessions without the
context window ever filling up with raw history.

## Where memories live

All memories are persisted via a
[`BaseStore`](checkpointers.md#cross-thread-store) backend — the same
store abstraction used for cross-thread key-value storage.  Every
backend that implements `BaseStore` works: `InMemoryStore` for local
development, `RedisBackend` / `PostgreSQLBackend` / `OracleBackend` /
`OpenSearchBackend` / `OCIBucketBackend` for distributed production
workloads.

Storage layout inside the store:

```
(namespace_prefix..., memory_type)  →  key: memory.key  →  value: {...}
```

With the default prefix `("locus_memory",)`:

```
("locus_memory", "user")       →  "role":             {content: "Senior Python engineer"}
("locus_memory", "feedback")   →  "no_db_mocks":      {content: "Never mock the DB. Why: ..."}
("locus_memory", "project")    →  "auth_rewrite":     {content: "Driven by compliance, not TD"}
("locus_memory", "reference")  →  "linear_pipeline":  {content: "Pipeline bugs → Linear INGEST"}
```

Each memory key acts as a stable identifier: re-extracting the same
fact under the same key **updates** the record, not duplicates it.

## Memory types

| Type | What to store | Decays? |
|---|---|---|
| `user` | Role, expertise, working style | Rarely |
| `feedback` | Behavioural rules — what to do/avoid and *why* | Rarely |
| `project` | Goals, deadlines, active decisions | Fast — include a *Why* |
| `reference` | Pointers to external systems (Jira, dashboards, configs) | Medium |

## Quick start

```python
from locus.agent import Agent
from locus.memory.manager import LLMMemoryManager, Memory, MemoryType
from locus.memory.store import InMemoryStore

store = InMemoryStore()   # swap for a persistent backend in production

agent = Agent(
    model="oci:openai.gpt-5.5",
    memory_manager=LLMMemoryManager(store=store),
)

# Session 1 — agent learns that the user dislikes mocking
async for event in agent.run("I prefer real DB connections — never mock the database."):
    ...

# Session 2 — the agent already knows
async for event in agent.run("How should I write the new integration tests?"):
    ...
# → agent uses real DB connections, no explanation needed
```

## Supplying an LLM extraction function

The built-in heuristic (pattern-matching on message text) is adequate
for demos.  For production, pass an async `extract_fn` that calls a
cheap model to identify what is worth remembering:

```python
from locus.memory.manager import LLMMemoryManager, Memory, MemoryType

async def my_extractor(messages: list) -> list[Memory]:
    # Call a fast auxiliary model.
    # The model receives the conversation; it returns structured memory entries.
    raw = await auxiliary_model.complete(
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user",   "content": format_conversation(messages)},
        ]
    )
    return parse_memories(raw.message.content)

manager = LLMMemoryManager(
    store=store,
    extract_fn=my_extractor,
)
```

A minimal extraction prompt:

```
You are a memory extraction assistant. Given a conversation, identify
facts worth remembering across sessions. Return JSON:

[
  {"type": "user",      "key": "role",       "content": "..."},
  {"type": "feedback",  "key": "no_mocks",   "content": "... Why: ... How to apply: ..."},
  {"type": "project",   "key": "auth",       "content": "... Why: ..."},
  {"type": "reference", "key": "linear",     "content": "..."}
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
    namespace_prefix=("tenants", tenant_id, "users", user_id),
)
```

Each combination gets its own set of memories — no cross-contamination.

## Persistent backends

The SDK exposes memory along a spectrum, from "works everywhere" to
"opinionated Oracle-specific":

| Path | Manager class | When to pick it |
|---|---|---|
| Portable / multi-backend | `LLMMemoryManager` + any `BaseStore` (InMemory / Redis / Postgres / OpenSearch / Oracle `OracleStore`) | You need backend portability, an LLM-free extractor, or a test-friendly path |
| Oracle-native | `OracleAgentMemoryManager` (Oracle's official [`oracleagentmemory`](https://pypi.org/project/oracleagentmemory/) client) | You're on Oracle AI Database and want LongMemEval-tuned recall, context cards, and scoped retrieval by `user_id` / `agent_id` / `thread_id` without writing your own extractor |

### Oracle agent memory

`OracleAgentMemoryManager` is a thin adapter that implements the
`BaseMemoryManager` contract on top of `oracleagentmemory.core.OracleAgentMemory`.
You get LLM-backed extraction (94.4 % on LongMemEval / GPT-5.5 xhigh),
prompt-ready context cards, scoped retrieval, and temporal reasoning —
without changing how your `Agent` consumes memory.

```python
import oracledb
from locus.memory.managers import OracleAgentMemoryManager

conn = oracledb.connect(
    user="locus_app",
    password=os.environ["ORACLE_PW"],
    dsn="mydb_low",
    config_dir="~/.oci/wallets/mydb",
    wallet_location="~/.oci/wallets/mydb",
    wallet_password=os.environ.get("ORACLE_WALLET_PW", ""),
)
manager = OracleAgentMemoryManager(
    connection=conn,
    embedder="openai/text-embedding-3-small",  # or any litellm-style model spec
    llm="openai/gpt-4o-mini",                  # or "oci/openai.gpt-4.1" with OCI kwargs
    table_name_prefix="locus_oam_",
)
agent = Agent(model="oci:openai.gpt-4.1", memory_manager=manager)

# Pass user_id (and optional thread_id) via metadata to scope retrieval:
agent.run_sync(
    "Hi, I'm Alice — I prefer us-chicago-1 for inference.",
    metadata={"user_id": "alice", "thread_id": "t-a"},
)
```

Install the optional extras:

```bash
pip install 'locus-sdk[agentmemory]'      # just the agent-memory client
pip install 'locus-sdk[checkpoints]'      # everything Oracle: checkpointers + agent memory
```

The upstream is currently alpha (`oracleagentmemory` 26.4.0, Dev
Status 3). When you need a portable backend or a fully-deterministic
LLM-free extractor, fall back to the portable path below.

See [notebook 13](../notebooks/notebook_09_oracle_agent_memory.md) for
a runnable end-to-end demo with two sessions on the same `user_id`.

### Portable path (any BaseStore backend)

`LLMMemoryManager` works against any `BaseStore` implementation. Use
this for InMemory tests, Redis/Postgres/OpenSearch deployments, or
when you need a deterministic regex-based extractor:

```python
from locus.memory.backends import RedisBackend, PostgreSQLBackend
from locus.memory.store_backends import OracleStore   # Oracle 26ai native

# Redis — distributed, fast (OCI Cache with Redis)
manager = LLMMemoryManager(store=RedisBackend("redis://localhost:6379"))

# PostgreSQL — relational, JSONB metadata ([OCI Database with PostgreSQL][oci-postgres])
manager = LLMMemoryManager(store=PostgreSQLBackend(host="...", database="..."))

# [Oracle 26ai][oracle-db] (`OracleStore`) — namespaced key/value with optional [VECTOR][ai-vector-search]
# search inside a namespace. Native to the SDK, no langchain dep. For
# Oracle workloads, prefer `OracleAgentMemoryManager` above unless
# you specifically want the BaseStore protocol or the LLM-free
# regex extractor.
manager = LLMMemoryManager(
    store=OracleStore(
        dsn="mydb_low",
        user="locus_app",
        password=os.environ["ORACLE_PW"],
        wallet_location="~/.oci/wallets/mydb",
        table_name="locus_store",
        dimension=1024,   # set for semantic-recall memory; omit for plain K/V
    ),
)
```

`OracleStore` is the SDK-native equivalent of
`langgraph-oracledb.OracleStore`: same connection envelope as the
checkpointer (one wallet, one schema), namespaces persist as PK columns,
and `put_with_embedding` + `search_by_embedding` give you semantic
recall against a native `VECTOR(N, FLOAT32)` column.

## What gets injected

At session start, all retrieved memories are formatted as a
`[Long-term Memory]` block and inserted as a system message immediately
after the main system prompt:

```
[System Prompt]
You are a helpful engineering assistant.

[Memory Block — injected by MemoryManager]
[Long-term Memory]
USER [role]: Senior Python engineer, new to React.
FEEDBACK [no_db_mocks]: Never mock the database. Why: prior mock/prod divergence.
PROJECT [auth_rewrite]: Auth rewrite driven by compliance, not tech debt.
REFERENCE [linear_pipeline]: Pipeline bugs tracked in Linear project 'INGEST'.

[Conversation continues...]
```

The main system prompt stays first and intact. The memory block sits
in position 2, visible to the model on its very first call.

## NoopMemoryManager

Use `NoopMemoryManager` as a test double or placeholder:

```python
from locus.memory.manager import NoopMemoryManager

agent = Agent(
    model="oci:openai.gpt-5.5",
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
Retrieval returns at most `retrieve_limit` memories (default 20).
Each memory is a single line in the injected block — typically 50–150
tokens total, regardless of how many sessions have accumulated.

For larger memory sets, plug in a vector-capable `BaseStore` backend
and override `retrieve` to run a semantic similarity search against the
current prompt before injecting:

```python
async def retrieve(self, limit: int = 20) -> list[Memory]:
    query_vec = await embedder.embed(self._current_prompt)
    results = await self.store.search_by_embedding(
        self._ns(MemoryType.FEEDBACK), query_vec, limit=limit
    )
    return [Memory.from_store_value(r.item.value) for r in results]
```

## See also

- [Conversation management](conversation-management.md) — in-session
  context-window management (`SlidingWindowManager`, `LLMCompactor`).
- [Checkpointers](checkpointers.md) — thread-level state persistence
  and the nine native backends.
- [Cross-thread store](checkpointers.md#cross-thread-store) — the
  `BaseStore` interface all memory backends implement.
- [Hooks](hooks.md) — intercept `memory.manager.*` events for custom
  logging or routing.
