# Memory

Three orthogonal layers, each with its own contract:

1. **Conversation management** — what to do with the message history
   *within* a single run (keep, window, summarize) when it grows past
   the model's context window.
2. **Cross-thread store** — durable key-value scoped by namespace, used
   for long-term memory entries the agent reads at session start and
   writes at session end.
3. **Long-term memory manager** — the policy layer that decides *what*
   to extract from a run and *when* to retrieve, sitting on top of any
   `BaseStore` backend.

For checkpointing (state persistence between runs), see
[Checkpointers](checkpointers.md) — those backends are also in
`locus.memory.backends`, with **Oracle ADB** as the production target.

## Conversation management

::: locus.memory.conversation.ConversationManager
::: locus.memory.conversation.NullManager
::: locus.memory.conversation.SlidingWindowManager
::: locus.memory.conversation.SummarizingManager

## Cross-thread store

The durable KV the long-term memory manager writes to. `InMemoryStore`
is for tests; production stores live in `locus.memory.backends` (see
[Checkpointers](checkpointers.md) for Oracle / OCI Object Storage /
OpenSearch / Postgres / Redis backends, which all implement
`BaseStore` as well as `BaseCheckpointer`).

::: locus.memory.store.BaseStore
::: locus.memory.store.InMemoryStore
::: locus.memory.store.NamespacedStore
::: locus.memory.store.StoreContext
::: locus.memory.store.StoreItem
::: locus.memory.store.StoreCapabilities
::: locus.memory.store.StoreCapabilityError
::: locus.memory.store.SemanticSearchResult

## Long-term memory manager

`LLMMemoryManager` is the default — it uses an auxiliary model to
extract and categorize memories at session end and retrieves the
top-k relevant entries at session start. `NoopMemoryManager` is the
pass-through used in tests.

::: locus.memory.manager.BaseMemoryManager
::: locus.memory.manager.LLMMemoryManager
::: locus.memory.manager.NoopMemoryManager
::: locus.memory.manager.Memory
::: locus.memory.manager.MemoryType

## Delta checkpointing

Storage-efficient checkpointer that persists only the diff between
consecutive states (~77% storage savings on long conversations).
Layered on top of any `DeltaStorage` backend.

::: locus.memory.delta.DeltaCheckpointer
::: locus.memory.delta.DeltaCheckpoint
::: locus.memory.delta.CheckpointMetadata
::: locus.memory.delta.DeltaStorage
::: locus.memory.delta.InMemoryDeltaStorage

## Registry

String-based checkpointer lookup — used when configuration passes a
provider name (e.g. `"oracle"`, `"redis"`) instead of an instance.
Custom backends register themselves via `register_checkpointer`.

::: locus.memory.registry.get_checkpointer
::: locus.memory.registry.register_checkpointer
::: locus.memory.registry.list_checkpointers
