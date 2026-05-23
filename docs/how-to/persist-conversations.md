# Persist conversations across restarts

The agent keeps conversation state in `AgentState`. Pass a
`BaseCheckpointer` and the same `thread_id` across invocations to
resume a conversation — even across process restarts.

## 1. Pick a backend

Checkpointers come in two shapes. Knowing which shape you're holding
matters: you pass the **native** ones straight to `Agent`, and you
**wrap** the storage-backed ones through a factory.

**Native checkpointers** (subclasses of `BaseCheckpointer` — pass to
`Agent` directly):

- `MemoryCheckpointer` — in-process dict; tests / REPL
- `FileCheckpointer` — JSON files on disk; single-machine dev
- `HTTPCheckpointer` — talks to a remote checkpoint service you run
- `OCIBucketBackend` — OCI Object Storage; lifecycle policies, region replication

**Storage-backed checkpointers** (wrap a dict-shaped storage with a
factory):

- `redis_checkpointer(...)` — Redis cluster (OCI Cache with Redis)
- `postgresql_checkpointer(...)` — managed Postgres (OCI Database with PostgreSQL)
- `opensearch_checkpointer(...)` — OpenSearch cluster (OCI Search with OpenSearch)
- `oracle_checkpointer(...)` — Oracle Autonomous Database

The native ones are normal classes — `OCIBucketBackend(...)` and
hand it to `Agent`. The storage-backed ones are the underlying
`RedisBackend` / `PostgreSQLBackend` / etc. wrapped by an adapter; if
you instantiate the backend class directly and pass it to `Agent`,
save/load will fail at runtime (the agent calls
`checkpointer.save(state, thread_id)` but backends expose
`save(thread_id, dict)`). Use the matching `*_checkpointer()` factory.

## 2. Instantiate and pass to the Agent

Native checkpointer (no wrapping):

```python
from locus.agent import Agent
from locus.memory.backends import OCIBucketBackend

checkpointer = OCIBucketBackend(
    bucket_name="my-app-checkpoints",
    namespace="my-namespace",
)

agent = Agent(
    model="oci:openai.gpt-5.5",   # any OCI model — see how-to/oci-models.md
    tools=[...],
    checkpointer=checkpointer,
)
```

Storage-backend with the factory:

```python
from locus.memory.backends import postgresql_checkpointer

checkpointer = postgresql_checkpointer(
    dsn="postgresql://locus:locus@db.example.com:5432/locus",
)
agent = Agent(model="oci:openai.gpt-5.5", tools=[...], checkpointer=checkpointer)
```

## 3. Use a stable thread_id

```python
# First turn — new thread
await agent.run("Plan a trip to Paris.", thread_id="user-42").__anext__()

# Second turn, possibly a different process instance
await agent.run("Now book the flights.", thread_id="user-42").__anext__()
```

The agent calls `checkpointer.load(thread_id)` at the start of every
run. If state exists, the new user turn is appended and the run
continues. If not, a fresh state is created.

## 4. Tune the checkpoint cadence

By default the agent writes a checkpoint at the end of every run. For
long runs with expensive tools, also write every N iterations:

```python
agent = Agent(
    ...,
    checkpointer=checkpointer,
    checkpoint_every_n_iterations=5,
)
```

## Testing it works

A brand-new `Agent` instance on the same `thread_id` should see the
prior conversation:

```python
agent1 = Agent(..., checkpointer=checkpointer)
await agent1.run("I'm Alex.", thread_id="t1").__anext__()
del agent1

# Simulates a process restart / different worker.
agent2 = Agent(..., checkpointer=checkpointer)
await agent2.run("Who am I?", thread_id="t1").__anext__()
# The model sees the earlier user turn.
```

Oracle Generative AI – Multi-Agent Locus SDK's integration suite has
this exact test against a live OCI bucket. See
`tests/integration/test_checkpointer_adapters.py`.
