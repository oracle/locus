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
- `S3Backend` — S3-compatible object storage; lifecycle policies, region replication

**Storage-backed checkpointers** (wrap a dict-shaped storage with a
factory):

- `redis_checkpointer(...)` — Redis cluster (a managed Redis)
- `postgresql_checkpointer(...)` — managed Postgres (managed Postgres)
- `mysql_checkpointer(...)` — MySQL with the official Connector/Python async driver
- `opensearch_checkpointer(...)` — OpenSearch cluster (managed OpenSearch)
- `s3_checkpointer(...)` — an S3-compatible bucket

The native ones are normal classes — `S3Backend(...)` and
hand it to `Agent`. The storage-backed ones are the underlying
`RedisBackend` / `PostgreSQLBackend` / `MySQLBackend` / etc. wrapped by an adapter; if
you instantiate the backend class directly and pass it to `Agent`,
save/load will fail at runtime (the agent calls
`checkpointer.save(state, thread_id)` but backends expose
`save(thread_id, dict)`). Use the matching `*_checkpointer()` factory.

## 2. Instantiate and pass to the Agent

Native checkpointer (no wrapping):

```python
from tulip.agent import Agent
from tulip.memory.backends import S3Backend

checkpointer = S3Backend(
    bucket_name="my-app-checkpoints",
    namespace="my-namespace",
)

agent = Agent(
    model="anthropic:claude-sonnet-4-6",   # any model — see concepts/models.md
    tools=[...],
    checkpointer=checkpointer,
)
```

Storage-backend with the factory:

```python
from tulip.memory.backends import postgresql_checkpointer

checkpointer = postgresql_checkpointer(
    dsn="postgresql://tulip:tulip@db.example.com:5432/tulip",
)
agent = Agent(model="anthropic:claude-sonnet-4-6", tools=[...], checkpointer=checkpointer)
```

MySQL with the official async driver:

```python
from tulip.memory.backends import mysql_checkpointer

checkpointer = mysql_checkpointer(
    dsn="mysql://tulip:tulip@db.example.com:3306/tulip",
)
agent = Agent(model="anthropic:claude-sonnet-4-6", tools=[...], checkpointer=checkpointer)
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

Tulip's integration suite has
this exact test against a live S3 bucket. See
`tests/integration/test_checkpointer_adapters.py`.
