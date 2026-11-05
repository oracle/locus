# Oracle agent memory

An Oracle-backed memory manager for an `Agent`. The example wires
the SDK's [`OracleAgentMemoryManager`](../api/memory.md) to
[`oracleagentmemory`](https://pypi.org/project/oracleagentmemory/),
Oracle's client for AI-agent memory held in Oracle Database 26ai, and
shows a fact learned in one thread being recalled by a new agent in
another.

Compared with [notebook 37's](notebook_37_oracle_store.md)
`LLMMemoryManager + OracleStore`, the portable path that works against
any `BaseStore` (InMemory, Redis, Postgres, OpenSearch, Oracle), this
manager brings in:

- LLM-backed memory extraction, in place of notebook 37's regex-based
  extractor; the upstream README reports LongMemEval results.
- Prompt-ready context cards that combine a summary of the thread with
  the most relevant retrieved records.
- Retrieval scoped by `user_id`, `agent_id` and `thread_id`, so
  memories carry across threads without extra wiring.
- Time-to-live retention: a store built with `MemoryRetentionConfig`
  gives records an expiry.

The trade-off is that it is Oracle-only; the SDK pins
`oracleagentmemory>=26.4.0,<27`. When you need a portable backend or a
deterministic, LLM-free extractor, use notebook 37.

What this notebook covers:

- `OracleAgentMemoryManager(connection=..., embedder=..., llm=...)`,
  the adapter that implements the SDK's `BaseMemoryManager` contract on
  top of `oracleagentmemory.core.OracleAgentMemory`. Pass it as
  `Agent(memory_manager=...)`.
- A two-session demo on the same `user_id` ("alice") but different
  `thread_id`s: session A teaches a fact, and session B's brand-new
  agent recalls it through the memory block the manager inserts at
  session start, which holds the thread's context card and the results
  of a search across threads scoped to the user.
- A side-channel inspection between the two sessions showing which
  records the LLM extractor produced inside Oracle Database 26ai.

## Prerequisites

Optional dependency; install the extras group:

```bash
pip install 'locus-sdk[agentmemory]'
# or the bundle that covers every Oracle backend:
pip install 'locus-sdk[checkpoints]'
```

Same Oracle wallet block as the rest of the Oracle Cloud and Database
26ai section:

```bash
export ORACLE_DSN=mydb_low                   # tnsnames alias
export ORACLE_USER=locus_app                 # least-privileged app schema
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if encrypted
```

`oracleagentmemory` drives its embedder and extraction model through
litellm. The demo defaults to OCI Generative AI (`oci/cohere.embed-v4.0`
for embeddings, `oci/openai.gpt-4.1` for extraction); litellm's OCI
driver does not read `~/.oci/config` itself, so the manager builds the
signer from your OCI profile, API key or `oci session authenticate`
token, and passes it along with the region and compartment
(`OCI_AUTH_TYPE=instance_principal` or `resource_principal` selects a
workload identity instead):

```bash
export OCI_PROFILE=DEFAULT                    # or OCI_SESSION_PROFILE=...
export OCI_REGION=eu-frankfurt-1
export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx
```

Any other litellm model spec works too, with that provider's own
credentials:

```bash
export LOCUS_OAM_EMBEDDER=openai/text-embedding-3-small
export LOCUS_OAM_LLM=openai/gpt-4o-mini
export OPENAI_API_KEY=sk-...
```

A memory manager is the third of the four memory layers in the drawing,
long-term memory that lives until you delete it: `OracleAgentMemoryManager`,
passed as `memory_manager=`, inserts Alice's memories at the start of a run
and hands the transcript to the extractor after it ends, in the same
Oracle Database 26ai that can hold the thread checkpoints and the
retrieval-augmented generation (RAG) corpus.

{{ locus_diagram memory-layers }}

## Key lines

The manager takes a synchronous `oracledb` pool rather than a single
connection: `oracleagentmemory` runs memory extraction on a background
thread, and a connection shared between that thread and the agent's
own queries is not thread-safe, so the pool hands each caller its own
session:

{{ locus_source examples/notebook_44_oracle_agent_memory.py::_build_connection }}

`schema_policy="create_if_necessary"` creates the missing schema objects
on first run; a production deployment uses `"require_existing"`, which
validates the existing schema and creates or modifies nothing, so the
schema is provisioned out of band and the runtime user needs no
privilege beyond using it. `thread_defaults` is forwarded to
`create_thread` for every thread the manager creates, here running
extraction after every appended message with a running context summary,
so Alice's preference is extracted from session A's single turn
(upstream deprecated these per-thread keywords in 26.6 in favour of
`memory_extraction_config`; the pinned `<27` releases still accept
them):

{{ locus_source examples/notebook_44_oracle_agent_memory.py::_build_manager }}

A new agent on thread B, built by `_build_agent` (full source below) so
that `memory_manager=manager` is its only link to session A, asks for
Alice's preference although the thread has no prior messages. Before
the model is called, `OracleAgentMemoryManager.on_session_start` has
inserted a system message right after the user's turn holding thread
B's context card from `OracleThread.get_context_card_async`, when it
renders anything (recent messages and a summary of this thread, both
empty for a new thread), and the records that
`OracleAgentMemory.search_async` scoped to `user_id` returns from every
thread; the second part is how the preference extracted on thread A
reaches the model:

{{ locus_source examples/notebook_44_oracle_agent_memory.py::session_b }}

## Run it

```bash
python examples/notebook_44_oracle_agent_memory.py
```

The script needs an Autonomous Database reachable through the four
variables in Prerequisites: `ORACLE_DSN`, `ORACLE_USER`,
`ORACLE_PASSWORD` and `ORACLE_WALLET` (`ORACLE_WALLET_PASSWORD` only
when the wallet is encrypted), plus the `oracleagentmemory` extra. When
any of the four variables is unset it prints the missing names and the
minimal wiring snippet, then exits cleanly without opening a
connection; when the package is not installed, the manager raises an
`ImportError` that names the extras command. With the database in
place, the embedder and the memory extractor run through litellm
against OCI Generative AI by default, authenticated from the OCI
profile named by `OCI_SESSION_PROFILE`, `OCI_PROFILE` or
`OCI_CLI_PROFILE`, in that order (`DEFAULT` otherwise), with
`OCI_REGION` (the profile's region otherwise) and `OCI_COMPARTMENT_ID`;
`LOCUS_OAM_EMBEDDER` and `LOCUS_OAM_LLM` switch them to another litellm
spec. The agent's own model comes from `examples/config.py`: OCI
Generative AI when `~/.oci/config` has a usable profile, otherwise the
bundled mock model; set `LOCUS_MODEL_PROVIDER` (`oci`, `openai`,
`anthropic` or `mock`) to force one.

## See also

- [Oracle 26ai cross-thread memory agent (portable `BaseStore` path)](notebook_37_oracle_store.md)
- [Oracle 26ai checkpointer](notebook_42_oracle_26ai_checkpointer.md)
- [Concepts — Long-term memory](../concepts/memory-manager.md#oracle-database-26ai-oraclestore-and-oracleagentmemorymanager)
- [Concepts — Oracle Database 26ai](../concepts/oracle-26ai.md)
- [`oracleagentmemory` on PyPI](https://pypi.org/project/oracleagentmemory/)
- [Oracle AI Agent Memory — documentation](https://docs.oracle.com/en/database/oracle/agent-memory/index.html)

## Source

??? note "Full source — `examples/notebook_44_oracle_agent_memory.py`"

    ````python
    --8<-- "examples/notebook_44_oracle_agent_memory.py"
    ````
