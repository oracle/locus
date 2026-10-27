# Oracle agent memory

An Oracle-backed memory manager for a Locus `Agent`.
Wires Locus's [`OracleAgentMemoryManager`](../api/memory.md) to
[`oracleagentmemory`](https://pypi.org/project/oracleagentmemory/) —
Oracle's official AI-agent-memory client backed by Oracle AI Database.

Compared with [notebook 11's](notebook_10_oracle_store.md)
`LLMMemoryManager + OracleStore` (which is the portable path that
works against any `BaseStore` — InMemory, Redis, Postgres,
OpenSearch, Oracle), this notebook brings in:

- **LLM-backed memory extraction** tuned on LongMemEval — 94.4 %
  recall on GPT-5.5 xhigh, vs. notebook 11's regex-based extractor.
- **Prompt-ready context cards** that combine a summary of the
  current thread + the most relevant retrieved memories.
- **Scoped retrieval** by `user_id` / `agent_id` / `thread_id` —
  cross-thread memory injection just works.
- **Temporal reasoning + automatic expiry** for stale facts.

The trade-off: it's Oracle-only and the upstream is currently alpha
(`oracleagentmemory` 26.4.0, Dev Status 3). When you need a portable
backend or a fully-deterministic LLM-free extractor, fall back to
notebook 11.

## What this covers

- `OracleAgentMemoryManager(connection=..., embedder=..., llm=...)` —
  the adapter that implements Locus's `BaseMemoryManager` contract on
  top of `oracleagentmemory.core.OracleAgentMemory`. Drop-in for
  `Agent(memory_manager=...)`.
- A two-session demo on the same `user_id` ("alice") but different
  `thread_id`s — session A teaches a fact, session B's brand-new agent
  recalls it via a cross-thread search injected into the system prompt.
- A side-channel inspection between the two sessions showing exactly
  what records the LLM extractor produced inside Oracle AI Database.

## Prerequisites

Optional dependency — install the extras group:

```bash
pip install 'locus-sdk[agentmemory]'
# or the bundle that covers every Oracle backend:
pip install 'locus-sdk[checkpoints]'
```

Same Oracle wallet block as the rest of the Oracle 26ai section:

```bash
export ORACLE_DSN=mydb_low                   # tnsnames alias
export ORACLE_USER=locus_app                 # least-privileged app schema
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if encrypted
```

For the embedder + LLM the demo uses OpenAI directly (simplest
credential to set up):

```bash
export OPENAI_API_KEY=sk-...
```

For OCI Generative AI instead, pass the OCI fields as default kwargs
on the `Embedder` / `Llm` (litellm's OCI driver does **not**
auto-discover them from `~/.oci/config`):

```python
from oracleagentmemory.core.embedders import Embedder
from oracleagentmemory.core.llms import Llm

embedder = Embedder(
    model="oci/cohere.embed-v4.0",
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1..xxx",
    oci_tenancy="ocid1.tenancy.oc1..xxx",
    oci_user="ocid1.user.oc1..xxx",
    oci_fingerprint="xx:xx:...",
    oci_key_file="~/.oci/api_key.pem",
)
manager = OracleAgentMemoryManager(connection=conn, embedder=embedder, llm=..., ...)
```

If `ORACLE_DSN` / `ORACLE_PASSWORD` aren't set the notebook prints the
wiring snippet and exits cleanly — no traceback, no half-initialised
state. Same goes for `oracleagentmemory` not being installed: the
manager raises a clear `ImportError` pointing at the extras command.

## Run

```bash
python examples/notebook_88_oracle_agent_memory.py
```

## Schema hygiene

The notebook uses `table_name_prefix="locus_notebook_13_oam_"` and
`schema_policy="create_if_necessary"` so the demo provisions its own
tables on first run. For production, set `schema_policy="managed"`
and create the schema out-of-band with a DBA-privileged account so
the runtime user can run with DML-only privileges.

## See also

- [Notebook 11 — Oracle 26ai cross-thread memory agent (portable BaseStore path)](notebook_10_oracle_store.md)
- [Notebook 07 — Oracle 26ai checkpointer](notebook_53_oracle_26ai_checkpointer.md)
- [Concepts → Memory manager](../concepts/memory-manager.md)
- [`oracleagentmemory` on PyPI](https://pypi.org/project/oracleagentmemory/)
- [Oracle AI Agent Memory — official docs](https://docs.oracle.com/en/database/oracle/agent-memory/index.html)

## Source

```python
--8<-- "examples/notebook_88_oracle_agent_memory.py"
```
