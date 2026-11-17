---
title: Oracle Database 26ai
description: "Every Oracle Database 26ai primitive in the SDK — vector search, in-database chunking and embeddings, table ingestion, durable and versioned threads, long-term memory and Oracle agent memory — with the notebook that runs each one."
---

# Oracle Database 26ai

The SDK treats Oracle Database 26ai as the place where an agent's state lives:
vectors for retrieval, threads for continuity, memories for what it has
learned, in the same database as the data it acts on. Each primitive is a
first-class class directly on the `oracledb` driver, with no
third-party adapter in between (`OracleAgentMemoryManager` alone sits on
Oracle's own agent-memory client), and it works with the same `Agent` as
everything else. The drawing groups them by what the database does for the
agent and names the `Agent` surface each group is reached through.

{{ locus_diagram oracle-26ai-primitives }}

```bash
pip install "locus-sdk[oci,oracledb]"
```

`OracleAgentMemoryManager` additionally needs the `agentmemory` extra
(`pip install "locus-sdk[agentmemory]"`), which installs Oracle's
`oracleagentmemory` client.

## What ships

| Need | Class | Notebook |
|---|---|---|
| Vector search for RAG on native `VECTOR` columns | `OracleVectorStore` | [38 — Oracle 26ai RAG](../notebooks/notebook_38_oracle_26ai_rag.md) |
| Ingest rows from your own tables as documents | `OracleADBLoader` | [39 — ADB loader](../notebooks/notebook_39_oracle_adb_loader.md) |
| Chunk text inside the database | `OracleInDBChunker` | [40 — In-database chunker](../notebooks/notebook_40_oracle_indb_chunker.md) |
| Embed inside the database with an ONNX model | `OracleInDBEmbeddings` | [41 — In-database embeddings](../notebooks/notebook_41_oracle_indb_embeddings.md) |
| Durable conversation threads | `oracle_checkpointer()` | [42 — Oracle 26ai checkpointer](../notebooks/notebook_42_oracle_26ai_checkpointer.md) |
| Every version of a thread, resume from any point | `OracleCheckpointSaver` | [43 — Versioned saver](../notebooks/notebook_43_oracle_versioned_saver.md) |
| Long-term memory store (`BaseStore`) | `OracleStore` | [37 — Oracle store](../notebooks/notebook_37_oracle_store.md) |
| Oracle's agent-memory client: LLM-backed extraction, in-database recall | `OracleAgentMemoryManager` | [44 — Oracle agent memory](../notebooks/notebook_44_oracle_agent_memory.md) |
| Tamper-evident audit trail on an immutable or blockchain table | `OracleAuditTrail` | [Where the audit trail lives](observability.md#where-the-audit-trail-lives) |
| Approval store replicas share, each decision under a row lock | `OracleApprovals` | [Pausing until someone decides](../api/control.md#pausing-until-someone-decides) |

Six of the classes have a synchronous twin (`OracleSyncVectorStore`,
`OracleSyncInDBEmbeddings`, `OracleSyncInDBChunker`, `OracleSyncADBLoader`,
`OracleSyncStore`, `OracleSyncCheckpointSaver`) for code that is not async;
`OracleSyncBackend` is the twin of the `OracleBackend` behind
`oracle_checkpointer()`, and `OracleAgentMemoryManager` has no twin because
it already takes a synchronous `oracledb` pool or connection.

## Connecting

Everything except `OracleAgentMemoryManager` takes the same connection
arguments: `dsn`, `user` and `password`; `config_dir`, the directory
holding `tnsnames.ora`, when `dsn` is an alias; for Autonomous Database
with mutual TLS, `wallet_location` and `wallet_password`; or `host`, `port`
and `service_name` for a plain listener. `OracleAgentMemoryManager` instead
takes `connection=`, a ready `oracledb` pool (preferred) or connection built
from the same values. The notebooks and the integration tests read these
values from `ORACLE_DSN`, `ORACLE_USER`, `ORACLE_PASSWORD`,
`ORACLE_CONFIG_DIR` and, for mutual TLS, `ORACLE_WALLET` and
`ORACLE_WALLET_PASSWORD`.

### TLS-only or mutual TLS

An Autonomous Database created without mutual TLS (`is_mtls_connection_required
= false`, the default for a private endpoint) has no wallet. The client checks
the server certificate against the public CAs, so `dsn`, `user` and `password`
connect on their own when `dsn` is the connect descriptor from the console
(`(description=...(protocol=tcps)...)`) or an Easy Connect string. To use a
`tnsnames.ora` alias instead, pass the directory holding the file as
`config_dir` — or set `TNS_ADMIN` and pass nothing. Do not pass that directory
as `wallet_location`: python-oracledb then looks for a wallet file in it and
fails with `DPY-6005`.

With mutual TLS, `wallet_location` is the unzipped wallet directory. It holds
its own `tnsnames.ora`, so it resolves aliases too; `config_dir` is only needed
when the aliases live elsewhere.

```python
import os

from locus.memory.store_backends import OracleStore

credentials = dict(user=os.environ["ORACLE_USER"], password=os.environ["ORACLE_PASSWORD"])

# TLS-only: the connect descriptor from the console, no wallet
store = OracleStore(dsn=os.environ["ORACLE_DSN"], **credentials)

# TLS-only: an alias in a tnsnames.ora of your own
store = OracleStore(dsn="mydb_low", config_dir="/opt/oracle/tns", **credentials)

# Mutual TLS: the wallet directory resolves the alias and holds the wallet
store = OracleStore(dsn="mydb_low", wallet_location="/opt/oracle/wallet", **credentials)
```

An `oracle://` [approval-store](../api/control.md#pausing-until-someone-decides)
or [report-store](evaluation.md#keeping-reports-and-reading-the-trend) spec takes `config_dir=` and
`wallet=` the same way, and a bare one reads `ORACLE_CONFIG_DIR` (else
`TNS_ADMIN`) and `ORACLE_WALLET` from the environment.

## Retrieval

The agent's own documents on a `VECTOR` column, searched with
`VECTOR_DISTANCE`, exposed to the model as a tool.

```python
import os

from locus.agent import Agent
from locus.rag import OCIEmbeddings, OracleVectorStore, RAGRetriever

db = dict(
    dsn=os.environ["ORACLE_DSN"],
    user=os.environ["ORACLE_USER"],
    password=os.environ["ORACLE_PASSWORD"],
    wallet_location=os.environ["ORACLE_WALLET"],  # mutual TLS; omit for a TLS-only database
    wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
)
retriever = RAGRetriever(
    embedder=OCIEmbeddings(model_id="cohere.embed-v4.0"),
    store=OracleVectorStore(table_name="runbooks", dimension=1536, **db),
)
agent = Agent(model="oci:openai.gpt-5.5", tools=[retriever.as_tool()])
```

When the text must not leave the database, chunk and embed inside it:
`OracleInDBChunker` splits with `DBMS_VECTOR_CHAIN`, `OracleInDBEmbeddings`
runs an ONNX model loaded into the database, and `OracleVectorStore` keeps
the result, so no document content goes to a hosted embedding API. The
model is a one-time load. On Autonomous Database, Oracle's prebuilt
all-MiniLM-L12-v2 (384 dimensions) loads directly from Object Storage with
[`DBMS_VECTOR.LOAD_ONNX_MODEL_CLOUD`](https://docs.oracle.com/en/database/oracle/oracle-database/26/arpls/dbms_vector1.html);
the download URL is in the Oracle Machine Learning post
[Pre-built Embedding Generation model for Oracle Database 26ai](https://blogs.oracle.com/machinelearning/post/use-our-prebuilt-onnx-model-now-available-for-embedding-generation-in-oracle-database-23ai):

```sql
BEGIN
  DBMS_VECTOR.LOAD_ONNX_MODEL_CLOUD(
    model_name => 'ALL_MINILM_L12_V2',
    credential => NULL,   -- pre-authenticated URL
    uri        => '<URL of all_MiniLM_L12_v2.onnx from the Oracle Machine Learning blog>');
END;
/
SELECT vector_dimension_count(
  vector_embedding(ALL_MINILM_L12_V2 USING 'hello' AS data)) FROM dual;  -- 384
```

Then `OracleInDBEmbeddings(model_name="ALL_MINILM_L12_V2", dimension=384)`
and `OracleVectorStore(dimension=384)`; notebook 41 runs the whole path.

## Threads and memory

`oracle_checkpointer()` gives every thread a durable home;
`OracleCheckpointSaver` keeps the full history of each thread; `OracleStore`
and `OracleAgentMemoryManager` hold what the agent learns across threads.
[Memory: short-term and long-term](memory.md) explains the three layers
and shows each one on Oracle Database 26ai.

## Where to go next

- [OCI Generative AI](providers/oci.md) for the model side of the same stack.
- [Dedicated AI Clusters](../how-to/oci-dac.md) for your own model endpoints.
- [RAG](rag.md), [Checkpointers](checkpointers.md) and [Long-term memory manager](memory-manager.md) for the general surfaces these classes plug into.
