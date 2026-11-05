# Oracle 26ai RAG agent

An `Agent` that answers questions with retrieval-augmented generation
(RAG) over Oracle Database 26ai. The subject is the agent runtime;
Oracle Database 26ai is the storage layer. Embeddings live in a native
`VECTOR(1536, FLOAT32)` column, similarity is the `VECTOR_DISTANCE` SQL
function, and on first use the store creates the table together with a
Hierarchical Navigable Small World (HNSW) vector index, `CREATE VECTOR
INDEX ... ORGANIZATION INMEMORY NEIGHBOR GRAPH WITH DISTANCE COSINE`, the
`OracleVectorStore` default. What the notebook shows is the agent deciding
when to search, issuing a real tool call, and grounding its reply in what
came back.

What this notebook covers:

- `OCIEmbeddings` producing 1536-dimension Cohere `embed-v4.0` vectors on
  OCI Generative AI.
- `OracleVectorStore` opening an async connection pool through an
  Autonomous Database wallet, creating the `locus_notebook_38` table with
  a native `VECTOR(1536, FLOAT32)` column and its HNSW index on first use,
  and serving cosine similarity queries with `VECTOR_DISTANCE`.
- `create_rag_tool(retriever, ...)` turning the `RAGRetriever` into a
  `@tool` that the `Agent` calls on its own. Each retrieval surfaces
  as a `ToolStartEvent` / `ToolCompleteEvent` pair, so the agent's
  decision to search is visible in the event stream.
- The same retriever, tool and agent loop over any backend: swap
  `OracleVectorStore` for `OpenSearchVectorStore`, `PgVectorStore` or
  `InMemoryVectorStore` and only the store constructor changes.

## Prerequisites

```bash
# Database side — Autonomous Database wallet (TLS) + credentials.
export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
export ORACLE_USER=locus_app                 # least-privileged app schema
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if wallet is encrypted

# Embedding side — OCI Generative AI in us-chicago-1.
export OCI_PROFILE=<your-profile>
export OCI_REGION=us-chicago-1
export OCI_COMPARTMENT=<compartment-ocid>
export OCI_AUTH_TYPE=api_key                 # or security_token
```

The notebook relies on the store's default `auto_create_table=True`, so
the first run provisions the `locus_notebook_38` table, its HNSW vector
index and a `created_at` index in the connecting user's schema, which
needs data definition language (DDL) privileges there. For production,
create the table and index out-of-band as the schema owner and pass
`auto_create_table=False`, so the runtime user needs only data
manipulation language (DML) privileges on the table. The
`CREATE USER locus_app`, `CREATE TABLE` and `CREATE VECTOR INDEX`
statements are in the
[`OracleVectorStore` reference](../api/rag.md#locus.rag.stores.oracle.OracleVectorStore);
for this notebook's embedder the column is `VECTOR(1536, FLOAT32)`, and
the reference's index example uses the Inverted File (IVF) organisation,
`ORGANIZATION NEIGHBOR PARTITIONS`, rather than the HNSW default; the
store issues the same `VECTOR_DISTANCE` query against either.

Both lanes of the pipeline run here against one `OCIEmbeddings` embedder
and one `OracleVectorStore`: seeding takes the ingest lane, embedding each
of the six passages with `embed_documents()` and writing it to the
`VECTOR(1536, FLOAT32)` table with `add_batch()`, and each question takes
the retrieve lane, where the agent calls the search tool that
`create_rag_tool(retriever)` built, the retriever embeds the query with
the same embedder's `embed_query()`, Oracle Database 26ai ranks the rows
with `VECTOR_DISTANCE` and the hits return to the agent with their
content, score, metadata and id (no reranker is configured, so the ranked
hits pass straight through):

{{ locus_diagram rag-pipeline }}

## Key lines

The wiring sits in `main()`: `OracleVectorStore(dsn=..., user=...,
password=..., wallet_location=..., wallet_password=...,
table_name="locus_notebook_38", dimension=1536, distance_metric="COSINE")`
is paired with `OCIEmbeddings(model_id="cohere.embed-v4.0", ...)` in a
`RAGRetriever`, and `create_rag_tool(retriever,
name="search_oracle_26ai_corpus", limit=3, threshold=0.0)` wraps
`retriever.retrieve()` as the single tool the `Agent` receives, with a
system prompt that tells the model to search before answering and
`max_iterations=4`.

Seeding is the ingest lane in one call: `RAGRetriever.add_documents`
embeds each passage with `OCIEmbeddings.embed_documents` and writes it,
vector included, through `OracleVectorStore.add_batch`, which creates the
table and index on first use:

{{ locus_source examples/notebook_38_oracle_26ai_rag.py::_seed_corpus }}

Each question goes through `agent.run`, and the loop reads the event
stream rather than the retriever: a `tool_start` event is the agent
deciding to call `search_oracle_26ai_corpus` with the query it chose, the
matching `tool_complete` event carries the tool's return value as a JSON
string in `result` (its `results` list holds the hits, each with
`content`, `score`, `metadata` and `id`, decoded here to print each
passage with its score), and `terminate` carries the grounded final answer
in `final_message`:

{{ locus_source examples/notebook_38_oracle_26ai_rag.py::_drive_agent }}

## Run it

```bash
python examples/notebook_38_oracle_26ai_rag.py
```

The script needs two live services. Oracle Database 26ai is reached
through an Autonomous Database wallet: `ORACLE_DSN`, `ORACLE_USER`,
`ORACLE_PASSWORD` and `ORACLE_WALLET` from Prerequisites, plus
`ORACLE_WALLET_PASSWORD` when the wallet is encrypted. The embeddings come
from Cohere `embed-v4.0` on OCI Generative AI, which needs
`OCI_COMPARTMENT`; `OCI_PROFILE`, `OCI_REGION` and `OCI_AUTH_TYPE` default
to `DEFAULT`, `us-chicago-1` and `api_key`, and the `LOCUS_OCI_*`
equivalents take precedence when set. If any of the five required
variables is missing the script prints a skip notice naming them, together
with the minimal wiring snippet, and exits cleanly without opening a
connection.

The agent's chat model is the one piece that comes from
`examples/config.py`: OCI Generative AI when `~/.oci/config` has a usable
profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one. Under the mock model
each question still produces one synthesised call to
`search_oracle_26ai_corpus`, with a one-word query taken from the prompt,
so the `VECTOR_DISTANCE` search does run in the database; the closing
answer is then placeholder text rather than a reading of the returned
passages.

## See also

- [Concepts — RAG](../concepts/rag.md)
- [Concepts — Oracle Database 26ai](../concepts/oracle-26ai.md)
- [Concepts — Events: `ToolStartEvent` and `ToolCompleteEvent`](../concepts/events.md#toolstartevent)
- [Notebook 45 — Cohere reranker on Oracle 26ai](notebook_45_cohere_reranker.md)
- [Oracle Database 26ai — documentation](https://docs.oracle.com/en/database/oracle/oracle-database/26/index.html)
- [Oracle AI Database 26ai — AI Vector Search User's Guide](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/index.html)
- [Oracle Autonomous Database — documentation](https://docs.oracle.com/en/cloud/paas/autonomous-database/index.html)
- [python-oracledb driver](https://python-oracledb.readthedocs.io/en/latest/index.html)
- [OCI Generative AI — documentation hub](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)

## Source

??? note "Full source — `examples/notebook_38_oracle_26ai_rag.py`"

    ````python
    --8<-- "examples/notebook_38_oracle_26ai_rag.py"
    ````
