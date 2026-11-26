---
title: Install extras
description: "Which pip extra enables which part of the SDK: model providers, serving and observability, checkpoint backends and memory, RAG stores and rerankers, the coding harness, durable runs, control, and the bundles."
---

# Install extras

The base install carries the agent loop, the tools, the event bus, the control
layer, the file and in-memory backends, and the OCI Generative AI and OpenAI
model providers. Every other service client and database driver is an extra,
so an environment holds only the clients it uses. Extras combine on one line:

```bash
python -m pip install "locus-sdk[redis,pgvector]"
```

Each table names the extra, what it enables, and the packages it adds.

## Model providers

| Extra | Enables | Packages |
|---|---|---|
| `openai` | Included in the base install; the extra remains for requirements that name it. OpenAI, and every OpenAI-compatible route: OpenRouter, Together, RunPod, Ollama, vLLM, LM Studio, llama.cpp and any `base_url=` | `openai` |
| `anthropic` | Anthropic | `anthropic` |
| `oci` | Included in the base install; the extra remains for requirements that name it. OCI Generative AI on both transports; the chat-completions transport runs on the OpenAI client and the request signer on `requests` | `oci`, `openai`, `requests` |
| `gemini` | Gemini Live realtime voice sessions (`locus.voice`); Gemini chat models go through the OpenAI-compatible route | `google-genai` |
| `llm` | Every provider above in one line | `openai`, `anthropic`, `oci`, `requests` |

## Serving, deployment and observability

| Extra | Enables | Packages |
|---|---|---|
| `server` | `AgentServer` and the A2A server: the FastAPI app with SSE streaming, JSON-RPC and HTTP+JSON bindings | `fastapi`, `uvicorn` |
| `a2a-grpc` | The A2A gRPC binding; `import locus.a2a` needs none of it | `grpcio`, `grpcio-status`, `protobuf` |
| `deploy` | `locus deploy`, which reads a YAML values file | `pyyaml` |
| `functions` | `AgentServer` behind an OCI Function (the Fn FDK) | `fdk` |
| `telemetry` | `TelemetryHook`: OpenTelemetry traces and metrics with the OTLP exporter | `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp` |
| `mcp` | MCP servers as tools | `mcp` |

## Checkpoints, stores and long-term memory

| Extra | Enables | Packages |
|---|---|---|
| `redis` | Redis checkpointer | `redis` |
| `postgresql` | PostgreSQL checkpointer | `asyncpg` |
| `mysql` | MySQL checkpointer; the same driver serves HeatWave vector columns | `mysql-connector-python` |
| `opensearch` | OpenSearch checkpointer | `opensearch-py` |
| `s3` | S3-compatible object-storage checkpointer: OCI Object Storage through its S3 API, MinIO, R2 | `boto3` |
| `oracledb` | Oracle Database 26ai: checkpointers, the versioned saver, store, vector store, in-database chunker and embeddings, the ADB loader | `oracledb` |
| `checkpoints` | Every checkpoint backend above in one line | `redis`, `asyncpg`, `mysql-connector-python`, `opensearch-py`, `boto3` and more |
| `encryption` | Checkpoint encryption at rest (AES-256-GCM) for every backend; plaintext checkpoints need nothing | `cryptography` |
| `agentmemory` | `OracleAgentMemoryManager`, on Oracle's agent-memory client | `oracleagentmemory` |
| `mem0` | `Mem0MemoryManager`, the portable self-hostable memory backend | `mem0ai` |

## RAG: vector stores, embeddings and rerankers

The Jina, Voyage and LLM rerankers need nothing beyond the base install.

| Extra | Enables | Packages |
|---|---|---|
| `pgvector` | PostgreSQL with pgvector, and `PgMemory`'s HRR encoding (without `numpy` rows store no vector and recall falls back to substring match) | `asyncpg`, `numpy` |
| `qdrant` | Qdrant | `qdrant-client` |
| `chroma` | Chroma | `chromadb` |
| `milvus` | Milvus, Milvus Lite (embedded) and Zilliz Cloud | `pymilvus` |
| `weaviate` | Weaviate; the embedded server downloads its binary on first use | `weaviate-client` |
| `elasticsearch` | Elasticsearch | `elasticsearch` |
| `redis-vector` | Redis Query Engine vector fields (Redis Stack, Redis 8) | `redis` |
| `pinecone` | Pinecone | `pinecone` |
| `mongodb` | MongoDB Atlas vector search | `pymongo` |
| `heatwave` | MySQL HeatWave vector columns, and plain MySQL 9+ | `mysql-connector-python` |
| `cohere` | Cohere embeddings and reranker over the direct API | `cohere` |
| `rerank-local` | A local cross-encoder reranker | `sentence-transformers` |
| `rerank` | Both rerankers above | `cohere`, `sentence-transformers` |
| `rag` | Every store, embedder and reranker Locus ships, for picking a store at runtime | `asyncpg`, `opensearch-py`, `oracledb`, `oci`, `requests` and more |

## Coding harness

Each client is imported only by its backend's connect or create call; a caller that passes its own client needs nothing from the extra.

| Extra | Enables | Packages |
|---|---|---|
| `openshell` | NVIDIA OpenShell sandboxes as the workspace | `openshell` |
| `oci-sandbox` | OCI GenAI Sandboxes as the workspace (the client ships in the OCI SDK from 2.189.0) | `oci` |
| `ssh` | Any host with `sshd` as the workspace | `paramiko` |
| `oci-container` | OCI Container Instances as the workspace: the OCI SDK provisions, `paramiko` works inside | `oci`, `paramiko` |
| `e2b` | E2B hosted sandboxes | `e2b` |
| `daytona` | Daytona hosted sandboxes | `daytona` |
| `browser` | Browser tools; after installing, run `playwright install chromium` | `playwright` |

## Durable runs and control

| Extra | Enables | Packages |
|---|---|---|
| `temporal` | Durable agent runs on Temporal | `temporalio` |
| `dbos` | Durable agent runs on DBOS | `dbos` |
| `audit` | Signed audit trails (Ed25519) that verify from public keys alone; unsigned trails need nothing | `cryptography` |

## Bundles and development

| Extra | Enables | Packages |
|---|---|---|
| `sdk` | Every provider and backend above | `openai`, `anthropic`, `cohere`, `sentence-transformers`, `opentelemetry-api` and more |
| `docs` | The documentation site; the API reference imports the package, so install the SDK alongside: `pip install -e ".[docs]"` | `mkdocs`, `mkdocs-material`, `mkdocstrings` |
| `dev` | The test and lint toolchain for a checkout | `hatch`, `pre-commit`, `ruff`, `mypy`, `pytest` and more |

## See also

- [Quickstart](quickstart.md): the first install and a first run.
- [Configure a model provider](configure-providers.md): the variables each provider reads.
- [Environment variables](environment-variables.md): every variable the SDK and the notebooks read.
