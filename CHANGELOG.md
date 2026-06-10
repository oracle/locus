# Changelog

All notable changes to Tulip are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and — from 1.0
onward — [Semantic Versioning](https://semver.org). See
[`DEPRECATION.md`](DEPRECATION.md) for the deprecation and breaking-change
policy.

## [Unreleased]

## [1.0.0] — 2026-06-09

First general-availability release. From 1.0.0 Tulip follows Semantic
Versioning: breaking changes only land in major versions, with the
deprecation path described in [`DEPRECATION.md`](DEPRECATION.md).

### Changed

- **License:** relicensed from UPL-1.0 to **Apache-2.0**. Portions
  originally released under UPL-1.0 remain available under those terms —
  see `NOTICE`.
- **Versioning:** the `0.2.0bN` beta line is retired; Tulip goes GA at
  `1.0.0` with no further pre-releases.
- **Docs:** documentation moves to <https://tulipagents.ai/> with a new
  information architecture (Learn / Cookbook / Workbench / Reference)
  and a redesigned home page.

### Added

- Initial public release of **Tulip** (`tulip-agents`), a vendor-neutral
  SDK for building auditable agent teams.
- Agent runtime with the Think → Execute → Reflect → Terminate loop,
  idempotent tools, composable termination algebra, Reflexion, Grounding,
  and the GSAR typed-grounding layer.
- Eight orchestration shapes (Sequential / Parallel / Loop pipelines,
  StateGraph, Orchestrator + Specialists, Swarm, Handoff, A2A) and the
  PRISM cognitive router.
- Model providers: OpenAI, Anthropic, and any OpenAI-compatible
  endpoint via `base_url`.
- RAG: `PgVectorStore`, `QdrantVectorStore`, `ChromaVectorStore`,
  `OpenSearchVectorStore`, `InMemoryVectorStore`; `OpenAIEmbeddings` and
  `CohereEmbeddings`; `CrossEncoderReranker` (local) and `CohereReranker`.
- Memory: checkpointers for Redis, PostgreSQL, MySQL, OpenSearch, S3 /
  MinIO / R2, file, in-memory, and HTTP; long-term memory via
  `Mem0MemoryManager` or the portable `LLMMemoryManager`.
- Observability EventBus, MCP client + server, FastAPI `AgentServer`,
  and an evaluation harness.
