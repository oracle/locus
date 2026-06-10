# Changelog

All notable changes to Tulip are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and — from 1.0
onward — [Semantic Versioning](https://semver.org). See
[`DEPRECATION.md`](DEPRECATION.md) for the deprecation and breaking-change
policy.

## [Unreleased]

### Added

- **`tulip.security` — evidence-grounded findings.** A new layer that
  makes Tulip a cybersecurity SDK rather than a general one:
  - `ground_finding()` / `ground_fingerprint()` turn a GSAR evidence
    partition into a typed `Finding` **only** when it clears the
    grounding threshold; otherwise they return an auditable
    `Abstention`. A `Finding` has no public constructor without a
    grounding score, so an ungrounded finding is unshippable by
    construction.
  - Typed schemas: `Finding`, `Indicator`, `FingerprintFinding`,
    `FingerprintVerdict`, and a `FingerprintClassifier` protocol for
    timing side-channel inference fingerprinting.
  - Threat-taxonomy reference enums: `AtlasTechnique` (MITRE ATLAS),
    `OwaspLLM` (OWASP Top 10 for LLM Applications, 2025), `OwaspASI`
    (OWASP Top 10 for Agentic Applications, 2026).
  - Pydantic + stdlib only — no new dependencies. mypy-strict, tested.

### Changed

- **Positioning: Tulip is the AI-cybersecurity agent SDK.** The cookbook
  (`examples/`) is re-aimed to security workflows, AI-security-led:
  prompt injection, jailbreaks, inference fingerprinting, RAG/memory
  poisoning, model extraction, and excessive agency as the primary
  track, with classic SOC/IR (triage, IOC enrichment, phishing, secure
  code review, incident response with approval gates) as the second.
  Scenarios are tagged to MITRE ATLAS / OWASP LLM / OWASP ASI, and the
  grounded-findings showcases (notebooks 27, 35, 37, 38–40, 45, 50) use
  the new `tulip.security` layer. GSAR typed grounding is the flagship:
  an ungrounded finding is a false positive by construction, so the
  agent abstains instead of shipping it. README, package description,
  keywords, and the `Topic :: Security` classifier updated to match.

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
- **Repo split:** the documentation site and the browser workbench move
  to dedicated repositories —
  [tuliplabs-ai/docs](https://github.com/tuliplabs-ai/docs) and
  [tuliplabs-ai/workbench](https://github.com/tuliplabs-ai/workbench).
  This repository carries the SDK and its cookbook (`examples/`).

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
