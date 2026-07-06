# Changelog

All notable changes to Tulip are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and — from 1.0
onward — [Semantic Versioning](https://semver.org). See
[`DEPRECATION.md`](DEPRECATION.md) for the deprecation and breaking-change
policy.

## [Unreleased]

### Changed

- **Positioning: Tulip leads as a first-class agentic framework — "the safest way to
  build agentic AI."** The identity is framework-first and safety-led: control is native
  to the core via three points — the **cognitive router** (PRISM) picks the runtime shape,
  **GSAR** grounds every claim (or abstains), and the **admission gate** (`admit()`) gates
  every risky action — packaged as safety. AI security is repositioned from the SDK's
  identity to its **flagship proof domain**. README, the `tulipagents.ai` landing, package
  description / keywords / classifiers, and `CONTRIBUTING` reflect the framework-first,
  safety-led identity. No API changes.

## [2.0.0] - 2026-06-25

### Changed

- **Breaking: the domain-neutral control core moves to `tulip.control`.** The new
  namespace owns `admit()` / `Action` / policy / audit / `governed_agent`;
  `tulip.security` keeps the security domain and no longer re-exports control.
  Renames, with no deprecation shims: `SecurityPolicy` → `ControlPolicy`,
  `Finding` → `Evidence`, `Verdict` → `VerificationResult`,
  `secure_agent` → `governed_agent`, `SecurityProfile` → `GovernanceProfile`.
  Update imports to `from tulip.control import Action, admit, ControlPolicy, AuditTrail`.

## [1.1.0] - 2026-06-24

### Added

- **Control-first repositioning — `admit()` as the headline.** The drop-in story:
  add the admission gate + tamper-evident audit around the agent you already have
  (any framework) in ~8 lines — risky actions are policy-gated and
  human-approvable, and every decision is a hash-chained record you can replay and
  cannot forge. New runnable examples: `can_you_make_it_go_rogue.py` (jailbreak the
  model — the gate still blocks the action), `governed_soc_action.py`
  (gate → hold-for-human → audit), `grounding_ablation.py` (same model ± grounding).
- **Adversarial `verify()`.** `AdversarialSkeptic` adds an LLM-backed skeptic that
  actively challenges a finding's evidence and emits typed `Refutation`s, alongside
  the existing deterministic checks — a hallucinated "critical" is refuted before it
  can drive an action.
- **`UnsandboxedCodeExecution` red-team probe** (OWASP ASI05) — effect-grounded
  proof-of-execution via an unforgeable nonce digest; registered in the `owasp-asi`
  suite. Response-only, target-agnostic, cannot false-positive.

## [1.0.0] — 2026-06-09

First general-availability release. From 1.0.0 Tulip follows Semantic
Versioning: breaking changes only land in major versions, with the
deprecation path described in [`DEPRECATION.md`](DEPRECATION.md).

### Changed

- **Positioning: Tulip is the AI-cybersecurity agent SDK.** The cookbook
  (`examples/`) is AI-security-led — prompt injection, jailbreaks, inference
  fingerprinting, RAG/memory poisoning, model extraction, and excessive agency
  as the primary track, with classic SOC/IR (triage, IOC enrichment, phishing,
  secure code review, incident response with approval gates) as the second.
  Scenarios are tagged to MITRE ATLAS / OWASP LLM / OWASP ASI; README, package
  description, keywords, and the `Topic :: Security` classifier reflect the
  cybersecurity identity.
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
- **`tulip.security` — evidence-grounded findings**, the layer that makes
  Tulip a cybersecurity SDK rather than a general one: `ground_finding()` /
  `ground_fingerprint()` turn a GSAR evidence partition into a typed `Finding`
  **only** above the grounding threshold, else an auditable `Abstention` — a
  `Finding` has no public constructor without a score, so an ungrounded finding
  is unshippable by construction. Typed schemas (`Finding`, `Indicator`,
  `FingerprintFinding`, `FingerprintVerdict`), a `FingerprintClassifier`
  protocol, and threat-taxonomy enums (`AtlasTechnique` / MITRE ATLAS,
  `OwaspLLM`, `OwaspASI`). Pydantic + stdlib only, mypy-strict.
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
