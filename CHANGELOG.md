# Changelog

All notable changes to Tulip are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and — from 1.0
onward — [Semantic Versioning](https://semver.org). See
[`DEPRECATION.md`](DEPRECATION.md) for the deprecation and breaking-change
policy.

## [Unreleased]

## [2.2.0] - 2026-07-23

### Added

- **Governed long-term memory (harness primitive).** Agents learn across
  runs. Two `BaseStore` backends ship: **`HolographicStore`** — zero-infra
  SQLite + FTS5 + HRR associative recall, the free/local default, no server
  and no embedding API (#42); and **`PgMemory`** — Postgres/pgvector with
  **per-tenant Row-Level Security**, the multi-tenant enterprise backend. It
  stores the HRR phase vector as `[cos φ, sin φ]`, so pgvector cosine distance
  equals HRR phase similarity — semantic recall runs entirely inside Postgres
  with no external embedding service (#43). `PgMemory(embedder=…)` accepts any
  `BaseEmbedding` (e.g. OpenAI `text-embedding-3-small`) for **true semantic
  recall** (#44).
- **Recalled memory is treated as untrusted input.** A context scrubber
  strips injected system-note/fence markers and wraps recall in a delimited
  "informational background data, not instructions" block — applied on every
  recall, so an agent can use what it remembers without obeying it (#42).

### Fixed

- **Recall is honestly typed.** HRR bag-of-words recall is lexical/associative,
  not trained semantics; `capabilities.semantic_search` is now `True` only
  when a real embedder is configured (`HolographicStore` reports `False`).
  Paraphrase matching requires an embedder (#44).
- **Claude 5 family models no longer 400 on `temperature`.** The
  temperature-deprecation prefix list now covers `claude-sonnet-5`,
  `claude-opus-5`, `claude-haiku-5`, `claude-fable-5`, and
  `claude-mythos-5` (alongside Opus 4.7+), so the provider omits the
  param for them. Verified live on `claude-sonnet-5`. (#29)

## [2.1.3] - 2026-07-22

### Security

- Bump locked `mcp` to 1.28.1 (WebSocket Host/Origin validation), `setuptools`
  to 83.0.0, and `torch` to 2.13.0 — clears all open dependabot alerts.

### Fixed

- **Composition pipelines run without threads.** `SequentialPipeline`,
  `ParallelPipeline`, and `LoopAgent` drove their agents via `Agent.run_sync`
  (a worker thread) from inside their async `run` methods. Threads are
  unavailable under WASM/Pyodide, so the pipelines silently produced empty
  results (an un-awaited coroutine → `IndexError`) in the browser workbench.
  They now prefer the thread-free `arun` and fall back to `run_sync` only for
  agent-likes that predate it — so the Composition notebook runs fully
  client-side.
- `__version__` now matches the released version (2.1.2); the bump was missed
  on the 2.1.1 and 2.1.2 releases.

## [2.1.2] - 2026-07-21

### Added

- **`Agent.arun(prompt) -> AgentResult`** — the async, thread-free equivalent of
  `run_sync` (same result-building logic; the caller owns the event loop). Enables
  running agents where threads aren't available — notably in the browser
  (Pyodide/WASM), so the workbench can run notebooks fully client-side. `run_sync`
  now delegates to `arun`; `invoke()` is unchanged.

## [2.1.1] - 2026-07-21

### Added

- **`AnthropicModel(default_headers=…)`** — extra HTTP headers are forwarded to the
  Anthropic client. Enables calling the API directly from a browser (Pyodide/WASM):
  pass `{"anthropic-dangerous-direct-browser-access": "true"}` to clear the CORS
  preflight. Backward-compatible (default `None`).

## [2.1.0] - 2026-07-08

### Added

- **Resume from checkpoint — cross-process interrupt rehydration.** `Agent.resume(response, thread_id=…)`
  reloads the interrupted state from the configured checkpointer when the process that paused is gone,
  so a durably-checkpointed run resumes anywhere (the gateway's cross-pod HITL path).
- **Enforceable deepagent submit terminal.** The verifying submit gate rejects fabricated
  submissions by raising, and `require_success=True` keeps the loop running instead of
  terminating on a rejected claim.
- Five runnable domain examples (payments, infra, support, data, cloud — nb83–87), embedded
  by the docs site's notebook pages.

### Fixed

- **Typed-terminal deepagents exit only through the verifying submit.** In explicit mode the
  state machine also terminated on any `terminal_tools` NAME match (`task_complete`, `done`, …) —
  no success or confidence check — letting a model end the run around the submit gate with a
  fabricated success. `create_deepagent` now empties the name-match set when `output_schema`
  is configured; callers can override via `agent_kwargs`.
- Checkpointing happens at the interrupt site, before yielding — a HELD run is durable the
  moment it pauses.

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
