# Changelog

All notable changes to Tulip are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and — from 1.0
onward — [Semantic Versioning](https://semver.org). See
[`DEPRECATION.md`](DEPRECATION.md) for the deprecation and breaking-change
policy.

## [Unreleased]

## [2.10.0] - 2026-08-16

Findings from a measured comparison against AWS Strands 1.52.0 — both SDKs
installed side by side, every capability probed by import, and the same governed
refund run through each. Two of the findings were ours.

### Fixed

- **`AuditTrail.verify()` promised more than a hash chain can deliver.** Its
  docstring read *"no edit, deletion, or reorder"*. Edits, reorders, and
  deletions from the **middle** are all caught. Truncation is not: drop records
  off the end — or discard the trail entirely — and what remains is a valid
  shorter chain that returns `True`.

  This is a property of hash chains generally, not of this implementation:
  nothing inside a chain can attest to a link that was never handed to it. The
  cryptography was never wrong; the sentence was, in the project's flagship
  security feature. Both the method and module docstrings now state the
  boundary exactly, and `tests/unit/test_audit_truncation.py` pins all four
  attacks — including the one that is *supposed* to go undetected, so the
  docstring cannot quietly drift back.

- **Seven public symbols had no docstring** — three `a2a.protocol_v1`
  converters, two `rogue.challenge` entry points, and `router.goal_frame`'s
  `Risk` and `Complexity`. Every public export in the SDK now carries one.

### Added

- **`verify(expected_head=...)`** closes the truncation gap for callers who want
  it closed. Persist `trail.head` somewhere the agent cannot reach and pass it
  back; every attack — truncation included — moves the head:

  ```python
  anchor = trail.head                    # to a WORM bucket, or a co-signer
  ...
  trail.verify(expected_head=anchor)     # False if anything was removed
  ```

- **`refusal_reason` on `gate_tool`** — what the *user* hears when an action is
  refused. The default is still the policy's own reason, which names the checks
  that fired. That is right for an audit record and wrong for a customer: run
  against a live model, it produced *"the blast radius (3) exceeds the maximum
  1"* and *"classified as a large_refund"* in a customer-facing sentence. Pass a
  string, or `(decision) -> str` to vary by outcome. The full policy reason is
  still what goes to the trail.

### Not done

- **Suspending a run on a hold** — `on_refusal="interrupt"` — was built and then
  pulled. The gate can raise the pause, and the runtime does suspend and
  checkpoint. But on resume the loop folds the human's answer in as the *result*
  of the held tool call, and nothing re-invokes the gate to actually perform the
  action. The agent can then tell the user a refund was issued when nothing ran.
  A silent false success is worse than no feature, so this needs the agent loop
  to carry the approval through a resume, not a new parameter on `gate_tool`.

## [2.9.0] - 2026-08-16

One gap closed in `gate_tool`, found by checking a claim rather than repeating it.

### Added

- **`ApprovalBridge`, and `approval=` on `gate_tool`.** A held action now carries
  an `approval_id` the agent can poll and a `next` telling it how, while a human
  decides on a channel the agent cannot reach.

  Without it a hold told the model `"held_for_approval"` and stopped there — true,
  and not actionable, which leaves the agent apologising to a user about a refund
  that may already have been approved.

  This was found by verifying a claim made in 2.8.0's own notes: that
  `gate_tool`'s refusal matches what the `tulip-frameworks` bridges return. The
  five core keys did match. The bridges send two more on a hold, and those were
  missing.

  `ApprovalBridge` is a structural `Protocol` with no import-time dependency, as
  the bridges' is — so one broker object satisfies both and neither package has to
  import the other. A **denial** deliberately gets no id: it is final, and offering
  one would invite the agent to wait for a decision that is not coming.

  Both parameters are keyword-only with defaults; nothing that worked in 2.8.0
  changes shape.

## [2.8.0] - 2026-08-16

A release about controls that were not controlling anything. Six settings and
one whole feature were documented, shipped, and did nothing — and the pattern
in every case is the same: the work is done, and the result never reaches the
caller.

### Added

- **`gate_tool` — put the admission gate in front of a tool, in one line.**

  ```python
  agent = Agent(model=model, tools=[
      lookup_order,                                     # read-only, ungated
      gate_tool(issue_refund, policy=ControlPolicy()),  # gated
  ])
  ```

  `tulip-frameworks` has shipped `gate_langchain_tool` and its siblings for a
  while, so a LangChain user could put `admit()` in front of a tool trivially
  while a Tulip user hand-wrote the try/except — on the one feature the project
  is built around. Everything needed was already in core: `tulip.control.action`
  was promoted there in 2.3.0 "so the SDK, the gateway, the registry and these
  bridges share one derivation instead of four", and only the bridges used it.

  The returned tool keeps the original's name, description and parameter
  schema, so the model cannot tell the difference — the gate is not something
  it can be talked around. A refusal comes back as a readable result rather
  than an exception, in the same shape the bridges return, so a policy reads
  the same either way. `on_refusal="raise"` is there for a caller that would
  rather stop.

  Gating a **sandboxed** tool composes rather than replacing it: the gate
  decides, then the original tool runs in its own sandbox. A refusal never
  reaches the sandbox at all.

- **`GSARValidationError`**, which `GSARConfig.fail_on_low_score` had been
  documented as raising since it was written. See below.

### Fixed

- **Ten async clients outlived the event loop that built them.** `httpx` binds
  a connection pool to the loop running when the client is created, and
  `openai` and `anthropic` are `httpx`. Cached on `self`, they worked exactly
  once per process:

  ```
  loop 1: OK    loop 2: APIConnectionError: Connection error.    loop 3: OK
  ```

  Two things made it expensive. The message reads as a provider outage, so the
  first hour goes to the key, the network and a status page. And it *recovers*
  on the third loop, because the failed request evicts the dead connection — so
  it presents as an intermittent network blip. Two `asyncio.run()` calls, a
  notebook cell run twice, or FastAPI's `TestClient` all reach it.

  Fixed across `models/native/openai`, `models/native/anthropic`,
  `rag/embeddings/openai`, `providers/image`, `providers/speech`,
  `memory/backends/http`, `memory/backends/mysql`,
  `memory/store_backends/postgresql`, `rag/stores/opensearch` and
  `rag/stores/pgvector`, and factored into `tulip.core.loop_bound` — the
  pattern had already been hand-written three times with three different cache
  keys. A guard now fails on a lazily-cached client with no loop key.

- **`on_iteration_start` and `on_iteration_end` never fired.** `HookProvider`
  documents eight callbacks; six worked. The dispatch machinery existed and
  nothing called it. A hook that never fires is worse than one that does not
  exist: you write it, attach it, see no error, and conclude the run never
  reached that phase.

- **`AgentResult.grounding_score` and `.ungrounded_claims` were always `None`
  and `[]`.** The grounding loop ran — it can trigger replans — and emitted its
  verdict; nothing carried it to the result. (Note: grounding only runs when the
  agent used a tool, since evidence comes from tool results.)

- **`ExecutionMetrics.reflexion_evaluations` and `.grounding_evaluations` were
  always `0`.** The runtime counts both; they were locals in a generator.

- **`GSARConfig.fail_on_low_score` did nothing.** It was documented as raising
  `GSARValidationError`, and that exception existed only inside the sentence
  promising it. An agent explicitly configured to refuse un-grounded output
  shipped it silently — the one outcome the setting exists to prevent.

- **`GuardrailConfig.action_overrides` failed silently on a wrong key.** Rule
  names are prefixed: a pattern under `blocked_content_patterns["sql_injection"]`
  raises `blocked_sql_injection`. Overriding the bare name was a lookup miss
  that fell through to `default_action` without a word — you believed you
  downgraded a rule to WARN, it stayed at BLOCK. The project's own test fell
  into it. An override no rule can consult is now refused, and names the one you
  probably meant.

- **The memory backends accepted unknown constructor kwargs.** `notebook_68`
  passed `namespace=` where the field is `prefix=`; nothing complained, the
  namespace did not apply, and every run shared one Redis keyspace.

### Changed

- **`SteeringHook` now says which of its two controls can fail open.** Measured
  against a self-hosted Qwen3.6-35B: with `policy="Never allow delete or
  destructive operations."` the judge did not intervene and the agent reported
  deleting the table, while a tool calling `admit()` held under the same model
  and prompt. `policy` is advisory and enforced by a judge; `interrupt_tools`
  is a set-membership check that never consults the judge and cannot fail open.
  The docstring documented them identically.

## [2.7.0] - 2026-08-15

One deprecation, one retrieval bug, and the two most-read examples finally
matching their own documentation.

### Deprecated

- **`tulip.loop` is deprecated and will be removed in 3.0.0.** It is a second
  ReAct implementation, parallel to the one the supported `Agent` runs, and
  `Agent` has never used it — the only reference from the production runtime
  was one private helper, `_find_matching_execution`, now moved to
  `tulip.tools.executor.find_matching_execution`.

  Two implementations of the same idea is worse than either alone: they drift,
  and a bug fixed in one stays live in the other. Nothing in `tulip.loop` is a
  capability `Agent` lacks.

  Every name still imports and works until 3.0.0, and each access emits
  `TulipDeprecationWarning`. To find them in your own code:

  ```
  python -W error::DeprecationWarning -m pytest
  ```

  | Instead of | Use |
  |---|---|
  | `ReActLoop`, `create_react_loop` | `tulip.agent.Agent` |
  | `ReActLoopConfig` | `tulip.agent.AgentConfig` |
  | `LoopRunner` | `await agent.arun(prompt)` |
  | `BatchRunner` | `tulip.evaluation.EvalRunner` |
  | `StreamingCollector` | `async for event in agent.run(prompt)` |
  | `ConditionalRouter` | `StateGraph` conditional edges, or `tulip.router` |
  | `ThinkNode` / `ExecuteNode` / `ReflectNode` | internal to `Agent`; hook them with `tulip.hooks` |

  This is also the first use of `TulipDeprecationWarning`. The policy in
  `DEPRECATION.md` had been documented in two files and never exercised, so
  until now nothing proved a deprecation would actually reach a consumer.

### Fixed

- **`Mem0Manager` retrieved nothing.** Reads went out unscoped, so a lookup
  that should have been narrowed to the thread returned the wrong rows or
  none. Scoped through filters now.

- **Notebooks 06 and 07 are what their pages say they are.** The pages
  described LEDGER, a transaction-triage agent, and a deployment-readiness
  check. The code was a general-purpose assistant answering "What is the
  capital of Japan?" and a weather lookup. `grep -rl LEDGER examples/*.py`
  found nothing. These are the first two examples a new reader opens, and the
  drift was concentrated exactly there. The pages were right, so the code
  moved.

- **Four cross-references pointed readers at agents that do not exist.**
  `notebook_27` called notebook 26's orchestrator MARSHAL (it is STEWARD),
  `notebook_70` called notebook 27 CURATOR (it is RIGHTSIZER), and CURATOR
  appears nowhere in the repo. A reader who followed one arrived somewhere
  else with no way to tell which end was wrong.

- **The bundled mock answered every triage prompt identically**, so three
  transactions with three different right answers each got the same wrong one.

### Documentation

- `examples/README.md` explains the numbering: the numbers are stable
  identifiers, never reused, carrying no ordering. The gaps are not missing
  files — 10, 41-44 and 53-54 never existed. Renumbering to close them would
  break every published URL and every in-prose cross-reference to fix an
  appearance.

## [2.6.0] - 2026-08-15

The release that came out of a full audit of the SDK against its own
documentation. The framework was not weak; the first hour with it was broken.
Every defect found was in hand-written prose or fixture code, never in
generated reference, and they had one root cause: nothing ran the
documentation.

### Added

- **A step can say what it must find out.** `RequiredProbe` declares evidence
  a playbook step has to gather before it can honestly be called done.
  `expected_tools` asks "was the right tool called?"; this asks "was the right
  thing *looked at*?" — a different question, and the one an auditor actually
  has, since an agent can call the correct tool against the wrong target and
  satisfy the first while failing the second. A step names a capability with
  `uses`, and the skill supplies both the tools and the probes.

- **Eighteen model providers, up from two.** `openai:` and `anthropic:` are
  native; `ollama:` · `vllm:` · `lmstudio:` · `llamacpp:` · `litellm:` ·
  `groq:` · `together:` · `openrouter:` · `deepseek:` · `mistral:` · `xai:` ·
  `fireworks:` · `cerebras:` · `perplexity:` · `nvidia:` arrive with their
  base URL and key convention filled in, plus `openai-compatible:` for
  anything else. Before this, anyone on a self-hosted or gateway endpoint had
  to build the model object by hand.

- **`tulip.testing`** — `ScriptedModel`, `FunctionModel`, `text()`,
  `tool_call()`. The repo contained thirteen private `_ScriptedModel` classes,
  which is the shape of a missing feature. Includes a recording surface
  (`received_messages`, `offered_tools`, `call_count`, `last_prompt`) so a
  test can assert what the agent *sent*, not only what it returned.

- **`model_kwargs` on `AgentConfig`**, forwarded to `get_model()` when `model`
  is a string. `AgentConfig` sets `extra="forbid"`, so provider configuration
  could not travel with the documented one-string form — which made
  `openai-compatible:` reachable only through an environment variable, since
  that prefix *requires* a `base_url`.

- **`agent_name` on every event.** The docs asserted this and it did not
  exist. A caller merging two agents' streams had no way to tell the
  researcher's tool call from the writer's. A nested agent's events are never
  relabelled by the orchestrator around them.

- **Evals run against a `StateGraph`** via `as_eval_target()`. The docs showed
  `EvalRunner(agent=graph)`; every case errored. `expected_tools` and
  `expected_tool_sequence` now match on node ids, which is what a graph
  regression suite is actually for.

- **An LLM judge that exists.** `tulip.evaluation` advertised "LLM-as-judge
  scoring" and shipped 250 lines of boolean checks. `LLMJudge` grades against
  a written rubric and returns a typed `Verdict`; `check_trajectory` asserts
  tool *order*, which `expected_tools` could never express. The judge never
  retries for a pass, and raises rather than scoring zero when it cannot be
  reached — a "failure" that means the judge was down is worse than no eval.

- **RAG has an entrance.** `load_text`, `load_markdown`, `load_html`,
  `load_pdf`, `load_directory` and `recursive_chunks`. The vector stores and
  rerankers were real; the pipeline was blocked at its front door.

- **MCP is wired in.** `mcp_servers` on `AgentConfig`, and a helper that works
  in an async context — `to_tulip_tools()` called `run_until_complete()` from
  a sync method and raised inside a running loop.

- A **chat loop** example, and a **framework-interop** example that builds a
  real LangChain tool, drives it through LangGraph's own ReAct loop, watches a
  $4,000,000 refund execute, then wraps that one tool and runs the identical
  agent again.

### Fixed

- **Backend clients were cached across event loops.** `redis.asyncio` binds a
  connection pool to the loop that created it, so the second loop inherited a
  dead pool and failed with `Event loop is closed`. Not exotic: FastAPI's
  `TestClient` runs each request through its own portal, and any code calling
  `asyncio.run()` twice hits it. Fixed for Redis, OpenSearch and PostgreSQL.

- **`EvalRunner.run()` ignored `expected_tool_sequence`.** The sync path was a
  hand-copied second implementation that had drifted, so a case asserting the
  wrong tool order came back green. An ordering assertion silently never
  evaluated is worse than no assertion, because the report says it was
  checked.

- **The sliding window dropped the task.** When a window retained no user turn
  at all — an agent loop, assistant/tool all the way down — the opening
  request went with it, leaving the model working from a role description and
  a wall of tool output. On Qwen-family templates it fails outright.

- **OpenAI-compatible endpoints always get a user turn**, which several
  servers require and which a system-prompt-only request did not send.

- **Adherence counted the wrong probes** and reported 1.00 while failing.

- **The bundled mock could never call a tool**, so every tool-centric example
  printed `Tool calls made: 0` — including the notebook whose page says this
  is what turns an LLM into an agent.

- **Nineteen pages gave a "live model" command that silently ran the mock**,
  because `get_model()` read only `TULIP_MODEL_PROVIDER`.

- Six documented claims the code did not back, and a further four found on a
  second pass.

### Changed

- **CI runs the documentation.** Every Python block in `README.md` and
  `examples/README.md` is checked against the installed SDK — it must compile,
  every `from tulip... import X` must resolve, and keyword arguments must
  exist on the callable. Compile rather than parse, because `ast.parse`
  *accepts* top-level `await` and only `compile()` rejects it — which is
  exactly how a quickstart shipped raising `SyntaxError`.

- Fourteen test definitions across four files were dead: Python keeps the last
  binding, so a class defined twice in one module silently discards the
  earlier one. A guard now fails on same-module shadowing.

## [2.5.1] - 2026-08-12

A runtime fix and a version string that lied. Both were found by running
the SDK against real self-hosted models rather than by reading it.

### Fixed

- **A JSON-shaped tool call is now a tool call.** `_parse_text_tool_calls`
  recognised only call syntax — `search(query="x")` — so the JSON form that
  Ollama and the Hermes/Qwen templates emit whenever the server does not lift
  it into a structured `tool_calls` field was read as prose:

  ```json
  {"name": "isolate_production", "arguments": {}}
  ```

  Found with a real `qwen2.5-coder:7b`, which was talked into isolating
  production and emitted exactly that. The call was never dispatched, so it
  was never weighed by `admit()`, never written to the `AuditTrail`, and the
  run reported the model as having *declined*. It had not declined; the
  runtime could not see the attempt.

  Nothing executed, so this was fail-safe on the action — but not on the
  record, and for a runtime whose claim is that every consequential decision
  lands on a tamper-evident trail, an attempted dangerous action that leaves
  no trace is a governance gap. "Tried to wipe production" and "declined" must
  not look identical.

  **Behavioural note for anyone upgrading:** agents pointed at small
  self-hosted models will now perform tool calls that this version previously
  dropped in silence. That is the intended behaviour, and those calls now
  clear your `ControlPolicy` first — but if you were unknowingly relying on
  them not firing, they will fire now.

  Both shapes are validated against the tool registry, deduplicated so one
  call written in both cannot fire twice, and scanned by balancing braces
  rather than by regex so a nested `arguments` object is not truncated.
  Fenced blocks and double-encoded `"arguments": "{...}"` are handled.

- **`tulip.__version__` was a release behind.** `tulip_agents-2.5.0` shipped to
  PyPI with `METADATA Version: 2.5.0` and `__version__ == "2.4.0"` inside it —
  the literal in `src/tulip/__init__.py` and the one in `pyproject.toml` are
  maintained by hand and had drifted. Anything reading `__version__` for
  telemetry, a bug report, or a compatibility check was told the wrong release
  for the whole of 2.5.0. Corrected, and `tests/unit/test_version_is_consistent.py`
  now fails CI on drift instead of leaving it for PyPI to reveal.

### Changed

- **`Agent.__init__`'s docstring names the 36 options introspection cannot
  see.** `Agent` is a Pydantic model that also defines `__init__(**kwargs)`;
  `ModelMetaclass` builds `__signature__` from the explicit parameters and
  drops the `**kwargs`, so `termination`, `output_schema`, `memory_manager`,
  `web_search` and 32 others are invisible to `help()`, to editor autocomplete
  and to `inspect.signature()`. They are real and supported; `__signature__`
  itself is unchanged here.

- **`examples/can_you_make_it_go_rogue.py` runs without an API key**, against
  your own OpenAI-compatible endpoint, or against a frontier model — and no
  longer claims the gate won when the model simply refused.

## [2.5.0] - 2026-08-12

Everything here has been on `main` since 2.4.0 and the gateway already depends
on it. Cutting the release is the point: the gateway's CI resolves this package
**from source** while its production image installs it **from PyPI**, so a
symbol added here and never released passes every test and then fails inside
the container. That is not hypothetical — dev's cognitive router answered 500
on every routed run with
`PolicyGate.__init__() got an unexpected keyword argument 'denied_protocols'`
until this went out.

### Added

- **`PolicyGate.denied_protocols`** — a deployment can refuse protocol shapes
  by declaration, and the router will not select what policy has denied. The
  gateway wires this into `/v1/dispatch` and the CLI.
- **`dispatch()` accepts a pinned `GoalFrame`** — the resume seam. A resumed
  dispatch replays under the frame the approval was granted against, instead of
  re-extracting one a live model might frame differently.
- **`TerminateEvent` carries the segment's token usage** — what the gateway
  meters a run's cost from.
- **`InterruptEvent` carries structured input fields** — the field spec the
  Console renders as a form rather than as a sentence asking for one.

### Fixed

- **Governance and conversation survive a resume.** The resume loop was
  hook-blind and note-injecting; a redeemed tool no longer arrives ungoverned.
- **A second `ask_user` during a resume re-pauses** instead of running on.
- **SSRF blocked in `web_fetch`** (private and metadata destinations).
- **`ChromaStore`** warns self-hosted-server operators about CVE-2026-45829.
- **`decision_status`** typed as `Literal["resolved", "abstain"]` (GSAR).
- Dependency bumps clearing Dependabot alerts: aiohttp 3.14.3,
  cryptography 50.0.0, h2 4.4.1.

## [2.4.0] - 2026-08-04

### Added

- **The OpenAI provider speaks the Responses API.** `OpenAIModel` gains an
  `api` setting — `"chat_completions"`, `"responses"`, or `"auto"` (the
  default), which routes the model families only `/v1/responses` serves
  (gpt-5.6-*) there and keeps everything else on chat-completions. GPT-5.6
  rejects function tools on chat-completions whenever reasoning is active
  ("Function tools with reasoning_effort are not supported … use
  /v1/responses or set reasoning_effort to 'none'"), so the family could
  previously call tools only with reasoning disabled — defeating its
  purpose. Auto-selection never fires for a custom `base_url`:
  OpenAI-compatible gateways (Together, vLLM, LiteLLM) serve
  chat-completions, not `/v1/responses`. Both `complete()` and `stream()`
  are covered; chat-completions spellings translate so callers don't care
  which transport is active (`max_tokens` → `max_output_tokens`,
  `reasoning_effort` → `reasoning.effort`, `response_format` →
  `text.format`, chat-shaped `tool_choice` flattened), and usage + stop
  reasons land in the chat vocabulary (`stop` / `tool_calls` / `length`).
  Reasoning stays on: no effort is ever defaulted. The transport stays
  stateless (`store=False`) — raw output items (reasoning items with their
  `encrypted_content`, function calls) ride along in the assistant
  `Message.metadata` and are replayed verbatim next turn, which is what
  reasoning models require to continue a tool-calling turn without
  server-side storage. Dropped for lack of a Responses equivalent: `seed`,
  `stop` sequences, penalties; streamed turns reconstruct history without
  reasoning items (#60).
- **Sandboxed tool execution.** `@tool(sandbox=True)` ships the function's
  source into an isolated box and runs it there — the host process never
  executes the body, and direct `tool(...)` calls are sandboxed too, so
  there is no bypass. The zero-infra default is the new
  `tulip.tools.sandbox.SubprocessSandbox` (fresh working directory,
  `python -I`, environment scrubbed to `PATH`/`LANG` plus what the manifest
  explicitly grants, per-call timeout). Stronger boundaries plug in through
  the structural `ToolSandbox` protocol: `TULIP_SANDBOX=docker` (or a
  provider name / object / `SandboxSpec`) resolves Docker, Firecracker,
  SSH and Lambda providers from the optional `tulip-sandbox` package by
  duck typing — neither package imports the other. Runs emit
  `tool.sandbox.started` / `tool.sandbox.completed` on the event bus (#7).
- **Policy-required sandboxing.** `ControlPolicy.require_sandbox_for` names
  the labels whose actions must execute in a sandbox: `approve()` denies a
  matching action that doesn't carry the new `SANDBOXED_TAG` tag, and the
  new `SandboxEnforcerHook` enforces the same rule at the agent loop's
  `on_before_tool_call` seam — an un-sandboxed call to a tool labelled
  (via the new `@tool(labels={...})`) with a required label is cancelled
  before it runs, and `tool.sandbox.denied` is emitted (#7).

## [2.3.0] - 2026-08-01

### Added

- **Token-level streaming from the agent loop.** `agent.run(..., stream_tokens=True)`
  also yields `ModelChunkEvent` as the model produces them, so text and
  chain-of-thought render while the turn is still running. Tool and termination
  events are unchanged and the assembled response is identical to the
  non-streaming one, so hooks, retries, grounding and termination behave the
  same. Off by default — it changes which event types a consumer sees.
  Previously a streaming chat UI had to abandon the loop and re-implement ReAct
  over a raw provider client, losing admission, audit and the tool-loop guard
  with it (#52).
- **The full Chat Completions surface is reachable.** `complete()` / `stream()`
  read six keys out of `**kwargs` and dropped the rest — of the 36 parameters
  the API accepts, 23 were silently discarded, including `tool_choice`,
  `parallel_tool_calls`, `stream_options`, `logprobs` and `reasoning_effort`.
  Any Chat Completions parameter the caller passes is now forwarded; the
  accepted set is introspected from the `openai` package's own request
  TypedDicts, so a field OpenAI adds is forwardable on a dependency bump rather
  than waiting on a hand-maintained list (#56).
- **`extra_body` on the OpenAI provider** for fields outside the OpenAI schema —
  vLLM's `chat_template_kwargs` (`enable_thinking`), `top_k`, `min_p`,
  `repetition_penalty`. Per-call values merge over config, and it applies to
  reasoning models too, which reject sampling parameters but still accept
  provider extensions (#56).
- **Per-run model parameters from `Agent`.** `run()`, `arun()` and `run_sync()`
  take `model_kwargs`, forwarded to the model call and winning over agent
  config. Model configuration is fixed for a model's lifetime, which is the
  wrong shape for anything that must vary per run — `tool_choice` above all (#55).
- **`ModelResponse.logprobs` and `ModelResponse.candidates`.** Both reached the
  server already but had nowhere to land, so the tokens were paid for and
  discarded; `n>1` is now usable and single-candidate callers see an empty
  list (#53).
- **`ModelChunkEvent.usage` and `.stop_reason`** on the terminal chunk, so a
  streaming caller can meter a turn and tell a natural stop from a `length`
  truncation — which on reasoning models otherwise surfaces as an empty reply
  rather than an error (#54).

### Fixed

- **Sampling the caller configured is no longer discarded.** The loop sent
  `AgentConfig.temperature` (0.7) and `max_tokens` (4096) unconditionally, and
  those land as *per-call* arguments that beat a provider's own config — so
  `get_model("openai:…", temperature=1.0, max_tokens=8192)` was silently
  ignored and every turn went out at 0.7 / 4096. Both now default to `None`
  (defer to the model) and are sent only when explicitly set. Effective
  defaults are unchanged, since `ModelConfig` also defaults to 0.7 / 4096.
- **`temperature` / `top_p` of `None` are omitted from the request**, letting a
  server's own defaults apply. Self-hosted models publish their recommended
  sampling in `generation_config.json`, and a value sent unasked overrides it.
- **Mid-run guidance no longer 400s on OpenAI-compatible servers.** The loop
  injects grounding replans, repair prompts and iteration nudges as *system*
  messages, and several chat templates accept a system message only in first
  position — vLLM serving Qwen rejects the request outright with `System
  message must be at the beginning`, killing a run partway through and only
  when it happened to need guidance. Later system messages are now re-encoded
  as marked user notes, preserving the text and its steering (#57).
- **Anthropic streaming dropped every tool call.** `stream()` read only
  `text_stream`, so `tool_use` blocks, usage and the stop reason never
  surfaced — a streaming tool-using agent silently made no tool calls at all.
  It now reads the assembled final message (#52).

- **`PgMemory` could not create its own schema with default settings.** `dim`
  defaulted to 1024 and the HRR `[cos φ, sin φ]` encoding doubles it, asking
  pgvector for a 2048-dimension column — over the 2000-dimension ceiling for an
  HNSW index, so `CREATE INDEX` raised `ProgramLimitExceededError` and no fact
  was ever written. `dim` now defaults to **512** (a 1024-wide column), an
  explicit `dim` whose doubled width cannot be indexed is rejected at
  construction with both numbers named, and an *embedder* wider than the limit
  is allowed but warns loudly that the table has no ANN index.
- **`PgMemory` hid its own schema failures.** `_get_pool` assigned `self._pool`
  before running `_ensure_schema`, so a schema error surfaced on the first call
  only; every later call found a pool, skipped schema creation and ran against a
  half-built table (sequential-scan recall, silently). The pool is now published
  only after schema creation succeeds, and first use is serialised by a lock.
- **`PgMemory` now detects a pre-existing table of a different vector width**
  (`CREATE TABLE IF NOT EXISTS` kept it silently) and fails with the two widths
  and the remedy instead of a per-INSERT `expected N dimensions, not M`.

### Documentation

- Notebook 11 gains a token-streaming example, and its header no longer implies
  the default streams tokens.
- Notebook 56 documents model configuration, per-run `model_kwargs`, and the
  self-hosted sharp edges: omitting sampling with `None`, `extra_body`, and
  server-side rejections such as vLLM refusing `min_p` / `logit_bias` under
  speculative decoding (#56).

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
