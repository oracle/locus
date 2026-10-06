# ObservationPack: purpose, design and status

Handover notes for whoever picks this up next. Last updated 2026-10-03, after PR #100 merged.

## Why we are building it

A tulip-code benchmark run (14 real tasks, DeepSeek V4 Pro) passes the most
tasks of the harnesses we compared (60%), but sends about **5M input tokens per
run**. A large share of that is old tool output. A coding agent reads a 20 KB
file or a long test log once. Today that output then rides along in every
later request until compaction clears it. When compaction does clear it, the
bytes are gone, so the model has to run the tool again.

NVIDIA's SoL-Pi paper (arXiv 2609.20519, repo `NVlabs/SoL-Pi`,
`src/sol-pi/extensions/observation-pack/`, MIT) tested several harness
mechanisms. ObservationPack was the best single one in their ablation:
score 44.8 → 47.2 with 6% fewer tokens. It is also **lossless**: the model can
always get the exact bytes back. We ported the idea to Python in the SDK
(tulip-agents) and turned it on in tulip-code, which is built entirely on the
SDK.

## What it does

1. **A large output goes whole, then as a stand-in.** A text tool output over
   10 KiB is sent in full for its first 2 model requests. After that, the
   request carries a short placeholder: an id, the size, about 1 KB of the
   first and last lines, and a hint to call `obs_recall`.
2. **Only the outgoing request changes.** The run's stored history and its
   checkpoints always keep the full output.
3. **The exact bytes are archived.** They go to a content-addressed archive
   per session: `<directory>/<session>/observation-pack/objects/<obs_id>.txt`.
   The archive is written exclusively, refuses symlinks, and checks hashes.
   `swapped.txt` keeps the swaps across a restart, and `ledger.jsonl` logs
   what happened.
4. **`obs_recall(id, offset | line)` reads them back.** It is registered
   automatically. It pages the exact bytes, up to 16 KB or 400 lines per call,
   and never splits a UTF-8 character.
5. **It fails open.** Any error sends the full output instead.

### The prompt-cache problem, and how it is handled

Providers charge about a tenth for the part of a request that is
byte-identical to the previous one. Our runs hit that cache about 92% of the
time, because history only grows at the end. Swapping an output for its
placeholder changes an earlier part of the request, so everything after it
loses the discount on that request. Doing that on every request would cost
more than it saves.

`SwapCostModel` therefore **batches and prices** swaps. A batch of due outputs
waits until both of these are true:

- together they free at least `min_batch_bytes` (32 KiB);
- `saved_tokens × cache_read_cost × horizon ≥ rewrite_tokens × (cache_write_cost − cache_read_cost)`.

Here `horizon` is the number of requests made so far (at least 4), capped at
the requests left before compaction is due.

The exception is a swap behind a part of the request that is changing anyway
(a compaction, or a sliding window). That swap costs nothing and goes
straight away. Once an output is swapped it stays swapped, so the new prefix
is stable. All prices and thresholds are config fields, so they can be tuned
from benchmark logs.

Scripted runs, one 13 KB read per step, cost weighted with cached input at 0.1:

| Run | Cache breaks | Cost vs. no pack |
|---|---|---|
| 40 steps, 1M window | 12 (38 if every output were swapped on its own) | −42% |
| 60 steps, 128k window | 19, and no compactions (2 without the pack) | −28% |
| 40 steps, 30k window | swaps ride on compaction | about +1% |

### How it works with compaction (ContextCompactor, #86)

- **Compaction triggers later.** The trigger measures the request as it is
  actually sent, so outputs that are already swapped no longer push the run
  towards compaction.
- **Clearing old output is lossless.** Stage-1 clearing replaces an old
  output with a one-line stub that names its archive id. It falls back to the
  old lossy stub only if archiving fails.
- **Summaries keep the ids.** A summary ends with a "Recallable tool outputs"
  list of the ids it folded in, and the next summary carries that list
  forward.

### Mechanism ledger

The pack records into Action Fusion's per-run `MechanismLedger` as
`observation_pack`, with one of these outcomes:

| Outcome | When |
|---|---|
| `swap` | once per swap batch |
| `placeholders` | once per request that sent any, with the bytes not resent |
| `recall` | each `obs_recall` call |
| `cleared_recallable` | an output compaction cleared into a recallable stub |
| `archived_on_arrival` | an output too large to send whole, archived when it came in |
| `fail_open` | something failed and the full output was sent |

Each row carries the `session`, and the `agent` name when the pack belongs to
a named agent such as a subagent. Per-session counters are also available from
`agent.observation_pack.stats(thread_id)`.

## Where things are

| Piece | Location |
|---|---|
| Mechanism | `src/tulip/memory/observation_pack.py` (SoL-Pi MIT notice in the header and in `THIRD_PARTY_LICENSES.txt`) |
| Config | `ObservationPackConfig` in `src/tulip/agent/config.py`; `AgentConfig(observation_pack=True \| ObservationPackConfig(...))`, off by default in the SDK |
| Request transform | `_get_model_response` in `src/tulip/agent/runtime_loop.py` |
| Compaction hooks | `src/tulip/memory/compaction.py` (`archive=` on `compact`) |
| Tests | `tests/unit/test_observation_pack.py`; prefix-break bounds in `tests/unit/test_cache_prefix.py` |
| tulip-code wiring | `src/tulip_code/observation.py`; on by default; off with `TULIP_CODE_OBSERVATION_PACK=0` or `"observationPack": false`; the result record has `observation_pack` and `mechanisms.observation_pack` |

## Status

### Done and merged into `harness/integration-all`

- **tulip-agents PR #99:** the mechanism, cache batching, compaction
  integration, and ledger rows. Merge commit `535b755`.
- **tulip-code PR #23:** on by default, ablation switch, result record.
  Merge commit `94b96b8`.
- Both full suites passed in the VM before the merge.

### Also merged: whole outputs past the cap, and subagents (PR #100, merge `a9aedf0`)

**1. Nothing lost to the per-result cap.**

- With the pack on, `max_tool_result_length` (32,000 chars) no longer cuts
  output before the pack sees it. An output up to `max_inline_chars`
  (default 128,000 chars) goes whole.
- That limit is never more than an eighth of a known context window and never
  less than the old cap.
- A larger output is archived whole when it arrives
  (`ObservationPack.intake`) and sent cut around a marker. The marker gives
  the archive id and the byte offset of the cut, so `obs_recall` reads exactly
  what was left out.
- Later placeholders and compaction stubs for that output point at the
  archived original, not at the cut copy.
- The old cap still applies when the pack is off, when archiving fails, when
  the output contains images, or when an explicit `tool_result_store` is
  configured.

**2. Subagents get the pack too.**

- A subagent started by a run that has the pack (the `task` tool,
  `run_subagent`, `Subagent`) gets the parent's settings and `obs_recall`.
- Its archive lives in its own namespace under the parent's session:
  `<session>/subagents/<task>/observation-pack/`.
- Its ledger rows are attributed with `agent=<subagent name>` and its own
  session.
- A caller that passes `observation_pack=` explicitly keeps its own choice.
- How it reaches the subagent: the parent's settings ride on the
  parent-run context (`enter_parent_run(observation_pack=...)`), and
  `_build_child` in `src/tulip/agent/subagent.py` picks them up.

Both full suites passed in the VM before the merge. tulip-code needed no
change: its `task` tool builds subagents through the SDK, and the inline
limit comes from the SDK config.

### Not started (deliberately)

- **A real benchmark ablation.** Run the 14 tasks with
  `TULIP_CODE_OBSERVATION_PACK=1` and with `=0`, then compare input tokens,
  cache hits, cost and pass rate. This needs paid model calls, so it waits for
  a go-ahead.
- **Tuning.** Adjust `min_batch_bytes`, `cache_read_cost` /
  `cache_write_cost`, `horizon_requests` and `max_inline_chars` once those
  benchmark logs exist. The ledger rows and `stats()` give the numbers needed.

## Working rules for these repos

- Base branches on `origin/harness/integration-all`. Never edit, commit or
  switch branches in the shared checkouts under `/home/fede/Projects/tuliplabs/<repo>`.
  Create worktrees under `/home/fede/Projects/tuliplabs/.worktrees/` with
  **absolute** paths.
- Commit as `tuliplabs <328004975+tuliplabs@users.noreply.github.com>` with
  `-s` (DCO). No AI attribution anywhere.
- Run `gh` as `GH_TOKEN=$(gh auth token -u tuliplabs) gh ...`. No paid model
  calls.
- Locally, run only the touched tests, with at most 2 workers. Full suites run
  in the VM.
