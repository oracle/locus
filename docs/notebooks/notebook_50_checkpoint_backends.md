# Checkpoint backends

A customer-support case runs for days and survives restarts, shift handoffs and redeploys only if its conversation state lives outside any one process. This notebook runs in two steps:

- It first drives the [time-travel surface](../concepts/checkpointers.md#time-travel) every checkpointer has — `list`, `fork`, and `run(..., from_checkpoint=...)` — on the in-memory checkpointer, which needs no credentials.
- It then drives the checkpointer contract by hand against `S3Backend`, which keeps every checkpoint as objects under one bucket prefix: here OCI Object Storage through its S3-compatible API, and with the same code S3, MinIO or Cloudflare R2, all via boto3.

(`OCIBucketBackend` keeps the same object layout and reports the same capabilities over the OCI Python SDK; it signs with an OCI configuration profile, a session token or an instance or resource principal instead of the Customer Secret Key pair the S3-compatible API uses.) The [conversation memory](notebook_05_agent_memory.md) notebook covers the checkpointer contract itself.

The in-memory checkpointer covered there is the developer default;
object storage is the
[production default](../concepts/checkpointers.md#picking-a-backend). A
support case round-trips through `save` → `load` → `save` across shift
handoffs, so multi-day continuity is the checkpointer contract applied
in a loop.

`S3Backend` reports `metadata_query`, `vacuum`, `branching`,
`list_threads`, `list_with_metadata` and `persistent_checkpoint_ids`,
with `search` and `ttl` false. One caveat on `metadata_query`:
`S3Backend` implements only `get_metadata`; `query_by_metadata` falls
through to `BaseCheckpointer`, which raises `NotImplementedError` even
though the descriptor advertises the capability.

What this notebook covers:

- List a case's checkpoints with the step each was written at and the
  checkpoint it descends from; fork the case at an earlier checkpoint
  and work the fork without touching the original; re-run a turn from a
  past checkpoint so the thread branches in place; draw both threads as
  one Mermaid history with the fork as a branch. This part runs in
  memory with the bundled scripted model and no credentials.
- Save and load a support case's `AgentState` via `S3Backend`.
- Inspect the reported capabilities.
- Walk thread history with `list_threads` / `list_checkpoints`.
- Vacuum old checkpoints with `S3Backend.vacuum`.
- Branch on the capability descriptor before calling `search`.

## Prerequisites

- The `s3` extra (`pip install "locus-sdk[s3]"`): boto3 is not a core
  dependency and `S3Backend` imports it lazily, so without it the
  missing-variable path below still exits cleanly, but the first call
  that needs a client, here the first `save`, raises `ImportError`.
- An S3-compatible bucket named in `S3_BUCKET`, a key pair in
  `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, and `S3_ENDPOINT_URL`
  for any endpoint other than S3 itself. The notebook checks the three
  required variables before it builds the backend, prints the missing
  names and exits.

`S3Backend` is the first of the two checkpointer shapes in the drawing:
it subclasses `BaseCheckpointer` and is passed to an agent unchanged,
as are `OCIBucketBackend` and the native `PgCheckpointer`. Redis,
PostgreSQL, MySQL, OpenSearch and Oracle Database are the second shape,
dict-shaped storage behind `StorageBackendAdapter`; each
`*_checkpointer()` factory does the wrapping, and the adapter derives
its capability descriptor from the methods the storage class defines.

Object storage has no index, so `S3Backend` does not report `search`
and the notebook branches on the descriptor rather than calling and
catching. `opensearch_checkpointer()` and `oracle_checkpointer()` do
report `capabilities.search`: `OpenSearchBackend.search` runs a
`multi_match` query over the indexed checkpoint JSON and
`OracleBackend.search` a case-insensitive `LIKE` over the JSON column.
PostgreSQL and MySQL offer exact matches instead: `query_by_metadata`
through the checkpointer, and `search_data`, a JSON-containment match
on one top-level state field, on the backend object only, so their
adapters report `search` as false.

The notebook calls `save`, `load` and `list_checkpoints` by hand, the
same contract the agent loop exercises at the save points shown:

{{ locus_diagram checkpointer-shapes }}

## Key lines

The time-travel part needs nothing but a checkpointer: `list` describes
the history newest first, `fork` starts a second case from the day-2
checkpoint, and `draw_mermaid_history` draws both. The agent's own saves
answer to the same calls, so a turn can be re-run from an earlier
checkpoint and the new turn descends from it:

{{ locus_source examples/notebook_50_checkpoint_backends.py::time_travel_in_memory }}

One `S3Backend` over a bucket and a key prefix is the whole
checkpointer. `endpoint_url` points it at an OCI Object Storage
S3-compatibility endpoint or a local MinIO and is left unset for S3
itself; no key pair is passed, so boto3 resolves credentials from its
default chain, which is where `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` are read:

{{ locus_source examples/notebook_50_checkpoint_backends.py::_make_backend }}

## Run it

```bash
python examples/notebook_50_checkpoint_backends.py        # the in-memory part only

# OCI Object Storage through its S3-compatible API; the credentials are
# an OCI Customer Secret Key pair created for your user.
export S3_ENDPOINT_URL=https://<namespace>.compat.objectstorage.<region>.oraclecloud.com
export S3_BUCKET=locus-checkpoints
export AWS_ACCESS_KEY_ID=<customer-secret-key-id>
export AWS_SECRET_ACCESS_KEY=<customer-secret-key>
python examples/notebook_50_checkpoint_backends.py
```

The in-memory part runs on the bundled scripted model and needs no
credentials. The rest needs an S3-compatible bucket: `S3_BUCKET` names
it, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` sign the requests,
and `S3_ENDPOINT_URL` selects any endpoint other than S3 itself. Without
the three required variables the script prints their names after the
in-memory part and exits cleanly, before any client is created, so it
makes no network call.

Any other S3-compatible endpoint works the same way, for example a
local MinIO:

```bash
export S3_ENDPOINT_URL=http://localhost:9000
export S3_BUCKET=locus-checkpoints
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
python examples/notebook_50_checkpoint_backends.py
```

## See also

- [Conversation memory](notebook_05_agent_memory.md) — the checkpointer
  contract on `MemoryCheckpointer`.
- [Oracle 26ai checkpointer](notebook_42_oracle_26ai_checkpointer.md) —
  the second shape, `oracle_checkpointer()`, driven by the agent loop.
- [Agent server](notebook_65_agent_server.md) — `RedisBackend` behind
  `StorageBackendAdapter` under an HTTP API.
- [Checkpointers](../concepts/checkpointers.md) — picking a backend, the
  two shapes, the capability descriptor and the per-backend time-travel
  table.
- [How-to: persist conversations](../how-to/persist-conversations.md) —
  `OCIBucketBackend` and the other native checkpointers.

## Output

Running it offline — no credentials, bundled mock model — prints the in-memory part only: a case's three checkpoints listed newest first with the step and parent of each, the fork taken at `day-2` ending differently from the original, the Mermaid history of both threads with the fork drawn as a branch (its generated first id shown as `fork_root`), and a turn re-run from the thread's first checkpoint descending from it; then the names of the S3 variables the storage part needs, which the capture environment does not set:

```text
============================================================
Notebook 50: Durable support-case state on S3-compatible storage
============================================================

=== Part 0: List, fork and re-run a case (in memory, no credentials) ===

case_tkt_4021 history (newest first):
  day-3  iteration=3  parent=day-2
  day-2  iteration=2  parent=day-1
  day-1  iteration=1  parent=None

forked 'case_tkt_4021' at 'day-2' -> 'case_tkt_4021-partial'
original still ends with: Day 3: refund issued in full.
fork ends with:           Day 3: partial refund; escalated to billing.

Mermaid, both threads:
graph LR
    subgraph thread_case_tkt_4021["case_tkt_4021"]
        direction LR
        cp_case_tkt_4021__day_1_1e3471dc["day-1<br/>iteration 1"]
        cp_case_tkt_4021__day_2_2dd8e41b["day-2<br/>iteration 2"]
        cp_case_tkt_4021__day_3_472b4b89["day-3<br/>iteration 3"]
    end
    subgraph thread_case_tkt_4021_partial_19a767fa["case_tkt_4021-partial"]
        direction LR
        cp_case_tkt_4021_partial_19a767fa__fork_root["fork_root<br/>iteration 2<br/>fork"]
        cp_case_tkt_4021_partial_19a767fa__alt_day_3_3c327834["alt-day-3<br/>iteration 3"]
    end
    cp_case_tkt_4021__day_2_2dd8e41b --> cp_case_tkt_4021__day_3_472b4b89
    cp_case_tkt_4021__day_1_1e3471dc --> cp_case_tkt_4021__day_2_2dd8e41b
    cp_case_tkt_4021_partial_19a767fa__fork_root --> cp_case_tkt_4021_partial_19a767fa__alt_day_3_3c327834
    cp_case_tkt_4021__day_2_2dd8e41b -.->|fork| cp_case_tkt_4021_partial_19a767fa__fork_root

case-4821 after re-running the second turn from the first checkpoint:
  checkpoints: 3
  newest descends from the first: True
  the thread now reads: ['Open ticket T-4821.', 'Hold the refund.']

Required environment variables not set; skipping the live demo so this file still runs cleanly in CI.

  - S3_BUCKET
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY

Create a bucket on OCI Object Storage (or any S3-compatible endpoint such as MinIO), then set the variables above and re-run.
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_50_checkpoint_backends.py`"

    ````python
    --8<-- "examples/notebook_50_checkpoint_backends.py"
    ````
