# The box runner

`python -m tulip.runner` (also installed as `tulip-runner`) is one agent's loop,
running inside an NVIDIA OpenShell sandbox for one run. A Tulip gateway starts
it as the sandbox's command. The gateway stays the control plane: it decides
every call, performs the tools a box must not, keeps the run's checkpoints and
writes every record. The runner holds no policy, no model key and no audit
chain of its own.

## Environment

| Variable | What it is |
| --- | --- |
| `TULIP_ADMIT_URL` | The gateway's base URL, reachable from the box. |
| `TULIP_ADMIT_TOKEN` | The box's workload token, sent as a bearer token. Never logged. |
| `TULIP_RUN_ID` | The run this box serves. |
| `<model.api_key_env>` | The OpenShell placeholder of the model's key, named by the manifest. |
| `<mcp[].credential_env>`, `<api[].credential_env>` | Placeholders of MCP and API connector credentials, named by the manifest. |
| `SHIV_ROOT` | Where `tulip-runner.pyz` unpacks itself (the image sets it). |

A placeholder is not a credential: the sandbox's proxy swaps it for the real
value on its bound host and nowhere else.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | The run finished, parked on a person, or the gateway said stop. |
| 2 | Refused: the manifest asks for a tool, a provider or a credential this runner does not have. |
| 1 | Error: no gateway configured or reachable, or the run failed. |

## What it does

1. `GET /internal/v1/runner/next` → `start`, `resume` or `stop`.
2. `GET /internal/v1/runner/manifest` → the `RunManifest`; one for another run
   is refused.
3. Builds the agent (`tulip.runner.build_runtime`). Tools, by `runs`:
   - `box`: the harness tools over `/sandbox`; `ask_user` pauses the run for an
     answer; `task` starts a subagent in the same box, as a child run
     (`POST /internal/v1/runs/{id}/children`).
   - `gateway`: `POST /internal/v1/runs/{id}/tools/{name}`, admitted and
     performed by the gateway.
   - `mcp:<id>`: `tools/call` on that mount, from the box.
   - `api:<id>`: one operation of that API connector, from the box. The
     request is `tulip.runner.api_request(connector, operation, arguments)`:
     the gateway computes the same request when it admits the call.
4. Every call but the gateway's own is admitted first
   (`POST /v1/admit {run_id, call_id, tool, arguments}`). An MCP or API call
   carries the admission's one-shot decision token as `x-tulip-decision`; the
   box guard checks it against the request on the wire.
5. Progress goes to `POST /internal/v1/runs/{id}/events` (`token`, `think`,
   `tool_start`, `tool_complete`, `harness.exec`, …), spooled to
   `/sandbox/.tulip/events.jsonl` through outages.
6. It ends with `POST /internal/v1/runs/{id}/result`:

   ```json
   {"status": "done", "final_message": "...", "stop_reason": "complete",
    "usage_reported": {"prompt_tokens": 20, "completion_tokens": 40}}
   {"status": "parked",
    "waiting": {"kind": "approval", "call_id": "c5", "tool": "bash", "approval_id": "ap-1"}}
   {"status": "parked", "waiting": {"kind": "question", "call_id": "q1", "question": "..."}}
   {"status": "refused", "error": "..."}
   {"status": "error", "error": "..."}
   ```

   `usage_reported` is the runner's own count; the box guard's metered count is
   the one that bills and budgets.

## Holds and parking

A call the gateway holds for a person is waited on up to
`budgets.park_ttl_s` (default 600 s). Approved in time, it is asked about again
with its `approval_id` and runs. Still undecided, the run pauses there: the
agent's state is checkpointed with the gateway
(`PUT /internal/v1/runs/{id}/checkpoint`), what it waits for is written to
`/sandbox/.tulip/parked.json` and reported, and the process exits 0.

When the decision comes the gateway starts the box again and `next` answers
`resume` with `{approval_id, decision}` (or `{answer}` for a question). The
runner restores the checkpoint and re-asks about the held call naming its
approval; the gateway admits only the exact call it approved.

## What a command cannot use

`bash` is admitted as a command, not as a model or connector call, so it must
not be able to make one:

- commands inherit the runner's environment **without** `TULIP_ADMIT_TOKEN`
  and without any credential placeholder;
- the runner marks itself non-dumpable (`prctl(PR_SET_DUMPABLE, 0)`), so a
  command running as the same user cannot read them out of
  `/proc/<runner>/environ` or its memory.

The image runs as uid 10001 with no capabilities. The box guard still meters
every model call per run, whoever sends it.

## Getting it

- `tulip-runner.pyz`, attached to each GitHub release, runs on any CPython
  3.12 on linux x86_64: `python3 tulip-runner.pyz`.
- `ghcr.io/tuliplabs-ai/tulip-runner:<version>`: `python:3.12-slim` with git,
  ripgrep and the `.pyz`; workspace `/sandbox`.
