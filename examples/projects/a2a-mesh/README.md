# a2a-mesh — multi-process Locus security agents over A2A

A runnable example of Locus agents talking to each other across
process boundaries via the Agent-to-Agent (A2A) protocol v1.0 —
JSON-RPC over HTTP + SSE, with capability-based discovery via the
`AgentCard`. The mesh models a SOC
where a threat-intel service and an alert-triage service run as separate
peers, and an orchestrator routes each request to the right one.

The starting point was [`notebook_30_a2a_protocol.py`][t34]; this is
the project version with proper service boundaries, a Makefile, an
orchestrator that does real capability-tagged discovery, and an
integration test.

```
                 ┌──────────────────────────────────────────────┐
                 │  orchestrator (CLI)                          │
                 │  src/a2a_mesh/orchestrator.py                │
                 │                                              │
                 │  1. GET /.well-known/agent-card.json         │
                 │  2. pick by skill id                         │
                 │  3. SendMessage / SendStreamingMessage       │
                 └────────┬───────────────────────┬─────────────┘
                          │ JSON-RPC + SSE        │ JSON-RPC + SSE
                          ▼                       ▼
        ┌──────────────────────────┐  ┌──────────────────────────┐
        │  threat-intel agent      │  │  soc-triage agent        │
        │  :8001                   │  │  :8002                   │
        │  skills:                 │  │  skills:                 │
        │   threat_intel,          │  │   alert_triage,          │
        │   ioc_enrichment         │  │   severity_scoring       │
        │                          │  │                          │
        │  A2AServer wraps an      │  │  A2AServer wraps an      │
        │  Agent + tools.          │  │  Agent + tools.          │
        └──────────────────────────┘  └──────────────────────────┘
```

## Quick start

```bash
cd examples/projects/a2a-mesh
make install            # hatch env + published SDK (`hatch run sdk-local`
                        # runs against this repo's checkout instead)

# In three separate terminals (or use `make mesh` for tmux):
make intel              # boots the threat-intel peer on :8001
make triage             # boots the SOC-triage peer on :8002
make demo               # discovers + queries both, prints the answer
```

By default the agents run with the bundled `MockModel` (no creds, no
network calls to a model provider). To run live, set the provider
before booting each service:

```bash
export LOCUS_MODEL_PROVIDER=openai
export LOCUS_MODEL_ID=gpt-4o
export OPENAI_API_KEY=sk-...
make intel
```

Anthropic works the same way (`LOCUS_MODEL_PROVIDER=anthropic
ANTHROPIC_API_KEY=...`). The model factory is in
[`src/a2a_mesh/_model.py`](src/a2a_mesh/_model.py).

## What the demo shows

`make demo` runs `python -m a2a_mesh.orchestrator "Is alert A-101 a true positive?"`
which:

1. Fetches the Agent Card of both peers
   (`GET /.well-known/agent-card.json` on `:8001` and `:8002`) and prints
   each one's skills, A2A `protocolVersion` and `supportedInterfaces`.
2. Picks the SOC-triage agent because the query names an alert id — the
   `alert_triage` skill id matches.
3. Sends the query with the A2A v1.0 `SendMessage` JSON-RPC method and
   prints the verdict, read from the returned task's artifacts.

Every call goes through `A2AClient` from [`locus.a2a.protocol`][a2a].
Pass `--stream` to use `SendStreamingMessage` instead: the task's
lifecycle (`task`, `status-update`, `artifact-update`) streams over SSE
as it happens. `--skill <id>` forces the route.

The peers still serve the pre-v1.0 flat endpoints for older clients.
Pass `--legacy` (with or without `--stream`) to call `/a2a/invoke` and
`/a2a/stream` instead, which shows the backwards compatibility:

```bash
python -m a2a_mesh.orchestrator --legacy "Is alert A-101 a true positive?"
python -m a2a_mesh.orchestrator --legacy --stream "Enrich 198.51.100.7."
```

## Why this isn't just `asyncio.gather`

A2A is for the case where each agent runs **as its own service**:
different teams, different deploy cadences, different tenancy, possibly
different runtimes (a Locus agent calling a non-Locus A2A peer or vice
versa) — e.g. a threat-intel platform and a SOC's triage stack owned by
two teams. Single-process mesh? Use [Orchestrator + Specialists][orch]
instead — this project is the cross-process equivalent.

## Tests

```bash
make test
```

`tests/test_mesh.py` boots both `A2AServer`s in-process via FastAPI's
`TestClient` and checks the v1.0 Agent Card, `SendMessage` and
`SendStreamingMessage`, the legacy endpoints, and the orchestrator's
routing on the skill ids of a v1.0 card. No real network.

## Files

| Path | What it is |
|---|---|
| `src/a2a_mesh/_model.py` | Shared model factory (mock by default, OpenAI / Anthropic via env) |
| `src/a2a_mesh/threat_intel.py` | Threat-intel `A2AServer` on `:8001` |
| `src/a2a_mesh/soc_triage.py` | SOC-triage `A2AServer` on `:8002` |
| `src/a2a_mesh/orchestrator.py` | CLI client — discovers + delegates by skill id (A2A v1.0, `--legacy` for the flat endpoints) |
| `web/` | React + Vite UI — console-styled A2A dashboard (see [`web/README.md`](web/README.md)) |
| `pyproject.toml` | Hatch env — deps and the `sdk`/`intel`/`triage`/`demo`/`test` scripts |
| `Makefile` | `install`, `intel`, `triage`, `demo`, `test`, `mesh` (tmux) |
| `tests/test_mesh.py` | In-process integration test |

## Web console

A React + Vite UI ships in [`web/`](web/), styled in a clean console
design language — global header with a thin red rule, sidebar workspace
nav, white cards on warm off-white. Discovers both peers from their
v1.0 Agent Cards, auto-suggests a skill from the prompt, and sends it
with `SendMessage` or streams the task events of
`SendStreamingMessage`.

```bash
cd web
npm install
npm run dev          # http://localhost:5173
```

Vite proxies `/api/intel/*` and `/api/triage/*` to the two services,
so it works against the same `make intel` / `make triage` pair from
above.

[t34]: https://github.com/oracle/locus/blob/main/examples/notebook_30_a2a_protocol.py
[a2a]: https://github.com/oracle/locus/blob/main/src/locus/a2a/protocol.py
[orch]: https://locusagents.ai/concepts/multi-agent/orchestrator/
