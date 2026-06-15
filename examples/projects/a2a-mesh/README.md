# a2a-mesh — multi-process Tulip security agents over A2A

A runnable example of three Tulip agents talking to each other across
process boundaries via the Agent-to-Agent (A2A) protocol — HTTP + SSE
with capability-based discovery via `AgentCard`. The mesh models a SOC
where a threat-intel service and an alert-triage service run as separate
peers, and an orchestrator routes each request to the right one.

The starting point was [`notebook_35_a2a_protocol.py`][t34]; this is
the project version with proper service boundaries, a Makefile, an
orchestrator that does real capability-tagged discovery, and an
integration test.

```
                 ┌──────────────────────────────────────────────┐
                 │  orchestrator (CLI)                          │
                 │  src/a2a_mesh/orchestrator.py                │
                 │                                              │
                 │  1. GET /agent-card on each peer             │
                 │  2. pick by skill tag                        │
                 │  3. POST /a2a/invoke or /a2a/stream          │
                 └────────┬───────────────────────┬─────────────┘
                          │ HTTP+SSE              │ HTTP+SSE
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
export TULIP_MODEL_PROVIDER=openai
export TULIP_MODEL_ID=gpt-4o
export OPENAI_API_KEY=sk-...
make intel
```

Anthropic works the same way (`TULIP_MODEL_PROVIDER=anthropic
ANTHROPIC_API_KEY=...`). The model factory is in
[`src/a2a_mesh/_model.py`](src/a2a_mesh/_model.py).

## What the demo shows

`make demo` runs `python -m a2a_mesh.orchestrator "Is alert A-101 a true positive?"`
which:

1. Calls `GET http://localhost:8001/agent-card` and
   `GET http://localhost:8002/agent-card` to discover both peers.
2. Picks the SOC-triage agent because the query names an alert id — the
   `alert_triage` skill matches.
3. Calls `POST /a2a/invoke` on the triage agent and prints the verdict.
4. As a follow-up, calls the threat-intel agent with `ioc_enrichment` to
   enrich the indicators behind the alert.

Both calls use `A2AClient`, so the wire format is the typed JSON
envelope from [`tulip.a2a.protocol`][a2a]. Streaming is also
demonstrated — pass `--stream` to `python -m a2a_mesh.orchestrator` and
events from the remote agent stream over SSE in real time.

## Why this isn't just `asyncio.gather`

A2A is for the case where each agent runs **as its own service**:
different teams, different deploy cadences, different tenancy, possibly
different runtimes (a Tulip agent calling a non-Tulip A2A peer or vice
versa) — e.g. a threat-intel platform and a SOC's triage stack owned by
two teams. Single-process mesh? Use [Orchestrator + Specialists][orch]
instead — this project is the cross-process equivalent.

## Tests

```bash
make test
```

`tests/test_mesh.py` boots both `A2AServer`s in-process via FastAPI's
`TestClient`, runs the orchestrator's discovery + delegation logic
against them, and asserts the reply round-trips. No real network.

## Files

| Path | What it is |
|---|---|
| `src/a2a_mesh/_model.py` | Shared model factory (mock by default, OpenAI / Anthropic via env) |
| `src/a2a_mesh/threat_intel.py` | Threat-intel `A2AServer` on `:8001` |
| `src/a2a_mesh/soc_triage.py` | SOC-triage `A2AServer` on `:8002` |
| `src/a2a_mesh/orchestrator.py` | CLI client — discovers + delegates by skill tag |
| `web/` | React + Vite UI — console-styled A2A dashboard (see [`web/README.md`](web/README.md)) |
| `pyproject.toml` | Hatch env — deps and the `sdk`/`intel`/`triage`/`demo`/`test` scripts |
| `Makefile` | `install`, `intel`, `triage`, `demo`, `test`, `mesh` (tmux) |
| `tests/test_mesh.py` | In-process integration test |

## Web console

A React + Vite UI ships in [`web/`](web/), styled in a clean console
design language — global header with a thin red rule, sidebar workspace
nav, white cards on warm off-white. Discovers both peers, auto-suggests
a skill from the prompt, and streams the typed event log from
`/a2a/stream`.

```bash
cd web
npm install
npm run dev          # http://localhost:5173
```

Vite proxies `/api/intel/*` and `/api/triage/*` to the two services,
so it works against the same `make intel` / `make triage` pair from
above.

[t34]: https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_35_a2a_protocol.py
[a2a]: https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/a2a/protocol.py
[orch]: https://tulipagents.ai/concepts/multi-agent/orchestrator/
