# a2a-mesh-web — console for the A2A security mesh

A React + Vite + TypeScript front-end for the locus
[a2a-mesh](../README.md) backend. Discovers running A2A (Agent-to-Agent)
peers via their A2A v1.0 `AgentCard`, routes a query by skill, and sends
it over JSON-RPC (`SendMessage`), or streams the task's events
(`SendStreamingMessage`) in real time.

Styled in a clean console design language — global header with a thin
red rule, sidebar workspace nav, white surface cards on a warm
off-white canvas, sand and sage accents. Same palette tokens used by
the locus docs site.

## Run

```bash
# In the parent directory, boot the two A2A services:
cd ..
make intel       # threat-intel peer  → http://127.0.0.1:8001
make triage      # SOC-triage peer    → http://127.0.0.1:8002

# In this directory:
npm install
npm run dev        # http://localhost:5173
```

The Vite dev server proxies `/api/intel/*` → `:8001` and
`/api/triage/*` → `:8002` (see `vite.config.ts`) so the React app
talks to both peers without CORS drama.

## What the UI does

1. **Discovers peers.** On mount, fetches
   `GET /.well-known/agent-card.json` on each configured proxy (falling
   back to the legacy `GET /agent-card` for pre-v1.0 peers). Reachable
   peers turn green, unreachable ones turn red; each card shows its skill
   ids and its A2A protocol version and interfaces.
2. **Suggests a skill** based on the prompt. A query like
   `Is alert A-101 a true positive?` matches `alert_triage`; a query
   like `Enrich 198.51.100.7` matches `threat_intel`. The matching skill
   tag is highlighted on the peer card.
3. **Auto-selects** the first reachable peer that advertises the
   suggested skill — you can override by clicking another peer.
4. **Sends the query** as A2A v1.0 JSON-RPC on `POST /`, with the
   `A2A-Version: 1.0` header. One shot uses `SendMessage` and shows the
   text of the task's last artifact. With streaming on,
   `SendStreamingMessage` frames render as a log: `Task`,
   `StatusUpdate`, `ArtifactUpdate`, `Error`.

## File map

| File | Role |
|---|---|
| `src/App.tsx` | Single-page console — discovery + form + reply panel |
| `src/api.ts` | A2A v1.0 client: Agent Card, `SendMessage`, `SendStreamingMessage` (SSE) |
| `src/types.ts` | `AgentCard`, `Peer`, `StreamedEvent` |
| `src/styles/locus.css` | locus palette + console layout primitives |
| `vite.config.ts` | Dev-server proxies to the two A2A services |

## Why a separate webapp?

The console is the place a SOC lead or on-call responder actually lands
to drive a triage flow. Backend agents stream typed events that already
render cleanly in a UI — A2A just makes the wire boundary explicit. This
is what one of those consoles looks like when you don't have to build
chrome.
