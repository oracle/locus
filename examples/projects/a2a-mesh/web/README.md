# a2a-mesh-web — console for the A2A security mesh

A React + Vite + TypeScript front-end for the tulip
[a2a-mesh](../README.md) backend. Discovers running A2A peers via their
`AgentCard`, routes a query by capability tag, and renders the typed
event stream from `POST /a2a/stream` in real time.

Styled in a clean console design language — global header with a thin
red rule, sidebar workspace nav, white surface cards on a warm
off-white canvas, sand and sage accents. Same palette tokens used by
the tulip docs site.

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

1. **Discovers peers.** On mount, hits `GET /agent-card` on each
   configured proxy. Reachable peers turn green, unreachable ones turn
   red.
2. **Suggests a skill** based on the prompt. A query like
   `Is alert A-101 a true positive?` matches `alert_triage`; a query
   like `Enrich 198.51.100.7` matches `threat_intel`. The matching skill
   tag is highlighted on the peer card.
3. **Auto-selects** the first reachable peer that advertises the
   suggested skill — you can override by clicking another peer.
4. **Sends the query.** Either via `POST /a2a/invoke` (one shot) or
   `POST /a2a/stream` (SSE). Streamed events render as a typed log:
   `Tool`, `Think`, `ModelChunk`, `Terminate`, `Error`.

## File map

| File | Role |
|---|---|
| `src/App.tsx` | Single-page console — discovery + form + reply panel |
| `src/api.ts` | Wrappers around `GET /agent-card`, `POST /a2a/invoke`, SSE stream |
| `src/types.ts` | `AgentCard`, `Peer`, `StreamedEvent` |
| `src/styles/tulip.css` | tuliplabs palette + console layout primitives |
| `vite.config.ts` | Dev-server proxies to the two A2A services |

## Why a separate webapp?

The console is the place a SOC lead or on-call responder actually lands
to drive a triage flow. Backend agents stream typed events that already
render cleanly in a UI — A2A just makes the wire boundary explicit. This
is what one of those consoles looks like when you don't have to build
chrome.
