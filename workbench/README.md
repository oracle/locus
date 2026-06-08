# workbench — Tulip pattern playground

A self-contained 3-tier workbench: **vanilla TypeScript front-end** ↔
**Node BFF** ↔ **Python pattern runner built on the SDK**. Bring your own
provider credentials (OpenAI / Anthropic) — no external
dependencies beyond the model provider.

```
┌──────────────────────────────────────┐
│  workbench/web — vanilla TS + Vite   │  :5173
│  Pattern catalog · provider settings │
└────────────────┬─────────────────────┘
                 │ /api/*
                 ▼
┌──────────────────────────────────────┐
│  workbench/bff — Node Express        │  :3101
│  Thin proxy + same-origin surface    │
└────────────────┬─────────────────────┘
                 │ /api/*
                 ▼
┌──────────────────────────────────────┐
│  workbench/backend — FastAPI runner  │  :8100
│  One endpoint per tulip pattern      │
└──────────────────────────────────────┘
```

## What you can run today

Seven patterns wired so far. Each runs against the provider you set in
the UI. Adding a new one is ~20 lines: write a coroutine + register it
in `PATTERN_RUNNERS`.

| Notebook | Pattern | Notes |
|---|---|---|
| 01 | Basic agent | One Agent answers |
| 02 | Agent + tools | ReAct loop, two trivial tools |
| 13 | Structured output | Pydantic `output_schema` → typed Verdict |
| 17 | Orchestrator + specialists | Coordinator + 2 specialists |
| 25 | Composition (Sequential) | researcher → summariser |
| 42 | Map-reduce code review | `Send` fan-out to N reviewers, reduce |
| 43 | StateGraph (critic loop) | Writer → Critic with `allow_cycles` |

Notebook 42 (DeepAgent) now also includes a `part5_datastores` section
that exercises `create_deepagent(datastores=...)` against an in-memory
RAGRetriever — the same auto-wiring is the foundation for the
[`examples/projects/deep-research/`][dr] gist ports, which run the
identical API against pgvector, OpenSearch, and Qdrant. The workbench
surfaces the in-memory variant; the full multi-backend versions live as
standalone runnable demos.

[dr]: ../examples/projects/deep-research/

## Provider auth

The web UI's **Provider settings** modal accepts one of:

- **OpenAI** — `api_key` + `model` (defaults `gpt-4o`)
- **Anthropic** — `api_key` + `model` (defaults `claude-sonnet-4-6`)

Settings live in `localStorage` under `tulip.workbench.provider`. They're
sent on every request body to the backend; never persisted server-side.

## Run locally

```bash
# 1. Start the python runner (in a venv with tulip + the project deps).
cd workbench/backend
PYTHONPATH=../../src \
  uvicorn --app-dir . runner:app --host 127.0.0.1 --port 8100

# 2. Start the BFF.
cd ../bff && npm install && npm run dev

# 3. Start the web app.
cd ../web && npm install && npm run dev

# 4. Open http://localhost:5173 → Provider settings → run.
```

Or via the workbench `Makefile`:

```bash
make install
# in three panes:
make backend   # python runner
make bff       # node BFF
make web       # vite dev server
# fourth pane to run the e2e suite:
make e2e
```

## Run in Docker

Single-image build, all three tiers in one container:

```bash
# from the repo root
docker build -t tulip-workbench -f workbench/Dockerfile .
docker run --rm -p 5173:5173 -p 3101:3101 -p 8100:8100 tulip-workbench
# open http://localhost:5173
```

For OpenAI / Anthropic, paste the key into *Provider settings* once
the UI is up — no extra container args needed.

## Tests

`workbench/e2e/` — Playwright + chromium.

```bash
cd workbench/e2e && npm install && npx playwright install chromium
npm test
```

The suite drives the UI against whichever provider you configure.
Override the target model with env:

```bash
OPENAI_API_KEY=sk-... npm test
```

## Adding a new pattern

`workbench/backend/runner.py`:

1. Write `async def _run_<id>(req: RunRequest) -> RunResponse:` —
   build agents/graph from `req.provider` and call `_drive_agent` /
   `_drive_pipeline`.
2. Add an entry to the `PATTERNS` list (id, title, notebook #, summary).
3. Register the runner in `PATTERN_RUNNERS`.

The web app will pick it up automatically on next refresh — the
catalog is fetched live from `/api/patterns`.
