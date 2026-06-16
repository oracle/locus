/**
 * Locus workbench BFF.
 *
 * Thin Node forwarder between the SPA and the Python pattern runner. The
 * BFF exists so the browser sees one same-origin endpoint surface, secrets
 * never leave the dev's machine, and we can swap backends behind the
 * stable /api shape later.
 */
import express from "express";
import type { Request, Response } from "express";

const PORT = Number(process.env.PORT ?? 3101);
const RUNNER = (process.env.RUNNER_URL ?? "http://127.0.0.1:8100").replace(/\/$/, "");

const app = express();
app.use(express.json({ limit: "1mb" }));

async function forward(req: Request, res: Response, init: RequestInit): Promise<void> {
  const path = req.originalUrl;
  try {
    const upstream = await fetch(`${RUNNER}${path}`, init);
    const text = await upstream.text();
    res.status(upstream.status).type(upstream.headers.get("content-type") ?? "application/json").send(text);
  } catch (err) {
    res.status(502).json({ error: `bff: ${(err as Error).message}` });
  }
}

app.get("/api/health", async (_req, res) => {
  try {
    const r = await fetch(`${RUNNER}/api/health`);
    const body = await r.json();
    res.json({ ok: true, runner: body });
  } catch (err) {
    res.status(502).json({ ok: false, error: `runner: ${(err as Error).message}` });
  }
});

app.get("/api/patterns", (req, res) => {
  void forward(req, res, { method: "GET" });
});

app.post("/api/models", (req, res) => {
  void forward(req, res, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req.body),
  });
});

app.post("/api/run/:pattern", (req, res) => {
  void forward(req, res, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req.body),
  });
});

async function streamForward(path: string, req: Request, res: Response) {
  let upstream: Response;
  try {
    upstream = (await fetch(`${RUNNER}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    })) as unknown as Response;
  } catch (err) {
    res.status(502).json({ error: `bff: ${(err as Error).message}` });
    return;
  }
  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text();
    res.status(upstream.status || 502).type("text/plain").send(text);
    return;
  }
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();
  const reader = (upstream.body as unknown as ReadableStream<Uint8Array>).getReader();
  const decoder = new TextDecoder();
  req.on("close", () => void reader.cancel().catch(() => undefined));
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    res.write(decoder.decode(value, { stream: true }));
  }
  res.end();
}

app.get("/api/notebooks", (req, res) => {
  void forward(req, res, { method: "GET" });
});
// /categories registered before /:tid so the parametric route doesn't match it.
app.get("/api/notebooks/categories", (req, res) => {
  void forward(req, res, { method: "GET" });
});
app.get("/api/notebooks/:tid", (req, res) => {
  void forward(req, res, { method: "GET" });
});
app.get("/api/skills", (req, res) => {
  void forward(req, res, { method: "GET" });
});
app.get("/api/skills/categories", (req, res) => {
  void forward(req, res, { method: "GET" });
});
app.get("/api/skills/:sid", (req, res) => {
  void forward(req, res, { method: "GET" });
});
app.get("/api/protocols", (req, res) => {
  void forward(req, res, { method: "GET" });
});
app.get("/api/protocols/categories", (req, res) => {
  void forward(req, res, { method: "GET" });
});
app.get("/api/protocols/:pid", (req, res) => {
  void forward(req, res, { method: "GET" });
});

// /api/notebooks/* — aliases of /api/notebooks/* exposed by the
// runner. The web frontend still calls /api/notebooks/* today, but the
// alias lets external tooling reach the catalog under the "Notebooks"
// name the docs use.
app.get("/api/notebooks", (req, res) => {
  void forward(req, res, { method: "GET" });
});
app.get("/api/notebooks/categories", (req, res) => {
  void forward(req, res, { method: "GET" });
});
app.get("/api/notebooks/:tid", (req, res) => {
  void forward(req, res, { method: "GET" });
});

// /api/database/test — Oracle 26ai connectivity probe used by the
// per-tab Database settings panel before launching a Run.
app.post("/api/database/test", (req, res) => {
  void forward(req, res, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req.body),
  });
});

// Telemetry SSE — these MUST stream; the existing `forward` helper buffers.
async function streamSseForward(path: string, _req: Request, res: Response): Promise<void> {
  const url = `${BFF_TARGET}${path}`;
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders?.();
  const upstream = await fetch(url, { method: "GET" });
  if (!upstream.ok || !upstream.body) {
    res.statusCode = upstream.status;
    res.end(`upstream ${upstream.status}: ${await upstream.text()}`);
    return;
  }
  const reader = upstream.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    res.write(value);
  }
  res.end();
}

// `/__stats` registered first so it doesn't get swallowed by the
// `:runId` parametric route below.
app.get("/api/events/__stats", (req, res) => {
  void forward(req, res, { method: "GET" });
});
app.get("/api/events", (req, res) => {
  void streamSseForward("/api/events", req, res);
});
app.get("/api/events/:runId", (req, res) => {
  void streamSseForward(`/api/events/${encodeURIComponent(req.params.runId)}`, req, res);
});
app.post("/api/notebooks/run", async (req, res) => {
  await streamForward("/api/notebooks/run", req, res);
});
app.post("/api/notebooks/runs/:runId/respond", (req, res) => {
  void forward(req, res, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req.body),
  });
});

app.post("/api/run/:pattern/stream", async (req, res) => {
  const path = `/api/run/${encodeURIComponent(req.params.pattern)}/stream`;
  let upstream: Response;
  try {
    upstream = (await fetch(`${RUNNER}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    })) as unknown as Response;
  } catch (err) {
    res.status(502).json({ error: `bff: ${(err as Error).message}` });
    return;
  }
  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text();
    res.status(upstream.status || 502).type("text/plain").send(text);
    return;
  }
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();
  const reader = (upstream.body as unknown as ReadableStream<Uint8Array>).getReader();
  const decoder = new TextDecoder();
  req.on("close", () => void reader.cancel().catch(() => undefined));
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    res.write(decoder.decode(value, { stream: true }));
  }
  res.end();
});

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`[bff] :${PORT} → runner ${RUNNER}`);
});
