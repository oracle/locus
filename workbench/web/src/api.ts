import type { ProviderConfig, Pattern, RunResponse } from "./types";

export type Notebook = {
  id: string;
  number: number;
  title: string;
  summary: string;
  filename: string;
  needs_stdin?: boolean;
  category?: string;
  category_order?: number;
};

export type NotebookDetail = Notebook & { source: string };

// Topic-progression headers the sidebars render. Same shape across
// the three catalogues — one record per declared category.
export type CategoryInfo = {
  id: string;
  name: string;
  description: string;
};

export async function listNotebooks(): Promise<Notebook[]> {
  const r = await fetch("/api/notebooks");
  if (!r.ok) throw new Error(`notebooks ${r.status}`);
  return (await r.json()) as Notebook[];
}

export async function listNotebookCategories(): Promise<CategoryInfo[]> {
  const r = await fetch("/api/notebooks/categories");
  if (!r.ok) throw new Error(`notebook categories ${r.status}`);
  return (await r.json()) as CategoryInfo[];
}

export async function getNotebook(id: string): Promise<NotebookDetail> {
  const r = await fetch(`/api/notebooks/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error(`notebook ${r.status}`);
  return (await r.json()) as NotebookDetail;
}

// ---------------------------------------------------------------------------
// Skills (AgentSkills.io SKILL.md packages under examples/skills/).
// ---------------------------------------------------------------------------

export type SkillSummary = {
  id: string;
  name: string;
  description: string;
  domain: string;
  allowed_tools: string[];
  license: string | null;
  path: string;
  category?: string;
};

export type SkillDetail = SkillSummary & {
  instructions: string;
  resources: string[];
};

export async function listSkills(): Promise<SkillSummary[]> {
  const r = await fetch("/api/skills");
  if (!r.ok) throw new Error(`skills ${r.status}`);
  return (await r.json()) as SkillSummary[];
}

export async function listSkillCategories(): Promise<CategoryInfo[]> {
  const r = await fetch("/api/skills/categories");
  if (!r.ok) throw new Error(`skill categories ${r.status}`);
  return (await r.json()) as CategoryInfo[];
}

export async function getSkill(id: string): Promise<SkillDetail> {
  const r = await fetch(`/api/skills/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error(`skill ${r.status}`);
  return (await r.json()) as SkillDetail;
}

// ---------------------------------------------------------------------------
// Router protocols (the 8 built-in orchestration shapes).
// ---------------------------------------------------------------------------

export type ProtocolSummary = {
  id: string;
  name: string;
  description: string;
  handles: string[];
  primary_for: string[];
  requires_capabilities: string[];
  risk_max: string;
  cost: string;
  latency: string;
  supports_streaming: boolean;
  supports_repair: boolean;
  category?: string;
  category_order?: number;
};

export type ProtocolDetail = ProtocolSummary & { runtime_shape: string };

export async function listProtocols(): Promise<ProtocolSummary[]> {
  const r = await fetch("/api/protocols");
  if (!r.ok) throw new Error(`protocols ${r.status}`);
  return (await r.json()) as ProtocolSummary[];
}

export async function listProtocolCategories(): Promise<CategoryInfo[]> {
  const r = await fetch("/api/protocols/categories");
  if (!r.ok) throw new Error(`protocol categories ${r.status}`);
  return (await r.json()) as CategoryInfo[];
}

export async function getProtocol(id: string): Promise<ProtocolDetail> {
  const r = await fetch(`/api/protocols/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error(`protocol ${r.status}`);
  return (await r.json()) as ProtocolDetail;
}

export type WorkbenchEvent =
  | { type: "stdout"; text: string }
  | { type: "stderr"; text: string }
  | { type: "exit"; code: number }
  | { type: "error"; text: string }
  | { type: "runStarted"; run_id: string };

export async function respondToInterrupt(runId: string, response: unknown): Promise<void> {
  const r = await fetch(`/api/notebooks/runs/${encodeURIComponent(runId)}/respond`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ response }),
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`respond ${r.status}: ${t}`);
  }
}

export function runNotebookSource(
  source: string,
  provider: ProviderConfig,
  onEvent: (e: WorkbenchEvent) => void,
  onClose: () => void,
): () => void {
  const ctrl = new AbortController();
  void (async () => {
    try {
      const r = await fetch("/api/notebooks/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // 8 minutes — long enough for the multi-protocol notebook 51
        // (5 prompts × multi-LLM-call protocols can stack to ~5min wall
        // time). Shorter notebooks still finish in well under a minute.
        body: JSON.stringify({ source, provider, timeout_seconds: 480 }),
        signal: ctrl.signal,
      });
      if (!r.ok || !r.body) {
        onEvent({ type: "error", text: `${r.status}: ${await r.text()}` });
        onClose();
        return;
      }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf("\n\n")) !== -1) {
          const block = buf.slice(0, nl);
          buf = buf.slice(nl + 2);
          for (const line of block.split("\n")) {
            if (!line.startsWith("data:")) continue;
            try {
              onEvent(JSON.parse(line.slice(5).trim()) as WorkbenchEvent);
            } catch {
              /* keepalive */
            }
          }
        }
      }
      onClose();
    } catch (err) {
      if ((err as Error).name !== "AbortError") onEvent({ type: "error", text: (err as Error).message });
      onClose();
    }
  })();
  return () => ctrl.abort();
}

// ---------------------------------------------------------------------------
// Pattern runner (the 8 pre-wired tulip patterns).
// ---------------------------------------------------------------------------

// A single Server-Sent event from the pattern streamer. The backend emits
// JSON objects per `data:` line — chunk events ({type, content, done}),
// a terminate event ({type, final_message}), and errors ({type, message}).
export type PatternStreamEvent = {
  type?: string;
  content?: string;
  done?: boolean;
  final_message?: string;
  message?: string;
  [key: string]: unknown;
};

export async function listPatterns(): Promise<Pattern[]> {
  const r = await fetch("/api/patterns");
  if (!r.ok) throw new Error(`patterns ${r.status}`);
  return (await r.json()) as Pattern[];
}

export async function runPattern(
  patternId: string,
  prompt: string,
  provider: ProviderConfig,
  options: { use_llm_picker?: boolean } = {},
): Promise<RunResponse> {
  const body: Record<string, unknown> = { prompt, provider };
  if (options.use_llm_picker !== undefined) {
    body.use_llm_picker = options.use_llm_picker;
  }
  const r = await fetch(`/api/run/${encodeURIComponent(patternId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`run ${r.status}: ${await r.text()}`);
  return (await r.json()) as RunResponse;
}

export function streamPattern(
  patternId: string,
  prompt: string,
  provider: ProviderConfig,
  onEvent: (ev: PatternStreamEvent) => void,
  onFinal: (reply: string) => void,
  onError: (msg: string) => void,
): () => void {
  const ctrl = new AbortController();
  void (async () => {
    let finalSent = false;
    try {
      const r = await fetch(`/api/run/${encodeURIComponent(patternId)}/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, provider }),
        signal: ctrl.signal,
      });
      if (!r.ok || !r.body) {
        onError(`stream ${r.status}: ${r.ok ? "no body" : await r.text()}`);
        return;
      }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf("\n\n")) !== -1) {
          const block = buf.slice(0, nl);
          buf = buf.slice(nl + 2);
          for (const line of block.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const raw = line.slice(5).trim();
            if (!raw) continue;
            let ev: PatternStreamEvent;
            try {
              ev = JSON.parse(raw) as PatternStreamEvent;
            } catch {
              onEvent({ content: raw });
              continue;
            }
            const type = ev.type ?? "";
            if (type.includes("Error") || ev.message) {
              onError(ev.message ?? "stream error");
            } else if (type.includes("Terminate") || ev.final_message !== undefined) {
              finalSent = true;
              onFinal(ev.final_message ?? "");
            } else {
              onEvent(ev);
            }
          }
        }
      }
      if (!finalSent) onFinal("");
    } catch (err) {
      if ((err as Error).name !== "AbortError") onError((err as Error).message);
    }
  })();
  return () => ctrl.abort();
}

export async function listModels(provider: ProviderConfig): Promise<{ models: string[]; error?: string }> {
  const r = await fetch("/api/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  if (!r.ok) throw new Error(`models ${r.status}`);
  const data = (await r.json()) as { models: string[]; error?: string };
  return data;
}
