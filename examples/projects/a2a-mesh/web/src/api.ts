import type { AgentCard, StreamedEvent, EventKind } from "./types";

// A2A v1.0 over JSON-RPC 2.0: the Agent Card at /.well-known/agent-card.json,
// SendMessage and SendStreamingMessage on POST /. Every request carries the
// A2A-Version header so the peer answers in the v1.0 wire shape.
const A2A_VERSION = "1.0";
const A2A_HEADERS = { "Content-Type": "application/json", "A2A-Version": A2A_VERSION };

type Part = { text?: string };
type Skill = string | { id: string };
type Interface = { protocolBinding?: string; protocolVersion?: string };

function partsText(parts: Part[] | undefined): string {
  return (parts ?? []).map((p) => p.text ?? "").join("");
}

// "TASK_STATE_WORKING" -> "working"
function shortState(state: unknown): string {
  return typeof state === "string" ? state.replace(/^TASK_STATE_/, "").toLowerCase() : "?";
}

function userMessage(prompt: string) {
  return { role: "ROLE_USER", parts: [{ text: prompt }], messageId: crypto.randomUUID() };
}

function rpcBody(method: string, prompt: string) {
  return JSON.stringify({
    jsonrpc: "2.0",
    id: crypto.randomUUID(),
    method,
    params: { message: userMessage(prompt) },
  });
}

export async function fetchCard(proxy: string): Promise<AgentCard> {
  let resp = await fetch(`${proxy}/.well-known/agent-card.json`, {
    headers: { "A2A-Version": A2A_VERSION },
  });
  // Pre-v1.0 peers only serve the legacy path.
  if (resp.status === 404) resp = await fetch(`${proxy}/agent-card`);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  const skills = ((data.skills ?? []) as Skill[]).map((s) => (typeof s === "string" ? s : s.id));
  const interfaces = ((data.supportedInterfaces ?? []) as Interface[]).map(
    (i) => `${i.protocolBinding ?? "?"} v${i.protocolVersion ?? "?"}`,
  );
  return {
    name: data.name,
    description: data.description,
    skills,
    url: proxy,
    protocolVersion: data.protocolVersion,
    interfaces,
  };
}

// SendMessage: one round trip; the reply is the text of the task's last artifact.
export async function send(proxy: string, prompt: string): Promise<string> {
  const resp = await fetch(`${proxy}/`, {
    method: "POST",
    headers: A2A_HEADERS,
    body: rpcBody("SendMessage", prompt),
  });
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  const env = await resp.json();
  if (env.error) throw new Error(`${env.error.code}: ${env.error.message}`);
  const task = env.result?.task;
  const artifacts = (task?.artifacts ?? []) as { parts?: Part[] }[];
  if (!artifacts.length) return `(no reply; task is ${shortState(task?.status?.state)})`;
  return partsText(artifacts[artifacts.length - 1].parts);
}

// One SendStreamingMessage frame -> one UI event.
function toEvent(env: Record<string, unknown>): { kind: EventKind; text: string; reply?: string } {
  if (env.error) {
    const error = env.error as { code?: number; message?: string };
    return { kind: "Error", text: `${error.code}: ${error.message}` };
  }
  const result = (env.result ?? {}) as Record<string, any>;
  if (result.task) {
    return { kind: "Task", text: `${result.task.id} · ${shortState(result.task.status?.state)}` };
  }
  if (result.statusUpdate) {
    const status = result.statusUpdate.status ?? {};
    const note = partsText(status.message?.parts);
    return { kind: "StatusUpdate", text: shortState(status.state) + (note ? `: ${note}` : "") };
  }
  if (result.artifactUpdate) {
    const text = partsText(result.artifactUpdate.artifact?.parts);
    return { kind: "ArtifactUpdate", text, reply: text };
  }
  if (result.message) {
    return { kind: "Message", text: partsText(result.message.parts) };
  }
  return { kind: "Other", text: JSON.stringify(result) };
}

export async function stream(
  proxy: string,
  prompt: string,
  onEvent: (e: StreamedEvent) => void,
  onDone: (final: string) => void,
  onError: (msg: string) => void,
): Promise<void> {
  let resp: Response;
  try {
    resp = await fetch(`${proxy}/`, {
      method: "POST",
      headers: { ...A2A_HEADERS, Accept: "text/event-stream" },
      body: rpcBody("SendStreamingMessage", prompt),
    });
  } catch (err) {
    onError(`network: ${(err as Error).message}`);
    return;
  }
  if (!resp.ok || !resp.body) {
    onError(`${resp.status} ${resp.statusText}`);
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let final = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let nl: number;
    while ((nl = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;
      try {
        const env = JSON.parse(payload);
        const { kind, text, reply } = toEvent(env);
        onEvent({ id: crypto.randomUUID(), kind, raw: env, text });
        if (reply) final = reply;
      } catch {
        /* skip non-JSON keepalives */
      }
    }
  }
  onDone(final);
}
