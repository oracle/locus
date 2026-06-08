import type { AgentCard, StreamedEvent, EventKind } from "./types";

export async function fetchCard(proxy: string): Promise<AgentCard> {
  const resp = await fetch(`${proxy}/agent-card`);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  return { ...data, url: proxy };
}

export async function invoke(proxy: string, prompt: string): Promise<string> {
  const resp = await fetch(`${proxy}/a2a/invoke`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: prompt, metadata: {} }],
      metadata: {},
    }),
  });
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  const agent = (data.messages || []).filter((m: { role: string }) => m.role === "agent");
  return agent.length ? agent[agent.length - 1].content : "";
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
    resp = await fetch(`${proxy}/a2a/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "user", content: prompt, metadata: {} }],
        metadata: {},
      }),
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
      if (!payload) continue;
      try {
        const ev = JSON.parse(payload);
        const kind = (ev.type ?? "Other") as EventKind;
        const text =
          ev.tool_name ??
          ev.final_message ??
          ev.content ??
          ev.reasoning ??
          ev.message ??
          "";
        onEvent({
          id: crypto.randomUUID(),
          kind,
          raw: ev,
          text: typeof text === "string" ? text : JSON.stringify(text),
        });
        if (kind === "TerminateEvent" && typeof ev.final_message === "string") {
          final = ev.final_message;
        }
      } catch {
        /* skip non-JSON keepalives */
      }
    }
  }
  onDone(final);
}
