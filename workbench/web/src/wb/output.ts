/** Output-panel rendering — append plain stdout/stderr lines, parse
 *  __LE__: event lines into chips, accumulate ModelChunkEvent tokens
 *  into a live THINK chip. */
import { $ } from "./dom";

const LE_PREFIX = "__LE__:";

let liveThinkBody: HTMLElement | null = null;

function out(): HTMLElement {
  return $("#wb-output");
}

export function clearOutput(): void {
  out().innerHTML = "";
  liveThinkBody = null;
}

export function showEmptyState(): void {
  out().innerHTML = `<div class="wb-output__empty">Hit <strong>Run</strong> to execute this notebook against your configured provider. Output streams here.</div>`;
  liveThinkBody = null;
}

export function appendOutput(line: string, kind: "stdout" | "stderr" | "exit" | "error"): void {
  if (kind === "stdout" && line.startsWith(LE_PREFIX)) {
    try {
      const ev = JSON.parse(line.slice(LE_PREFIX.length)) as Record<string, unknown>;
      appendEvent(ev);
      return;
    } catch (err) {
      console.warn("[wb/output] bad __LE__ line", line.slice(0, 80), err);
    }
  }
  const span = document.createElement("span");
  span.className = `ln ln--${kind}`;
  span.textContent = `${line}\n`;
  out().appendChild(span);
  out().scrollTop = out().scrollHeight;
}

function openLiveThinkChip(): HTMLElement {
  if (liveThinkBody && liveThinkBody.isConnected) return liveThinkBody;
  const row = document.createElement("span");
  row.className = "ln ln--event ln--event--live";
  row.dataset.testid = "live-think";
  const tag = document.createElement("span");
  tag.className = "event__kind event__kind--think";
  tag.textContent = "Think";
  const body = document.createElement("span");
  body.className = "event__body";
  row.appendChild(tag);
  row.appendChild(body);
  row.appendChild(document.createTextNode("\n"));
  out().appendChild(row);
  liveThinkBody = body;
  return body;
}

function closeLiveThinkChip(): void {
  if (liveThinkBody?.parentElement) liveThinkBody.parentElement.classList.remove("ln--event--live");
  liveThinkBody = null;
}

export function endLiveStream(): void {
  closeLiveThinkChip();
}

export function appendEvent(ev: Record<string, unknown>): void {
  const kind = (ev.type as string) ?? "Event";
  if (kind === "ModelChunkEvent") {
    const piece = (ev.content as string | undefined) ?? "";
    if (!piece) return;
    const body = openLiveThinkChip();
    const span = document.createElement("span");
    span.className = "chunk-piece";
    span.textContent = piece;
    body.appendChild(span);
    out().scrollTop = out().scrollHeight;
    return;
  }
  // ThinkEvent without an open live chip means token streaming was off
  // and we should render the reasoning as a normal chip body. With a
  // live chip we just close it (the chunks already painted the text).
  if (kind === "ThinkEvent" && liveThinkBody?.isConnected) {
    closeLiveThinkChip();
    return;
  }
  closeLiveThinkChip();

  // InterruptEvent has its own UI affordance handled by run.ts; we
  // still drop a small chip so the run history shows the pause.
  let text = "";
  if (kind === "QueryEvent") {
    text = (ev.prompt as string) ?? "";
  } else if (kind === "InterruptEvent") {
    const p = ev.payload;
    text = typeof p === "string" ? p : JSON.stringify(p);
  } else if (kind === "ToolStartEvent" || kind === "ToolCompleteEvent") {
    text = (ev.tool_name as string) ?? "";
  } else if (kind === "ThinkEvent") {
    const reasoning = (ev.reasoning as string) ?? "";
    text = reasoning.length > 600 ? reasoning.slice(0, 597) + "…" : reasoning;
  } else if (kind === "TerminateEvent") {
    text = "";
  } else {
    const raw =
      (ev.tool_name as string) ??
      (ev.final_message as string) ??
      (ev.content as string) ??
      (ev.reasoning as string) ??
      (ev.message as string) ??
      "";
    text = raw.length > 80 ? raw.slice(0, 77) + "…" : raw;
  }

  const row = document.createElement("span");
  row.className = "ln ln--event";
  const tag = document.createElement("span");
  const kindClass =
    kind === "TerminateEvent"
      ? "event__kind--terminate"
      : kind === "QueryEvent" || kind === "InterruptEvent"
        ? "event__kind--query"
        : kind === "ThinkEvent"
          ? "event__kind--think"
          : kind.startsWith("Tool")
            ? "event__kind--tool"
            : "";
  tag.className = `event__kind ${kindClass}`;
  tag.textContent = kind === "InterruptEvent" ? "Interrupt" : kind.replace("Event", "");
  const body = document.createElement("span");
  body.className = "event__body";
  body.textContent = text;
  row.appendChild(tag);
  row.appendChild(body);
  row.appendChild(document.createTextNode("\n"));
  out().appendChild(row);
  out().scrollTop = out().scrollHeight;
}

export function appendFinal(text: string): void {
  const node = document.createElement("div");
  node.className = "reply__final";
  node.dataset.testid = "final-reply";
  node.textContent = text;
  out().appendChild(node);
}

export function appendError(msg: string): void {
  const node = document.createElement("div");
  node.className = "event";
  node.dataset.testid = "error";
  node.innerHTML = `<span class="event__kind event__kind--error">Error</span><span class="event__body"></span>`;
  (node.querySelector(".event__body") as HTMLElement).textContent = msg;
  out().appendChild(node);
}

export function appendInterruptForm(node: HTMLElement): void {
  out().appendChild(node);
  out().scrollTop = out().scrollHeight;
}
