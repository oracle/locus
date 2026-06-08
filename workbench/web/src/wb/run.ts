/** Run lifecycle — orchestrates the click→subprocess→stream→output
 *  flow. Handles run-id capture, interrupt-event detection, and the
 *  pre-run UI flips (full-screen, busy pill, lock prev/next). */
import { runNotebookSource, type WorkbenchEvent } from "../api";
import { loadProvider } from "../settings";
import type { ProviderConfig, RunEvent } from "../types";
import { $ } from "./dom";
import { getEditorContent, setEditorContent } from "./editor";
import { promptInterruptResponse } from "./interrupt";
import { enterRunFullscreen } from "./layout";
import {
  appendError,
  appendOutput,
  clearOutput,
  endLiveStream,
} from "./output";
import { getCurrent, renderNavState } from "./notebooks";

let cancelRun: (() => void) | null = null;

export function installRunControls(): void {
  $<HTMLButtonElement>("#wb-run-btn").addEventListener("click", () => void runEdited());
  $<HTMLButtonElement>("#wb-stop-btn").addEventListener("click", stopRun);
  $<HTMLButtonElement>("#wb-reset-btn").addEventListener("click", () => {
    const cur = getCurrent();
    if (cur) {
      console.info("[wb/run] reset to original", cur.id);
      setEditorContent(cur.source);
    }
  });
  // #clear-btn was removed from the HTML; querySelector is the
  // tolerant lookup so the entire init chain doesn't abort if the
  // button is absent. (`$()` is strict and would throw.)
  const clearBtn = document.querySelector<HTMLButtonElement>("#clear-btn");
  clearBtn?.addEventListener("click", () => {
    clearOutput();
    $("#wb-output-pill").style.display = "none";
  });
}

function setRunning(running: boolean): void {
  $<HTMLButtonElement>("#wb-run-btn").style.display = running ? "none" : "inline-flex";
  $<HTMLButtonElement>("#wb-stop-btn").style.display = running ? "inline-flex" : "none";
  // Lock notebook navigation while a subprocess is in flight.
  const prev = $<HTMLButtonElement>("#wb-prev-btn");
  const next = $<HTMLButtonElement>("#wb-next-btn");
  if (running) {
    prev.disabled = true;
    next.disabled = true;
  } else {
    renderNavState();
  }
}

function stopRun(): void {
  console.info("[wb/run] user stop");
  cancelRun?.();
  setRunning(false);
  appendOutput("stopped by user", "error");
}

async function runEdited(): Promise<void> {
  const provider = loadProvider();
  if (!provider) {
    $("#wb-status").textContent = "set provider settings first.";
    console.warn("[wb/run] no provider configured");
    return;
  }
  const source = getEditorContent();
  if (!source.trim()) {
    $("#wb-status").textContent = "editor is empty.";
    console.warn("[wb/run] empty editor");
    return;
  }
  const cur = getCurrent();
  console.info("[wb/run] starting", { notebook: cur?.id, sourceLen: source.length });
  clearOutput();
  endLiveStream();
  const pill = $("#wb-output-pill");
  pill.style.display = "inline-flex";
  pill.className = "pill pill--busy";
  pill.innerHTML = `<span class="pill__dot"></span>running…`;
  enterRunFullscreen();
  setRunning(true);

  let runId: string | null = null;
  let stdoutLines = 0;
  let stderrLines = 0;

  cancelRun = runNotebookSource(
    source,
    provider as ProviderConfig,
    (e: WorkbenchEvent | RunEvent) => onEvent(e, () => runId, (id) => (runId = id)),
    () => {
      console.info("[wb/run] stream closed", { stdoutLines, stderrLines });
      endLiveStream();
      setRunning(false);
      cancelRun = null;
    },
  );

  function onEvent(
    e: WorkbenchEvent | RunEvent,
    getRunId: () => string | null,
    setRunId: (id: string) => void,
  ): void {
    const ev = e as WorkbenchEvent & RunEvent;
    if (ev.type === "runStarted" && (ev as { run_id?: string }).run_id) {
      const rid = (ev as { run_id: string }).run_id;
      setRunId(rid);
      console.info("[wb/run] runStarted", rid.slice(0, 8));
      return;
    }
    if (ev.type === "stdout" && ev.text?.startsWith("__LE__:")) {
      try {
        const inner = JSON.parse(ev.text.slice("__LE__:".length));
        if (inner.type === "InterruptEvent") {
          const rid = getRunId();
          console.info("[wb/run] interrupt", inner.payload);
          if (rid) promptInterruptResponse(rid, inner.payload);
        }
      } catch {
        /* fall through to standard output rendering */
      }
    }
    if (ev.type === "exit") {
      console.info("[wb/run] exit", ev.code);
      endLiveStream();
      appendOutput(`process exited with code ${ev.code}`, "exit");
      pill.className = ev.code === 0 ? "pill pill--up" : "pill pill--down";
      pill.innerHTML = `<span class="pill__dot"></span>exit ${ev.code} · ${stdoutLines} stdout · ${stderrLines} stderr`;
      return;
    }
    if (ev.type === "error") {
      console.error("[wb/run] error", ev.text);
      endLiveStream();
      appendError(ev.text);
      pill.className = "pill pill--down";
      pill.innerHTML = `<span class="pill__dot"></span>error`;
      return;
    }
    appendOutput(ev.text, ev.type as "stdout" | "stderr");
    if (ev.type === "stdout") stdoutLines++;
    else if (ev.type === "stderr") stderrLines++;
  }
}
