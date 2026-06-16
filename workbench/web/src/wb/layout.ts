/** Layout primitives: full-screen toggle (with output-only auto-mode
 *  during a run) and the draggable split divider between editor +
 *  output. */
import { requestEditorMeasure } from "./editor";
import { $ } from "./dom";

const SPLIT_KEY = "locus.workbench.split";

export function installFullscreenToggle(): void {
  const root = $<HTMLElement>("#workbench");
  const btn = $<HTMLButtonElement>("#wb-fullscreen-btn");

  const toggle = (auto: boolean) => {
    const willOpen = !root.classList.contains("wb--full");
    root.classList.toggle("wb--full", willOpen);
    if (auto) root.classList.add("wb--auto");
    else root.classList.remove("wb--auto");
    document.body.classList.toggle("body--full", willOpen);
    setTimeout(requestEditorMeasure, 0);
  };

  btn.addEventListener("click", () => {
    console.info("[wb/layout] fullscreen toggle (manual)");
    toggle(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && root.classList.contains("wb--full")) {
      console.info("[wb/layout] fullscreen exit (Escape)");
      root.classList.remove("wb--full", "wb--auto");
      document.body.classList.remove("body--full");
      setTimeout(requestEditorMeasure, 0);
    }
  });
}

/** Called by run.ts when the user hits Run — flips both wb--full +
 *  wb--auto so the output panel takes the full viewport. */
export function enterRunFullscreen(): void {
  const root = $<HTMLElement>("#workbench");
  root.classList.add("wb--full", "wb--auto");
  document.body.classList.add("body--full");
  setTimeout(requestEditorMeasure, 0);
}

export function installSplitResize(): void {
  const split = document.querySelector<HTMLElement>(".wb-split");
  const handle = document.querySelector<HTMLElement>("#wb-resize");
  if (!split || !handle) return;

  const saved = parseFloat(localStorage.getItem(SPLIT_KEY) ?? "");
  if (Number.isFinite(saved) && saved > 0.15 && saved < 0.85) {
    split.style.setProperty("--wb-left", `${saved}fr`);
    split.style.setProperty("--wb-right", `${1 - saved}fr`);
  }

  let startX = 0;
  let startLeftPx = 0;

  const onMove = (e: MouseEvent) => {
    const total = split.getBoundingClientRect().width;
    const dx = e.clientX - startX;
    const newLeft = Math.max(280, Math.min(total - 280, startLeftPx + dx));
    const ratio = newLeft / total;
    split.style.setProperty("--wb-left", `${ratio}fr`);
    split.style.setProperty("--wb-right", `${1 - ratio}fr`);
    requestEditorMeasure();
  };
  const onUp = () => {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    handle.classList.remove("wb-resize--dragging");
    document.body.style.cursor = "";
    const editorCard = split.children[0] as HTMLElement;
    const ratio = editorCard.getBoundingClientRect().width / split.getBoundingClientRect().width;
    localStorage.setItem(SPLIT_KEY, String(ratio));
    console.info("[wb/layout] split saved", { ratio: ratio.toFixed(3) });
  };
  handle.addEventListener("mousedown", (e: MouseEvent) => {
    startX = e.clientX;
    startLeftPx = (split.children[0] as HTMLElement).getBoundingClientRect().width;
    handle.classList.add("wb-resize--dragging");
    document.body.style.cursor = "col-resize";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    e.preventDefault();
  });
  handle.addEventListener("dblclick", () => {
    split.style.setProperty("--wb-left", "1fr");
    split.style.setProperty("--wb-right", "1fr");
    localStorage.setItem(SPLIT_KEY, "0.5");
    requestEditorMeasure();
    console.info("[wb/layout] split reset to 50/50");
  });
}
