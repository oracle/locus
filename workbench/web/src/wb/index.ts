/** Workbench public surface — wires the modules together and exposes
 *  the same `initWorkbench` / `refreshWorkbenchProvider` functions
 *  main.ts has always called. */
import { describeProvider, loadProvider } from "../settings";
import { $ } from "./dom";
import { ensureEditor } from "./editor";
import { installFullscreenToggle, installSplitResize } from "./layout";
import { installRunControls } from "./run";
import { installSidebarTabs } from "./protocols";
import { bootstrapNotebooks } from "./notebooks";

export { endLiveStream } from "./output";

export function initWorkbench(): void {
  console.info("[wb] init");
  ensureEditor("# pick a notebook from the sidebar to load its source");
  refreshWorkbenchProvider();
  void bootstrapNotebooks();
  installSplitResize();
  installFullscreenToggle();
  installRunControls();
  installSidebarTabs();
}

export function refreshWorkbenchProvider(): void {
  const provider = loadProvider();
  const pill = $("#wb-provider-pill");
  if (provider) {
    pill.className = "pill pill--up";
    pill.innerHTML = `<span class="pill__dot"></span>${describeProvider(provider)}`;
    pill.style.display = "inline-flex";
    console.info("[wb] provider", describeProvider(provider));
  } else {
    pill.className = "pill pill--down";
    pill.innerHTML = `<span class="pill__dot"></span>no provider`;
    pill.style.display = "inline-flex";
    console.warn("[wb] no provider configured");
  }
}
