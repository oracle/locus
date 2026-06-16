/** Backwards-compat shim — the workbench has been split into focused
 *  modules under `./wb/`. This file just re-exports the public surface
 *  so existing imports (`./workbench`) keep working unchanged. */
export { endLiveStream, initWorkbench, refreshWorkbenchProvider } from "./wb/index";
