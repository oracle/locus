import { listModels } from "./api";
import {
  defaultModelFor,
  defaultsFor,
  loadProvider,
  saveProvider,
} from "./settings";
import type { ProviderConfig, ProviderType } from "./types";
import { initWorkbench, refreshWorkbenchProvider } from "./workbench";

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

const $ = <T extends HTMLElement = HTMLElement>(sel: string): T => {
  const el = document.querySelector<T>(sel);
  if (!el) throw new Error(`missing: ${sel}`);
  return el;
};

const settingsBtn = $<HTMLButtonElement>("#settings-btn");
const settingsModal = $("#settings-modal");
const settingsClose = $<HTMLButtonElement>("#settings-close");
const settingsCancel = $<HTMLButtonElement>("#settings-cancel");
const settingsSave = $<HTMLButtonElement>("#settings-save");
const cfgProvider = $<HTMLSelectElement>("#cfg-provider");
const cfgApiKey = $<HTMLInputElement>("#cfg-apikey");
const cfgModel = $<HTMLSelectElement>("#cfg-model");
const cfgModelB = $<HTMLSelectElement>("#cfg-model-b");
const cfgModelC = $<HTMLSelectElement>("#cfg-model-c");
const cfgModelStatus = $("#cfg-model-status");
const rowApiKey = $("#row-apikey");
const providerWarning = $("#provider-warning");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let provider: ProviderConfig | null = loadProvider();

// ---------------------------------------------------------------------------
// Settings modal
// ---------------------------------------------------------------------------

function fillFromConfig(cfg: ProviderConfig) {
  cfgProvider.value = cfg.provider;
  cfgApiKey.value = cfg.api_key ?? "";
  // Stash the desired B/C selections; setModelOptions() reads these
  // when it paints the dropdowns after credentials validate.
  pendingModelB = cfg.model_b ?? "";
  pendingModelC = cfg.model_c ?? "";
  // Leave the dropdown empty until refreshModels() either validates the
  // credentials and fetches the live list, or surfaces a hint that the
  // form is incomplete. Pre-painting a default would defeat the
  // "no models until validated" rule.
  clearModelOptions();
}

// Remembered B/C selections waiting on a model list. Preserved across
// the (initially empty) → (live list) transition so reopening Settings
// keeps the user's prior pick.
let pendingModelB = "";
let pendingModelC = "";

function setModelOptions(models: string[], selected?: string) {
  cfgModel.innerHTML = "";
  cfgModelB.innerHTML = "";
  cfgModelC.innerHTML = "";
  if (models.length === 0) {
    cfgModel.disabled = true;
    cfgModelB.disabled = true;
    cfgModelC.disabled = true;
    return;
  }
  cfgModel.disabled = false;
  cfgModelB.disabled = false;
  cfgModelC.disabled = false;
  if (selected && !models.includes(selected)) models = [selected, ...models];
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    cfgModel.appendChild(opt);
  }
  if (selected) cfgModel.value = selected;
  // B and C share the same model list, plus a leading "(use Model A)"
  // entry so the user can explicitly opt out without typing.
  for (const dd of [cfgModelB, cfgModelC]) {
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "(use Model A)";
    dd.appendChild(blank);
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      dd.appendChild(opt);
    }
  }
  cfgModelB.value = models.includes(pendingModelB) ? pendingModelB : "";
  cfgModelC.value = models.includes(pendingModelC) ? pendingModelC : "";
}

function clearModelOptions() {
  cfgModel.innerHTML = "";
  cfgModelB.innerHTML = "";
  cfgModelC.innerHTML = "";
  cfgModel.disabled = true;
  cfgModelB.disabled = true;
  cfgModelC.disabled = true;
}

// True when the form has enough credentials that an /api/models call is
// worth firing. We never auto-fetch on partial input — pasting a key
// of length 5 shouldn't burn a request to the provider, and we never
// want to render a curated/canned list before the key validates.
function hasEnoughCredentials(p: ProviderType): boolean {
  if (p === "openai") return cfgApiKey.value.startsWith("sk-") && cfgApiKey.value.length >= 30;
  if (p === "anthropic") return cfgApiKey.value.startsWith("sk-ant-") && cfgApiKey.value.length >= 30;
  return false;
}

function promptFor(p: ProviderType): string {
  if (p === "openai") return "enter API key to load models";
  if (p === "anthropic") return "enter API key to load models";
  return "";
}


let modelsRefreshSeq = 0;

async function refreshModels() {
  const seq = ++modelsRefreshSeq;
  const p = cfgProvider.value as ProviderType;
  cfgModelStatus.textContent = "fetching…";
  try {
    const cfg = {
      provider: p,
      model: cfgModel.value,
      api_key: cfgApiKey.value || undefined,
    };
    const result = await listModels(cfg as ProviderConfig);
    if (seq !== modelsRefreshSeq) return;
    if (result.error) {
      cfgModelStatus.textContent = result.error;
      return;
    }
    cfgModelStatus.textContent = `${result.models.length} available`;
    const want = cfgModel.value || defaultModelFor(p);
    setModelOptions(result.models, want);
  } catch (err) {
    if (seq !== modelsRefreshSeq) return;
    cfgModelStatus.textContent = `error: ${(err as Error).message}`;
  }
}

function syncSettingsRows() {
  const p = cfgProvider.value as ProviderType;
  rowApiKey.style.display = "flex";
  const def = defaultsFor(p);
  if (!cfgModel.value) setModelOptions([def.model], def.model);
}

function openSettings() {
  fillFromConfig(provider ?? defaultsFor("openai"));
  syncSettingsRows();
  settingsModal.classList.add("modal--open");
  void refreshModels();
}

function closeSettings() {
  settingsModal.classList.remove("modal--open");
}

function saveSettings() {
  const p = cfgProvider.value as ProviderType;
  const cfg: ProviderConfig = {
    provider: p,
    model: cfgModel.value || defaultModelFor(p),
  };
  if (cfgModelB.value) cfg.model_b = cfgModelB.value;
  if (cfgModelC.value) cfg.model_c = cfgModelC.value;
  if (p === "openai" || p === "anthropic") {
    cfg.api_key = cfgApiKey.value.trim();
    if (!cfg.api_key) {
      alert(`${p} provider needs an API key.`);
      return;
    }
  }
  provider = cfg;
  saveProvider(cfg);
  closeSettings();
  providerWarning.style.display = "none";
  refreshWorkbenchProvider();
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

settingsBtn.addEventListener("click", openSettings);
settingsClose.addEventListener("click", closeSettings);
settingsCancel.addEventListener("click", closeSettings);
settingsSave.addEventListener("click", saveSettings);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && settingsModal.classList.contains("modal--open")) {
    closeSettings();
  }
});
cfgProvider.addEventListener("change", () => {
  const p = cfgProvider.value as ProviderType;
  setModelOptions([defaultModelFor(p)], defaultModelFor(p));
  syncSettingsRows();
  void refreshModels();
});
let refreshTimer: ReturnType<typeof setTimeout> | null = null;
const queueRefresh = () => {
  if (refreshTimer) clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => void refreshModels(), 400);
};
cfgApiKey.addEventListener("input", queueRefresh);

// --- Theme toggle (light / dark) ---
const themeBtn = $<HTMLButtonElement>("#theme-btn");
const themeSun = $<HTMLElement>("#theme-icon-sun");
const themeMoon = $<HTMLElement>("#theme-icon-moon");
const THEME_KEY = "tulip.workbench.theme";

function applyTheme(t: "light" | "dark") {
  document.documentElement.setAttribute("data-theme", t);
  themeSun.style.display = t === "dark" ? "none" : "block";
  themeMoon.style.display = t === "dark" ? "block" : "none";
}

const savedTheme = localStorage.getItem(THEME_KEY) as "light" | "dark" | null;
// Default to dark — the workbench is for hands-on coding, dark is the
// natural fit. User can flip to light via the header toggle.
const initialTheme: "light" | "dark" = savedTheme ?? "dark";
applyTheme(initialTheme);
themeBtn.addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
});

// ---------------------------------------------------------------------------
// Mobile sidebar toggle
// ---------------------------------------------------------------------------

const sidebarBtn = document.querySelector<HTMLButtonElement>("#sidebar-btn");
const appSide = document.querySelector<HTMLElement>("#app-side");
const sidebarOverlay = document.querySelector<HTMLElement>("#sidebar-overlay");

function closeSidebar() {
  appSide?.classList.remove("app__side--open");
  sidebarOverlay?.classList.remove("sidebar-overlay--open");
}

sidebarBtn?.addEventListener("click", () => {
  const open = appSide?.classList.toggle("app__side--open");
  sidebarOverlay?.classList.toggle("sidebar-overlay--open", open ?? false);
});

sidebarOverlay?.addEventListener("click", closeSidebar);

// Auto-close when the user picks a sidebar item on mobile.
appSide?.addEventListener("click", (e) => {
  if (window.innerWidth > 768) return;
  if ((e.target as HTMLElement).closest(".side__item, .side__tab")) closeSidebar();
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

if (!provider) providerWarning.style.display = "block";
initWorkbench();
