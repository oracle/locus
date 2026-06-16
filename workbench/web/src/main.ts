import { listModels, testDatabase } from "./api";
import {
  defaultModelFor,
  defaultsFor,
  loadDatabase,
  loadProvider,
  saveDatabase,
  saveProvider,
} from "./settings";
import type { DatabaseConfig, ProviderConfig, ProviderType } from "./types";
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
const cfgProfile = $<HTMLInputElement>("#cfg-profile");
const cfgRegion = $<HTMLInputElement>("#cfg-region");
const cfgCompartment = $<HTMLInputElement>("#cfg-compartment");
const cfgTransport = $<HTMLSelectElement>("#cfg-transport");
const cfgProjectOcid = $<HTMLInputElement>("#cfg-project-ocid");
const rowApiKey = $("#row-apikey");
const rowProfile = $("#row-profile");
const rowRegion = $("#row-region");
const rowCompartment = $("#row-compartment");
const rowTransport = $("#row-transport");
const rowProjectOcid = $("#row-project-ocid");
const providerWarning = $("#provider-warning");

// Database panel
const cfgDbDsn = $<HTMLInputElement>("#cfg-db-dsn");
const cfgDbUser = $<HTMLInputElement>("#cfg-db-user");
const cfgDbPassword = $<HTMLInputElement>("#cfg-db-password");
const cfgDbWallet = $<HTMLInputElement>("#cfg-db-wallet");
const cfgDbWalletPassword = $<HTMLInputElement>("#cfg-db-wallet-password");
const cfgDbTest = $<HTMLButtonElement>("#cfg-db-test");
const cfgDbStatus = $("#cfg-db-status");

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
  cfgProfile.value = cfg.profile ?? "";
  cfgRegion.value = cfg.region ?? "us-chicago-1";
  cfgCompartment.value = cfg.compartment_id ?? "";
  cfgTransport.value = cfg.oci_transport ?? "v1";
  cfgProjectOcid.value = cfg.project_ocid ?? "";
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
  // OCI: needs a profile and a compartment OCID. The compartment OCID
  // shape is ``ocid1.compartment.oc1..<base64>`` — wait until it's
  // recognisably complete before hitting the control plane.
  if (p === "oci-session" || p === "oci-apikey") {
    return (
      !!cfgProfile.value.trim()
      && cfgCompartment.value.trim().startsWith("ocid1.compartment.")
      && cfgCompartment.value.trim().length >= 40
    );
  }
  return false;
}

function promptFor(p: ProviderType): string {
  if (p === "openai") return "enter API key to load models";
  if (p === "anthropic") return "enter API key to load models";
  if (p === "oci-session" || p === "oci-apikey") return "enter compartment OCID to load models";
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
      profile: cfgProfile.value || undefined,
      region: cfgRegion.value || undefined,
      compartment_id: cfgCompartment.value || undefined,
      oci_transport: cfgTransport.value as ProviderConfig["oci_transport"],
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
  const isOci = p.startsWith("oci");
  const isResponses = isOci && cfgTransport.value === "responses";
  rowApiKey.style.display = isOci ? "none" : "flex";
  rowProfile.style.display = isOci ? "flex" : "none";
  rowRegion.style.display = isOci ? "flex" : "none";
  rowCompartment.style.display = isOci ? "flex" : "none";
  rowTransport.style.display = isOci ? "flex" : "none";
  rowProjectOcid.style.display = isResponses ? "flex" : "none";
  const def = defaultsFor(p);
  if (!cfgModel.value) setModelOptions([def.model], def.model);
  if (isOci && !cfgProfile.value) cfgProfile.value = def.profile ?? "";
  if (isOci && !cfgRegion.value) cfgRegion.value = def.region ?? "";
  if (isOci && !cfgCompartment.value) cfgCompartment.value = def.compartment_id ?? "";
}

function openSettings() {
  // First-open default depends on where we're running. On localhost the user
  // typically has a populated `~/.oci/config`, so OCI session is the natural
  // pick. On any non-localhost host (e.g. a remote-served build) the OCI
  // session option is filtered out of the dropdown — handing it as the
  // default makes the <select> render an empty value, which then crashes
  // `syncSettingsRows` because `defaultsFor("")` falls through the switch
  // and returns `undefined`. Pick `openai` in that case so the dropdown
  // shows a real option from the start.
  const firstOpenDefault: ProviderType = isLocalhost ? "oci-session" : "openai";
  fillFromConfig(provider ?? defaultsFor(firstOpenDefault));
  fillDatabaseFromState();
  syncSettingsRows();
  settingsModal.classList.add("modal--open");
  void refreshModels();
}

function fillDatabaseFromState() {
  const db = loadDatabase();
  cfgDbDsn.value = db?.dsn ?? "";
  cfgDbUser.value = db?.user ?? "";
  cfgDbPassword.value = db?.password ?? "";
  cfgDbWallet.value = db?.wallet_location ?? "";
  cfgDbWalletPassword.value = db?.wallet_password ?? "";
  cfgDbStatus.textContent = "";
  cfgDbStatus.style.color = "var(--or-text-mute)";
}

function readDatabaseForm(): DatabaseConfig {
  return {
    dsn: cfgDbDsn.value.trim(),
    user: cfgDbUser.value.trim(),
    password: cfgDbPassword.value,
    wallet_location: cfgDbWallet.value.trim(),
    wallet_password: cfgDbWalletPassword.value,
  };
}

async function handleDatabaseTest() {
  const cfg = readDatabaseForm();
  if (!cfg.dsn || !cfg.user || !cfg.password) {
    cfgDbStatus.textContent = "Need DSN, user, and password.";
    cfgDbStatus.style.color = "var(--or-warning, #b54708)";
    return;
  }
  cfgDbStatus.textContent = "Testing…";
  cfgDbStatus.style.color = "var(--or-text-mute)";
  cfgDbTest.disabled = true;
  try {
    const r = await testDatabase(cfg);
    cfgDbStatus.textContent = r.detail;
    cfgDbStatus.style.color = r.ok ? "var(--or-success, #067647)" : "var(--or-error, #b42318)";
  } catch (e) {
    cfgDbStatus.textContent = `Request failed: ${e}`;
    cfgDbStatus.style.color = "var(--or-error, #b42318)";
  } finally {
    cfgDbTest.disabled = false;
  }
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
  } else {
    cfg.profile = cfgProfile.value.trim() || "DEFAULT";
    cfg.region = cfgRegion.value.trim() || "us-chicago-1";
    cfg.compartment_id = cfgCompartment.value.trim();
    cfg.oci_transport = (cfgTransport.value as ProviderConfig["oci_transport"]) ?? "v1";
    if (cfg.oci_transport === "responses" && cfgProjectOcid.value.trim()) {
      cfg.project_ocid = cfgProjectOcid.value.trim();
    }
  }
  provider = cfg;
  saveProvider(cfg);
  saveDatabase(readDatabaseForm());
  closeSettings();
  providerWarning.style.display = "none";
  refreshWorkbenchProvider();
}

// ---------------------------------------------------------------------------
// Provider options — context-aware
// localhost → OCI session only (profile on disk)
// remote    → API key providers only (no profile access)
// ---------------------------------------------------------------------------

const isLocalhost = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);

(function filterProviderOptions() {
  // OCI session token needs a local ~/.oci/config profile — only available on localhost.
  if (!isLocalhost) {
    Array.from(cfgProvider.options)
      .filter(opt => opt.value === "oci-session")
      .forEach(opt => opt.remove());
  }
})();

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

settingsBtn.addEventListener("click", openSettings);
settingsClose.addEventListener("click", closeSettings);
settingsCancel.addEventListener("click", closeSettings);
settingsSave.addEventListener("click", saveSettings);
cfgDbTest.addEventListener("click", () => void handleDatabaseTest());
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
cfgProfile.addEventListener("input", queueRefresh);
cfgRegion.addEventListener("input", queueRefresh);
cfgCompartment.addEventListener("input", queueRefresh);
cfgApiKey.addEventListener("input", queueRefresh);
cfgTransport.addEventListener("change", () => {
  // Show/hide the Project OCID input when the user picks Responses.
  syncSettingsRows();
  queueRefresh();
});

// --- Theme toggle (light / dark) ---
const themeBtn = $<HTMLButtonElement>("#theme-btn");
const themeSun = $<HTMLElement>("#theme-icon-sun");
const themeMoon = $<HTMLElement>("#theme-icon-moon");
const THEME_KEY = "locus.workbench.theme";

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
