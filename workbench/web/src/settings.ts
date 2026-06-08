import type { ProviderConfig, ProviderType } from "./types";


// API keys are session-only (never written to localStorage).
// Non-sensitive provider fields persist across reloads.
const PROVIDER_PREFS_KEY = "tulip.workbench.provider-prefs";

type ProviderPrefs = {
  provider?: string;
  model?: string;
  model_b?: string;
  model_c?: string;
};

function saveProviderPrefs(cfg: ProviderConfig): void {
  localStorage.setItem(
    PROVIDER_PREFS_KEY,
    JSON.stringify({
      provider: cfg.provider,
      model: cfg.model,
      model_b: cfg.model_b ?? "",
      model_c: cfg.model_c ?? "",
    } satisfies ProviderPrefs),
  );
}

let memoryProvider: ProviderConfig | null = null;

export function loadProvider(): ProviderConfig | null {
  if (memoryProvider) return memoryProvider;
  try {
    const raw = localStorage.getItem(PROVIDER_PREFS_KEY);
    if (!raw) return null;
    const p = JSON.parse(raw) as ProviderPrefs;
    if (p.provider !== "openai" && p.provider !== "anthropic") {
      return null;
    }
    return {
      provider: p.provider as ProviderConfig["provider"],
      model: p.model ?? defaultModelFor(p.provider as ProviderType),
      model_b: p.model_b ?? "",
      model_c: p.model_c ?? "",
    };
  } catch {
    return null;
  }
}

export function saveProvider(cfg: ProviderConfig): void {
  memoryProvider = cfg;
  saveProviderPrefs(cfg);
}

export function defaultModelFor(p: ProviderType): string {
  switch (p) {
    case "openai":
      return "gpt-4o";
    case "anthropic":
      return "claude-sonnet-4-6";
  }
}

/** A full prefill for a freshly-selected provider.
 *
 *  Defensive: if `p` is anything outside `ProviderType` (e.g. the empty
 *  string a <select> reports after a removed option was assigned to it),
 *  fall through to openai defaults rather than returning undefined.
 *  Returning undefined here cascades into a TypeError in
 *  `syncSettingsRows`. */
export function defaultsFor(p: ProviderType): ProviderConfig {
  switch (p) {
    case "openai":
      return { provider: "openai", model: "gpt-4o" };
    case "anthropic":
      return { provider: "anthropic", model: "claude-sonnet-4-6" };
    default:
      return { provider: "openai", model: "gpt-4o" };
  }
}

export function describeProvider(cfg: ProviderConfig): string {
  switch (cfg.provider) {
    case "openai":
      return `OpenAI · ${cfg.model}`;
    case "anthropic":
      return `Anthropic · ${cfg.model}`;
  }
}
