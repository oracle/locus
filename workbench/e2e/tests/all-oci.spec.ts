/**
 * Per-notebook OCI v1 sweep — runs every non-stdin notebook through the
 * workbench against OCI Generative AI's /openai/v1 endpoint
 * (OCIOpenAIModel transport). Each test spawns its own browser context,
 * configures the OCI provider, then drives one notebook.
 *
 * All identifiers (profile, tenancy/compartment OCID) come from env so
 * nothing tenant-specific lives in this file:
 *
 *   OCI_PROFILE=YOUR_PROFILE \
 *   OCI_AUTH=apikey \
 *   OCI_REGION=us-chicago-1 \
 *   OCI_COMPARTMENT=ocid1.tenancy.oc1..xxxx \
 *   OCI_MODEL=openai.gpt-5.5-2026-04-23 \
 *   OCI_MODEL_B=xai.grok-4-fast-reasoning \
 *     npx playwright test tests/all-oci.spec.ts --workers=2
 *
 * Skipped entirely if OCI_PROFILE isn't set.
 */
import { test, expect, type Page } from "@playwright/test";
import { execSync } from "node:child_process";

const PROFILE = process.env.OCI_PROFILE;
const AUTH = (process.env.OCI_AUTH ?? "apikey").toLowerCase(); // apikey | session
const REGION = process.env.OCI_REGION ?? "us-chicago-1";
const COMPARTMENT = process.env.OCI_COMPARTMENT ?? "";
const TRANSPORT = process.env.OCI_TRANSPORT ?? "v1"; // v1 | auto | sdk
const MODEL = process.env.OCI_MODEL ?? "openai.gpt-5.5-2026-04-23";
const MODEL_B = process.env.OCI_MODEL_B ?? "";
const MODEL_C = process.env.OCI_MODEL_C ?? "";
const PER_NOTEBOOK_MS = Number(process.env.PER_NOTEBOOK_MS ?? 360_000);
const STAGGER_MS = Number(process.env.STAGGER_MS ?? 4_000);
const BFF = process.env.BFF_URL ?? "http://127.0.0.1:3101";

// Notebooks hardcoded against an OCI-only model (gpt-audio); they're
// fine here in principle but they expect an OCI session, not API key —
// keep them out of the sweep so a single auth shape covers everything.
const SKIP = new Set<string>([
  "notebook_50_audio_response",
  "notebook_51_audio_chat",
  // DeepAgent runs 4 parts with subagents — takes >10 min on any real model.
  // Covered by CLI tests; skipped here to keep the workbench sweep bounded.
  "notebook_42_deepagent",
  // Requires structured-output support; Cohere R-series returns 400 on
  // json_schema response_format. The guard exits 0 with a helpful message
  // only when LOCUS_MODEL_PROVIDER=oci+cohere — skip in the OCI sweep too.
  "notebook_15_reasoning_patterns",
  // RAG notebooks call OCIEmbeddings which defaults to api_key auth —
  // incompatible with session-token profiles (e.g. BOAT-OC1). The embed
  // client needs a separate API-key profile or LOCUS_OCI_AUTH_TYPE=security_token.
  "notebook_23_rag_basics",
  "notebook_25_rag_agents",
]);

test.use({ video: "off", trace: "off", screenshot: "off" });

type CatalogEntry = { id: string; number: number; title: string; needs_stdin?: boolean };

const catalog: CatalogEntry[] = PROFILE
  ? JSON.parse(execSync(`curl -sf ${BFF}/api/notebooks`).toString())
  : [];
const runnable = catalog.filter((t) => !t.needs_stdin && !SKIP.has(t.id));

async function configureOCI(page: Page): Promise<void> {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByTestId("settings-btn").click();
  const providerValue = AUTH === "session" ? "oci-session" : "oci-apikey";
  await page.getByTestId("cfg-provider").selectOption(providerValue);
  await page.getByTestId("cfg-profile").fill(PROFILE ?? "");
  await page.getByTestId("cfg-region").fill(REGION);
  await page.getByTestId("cfg-compartment").fill(COMPARTMENT);
  await page.getByTestId("cfg-transport").selectOption(TRANSPORT);
  await expect(async () => {
    const opts = await page.getByTestId("cfg-model").locator("option").allTextContents();
    expect(opts.includes(MODEL)).toBe(true);
  }).toPass({ timeout: 30_000 });
  await page.getByTestId("cfg-model").selectOption(MODEL);
  if (MODEL_B) await page.getByTestId("cfg-model-b").selectOption(MODEL_B);
  if (MODEL_C) await page.getByTestId("cfg-model-c").selectOption(MODEL_C);
  await page.getByTestId("settings-save").click();
}

async function runOne(page: Page, id: string): Promise<{ code: number; tail: string }> {
  await page.getByTestId(`notebook-${id}`).click();
  await expect
    .poll(
      () => page.evaluate(() => ((window as any).__wb?.getSource?.() ?? "").length),
      { timeout: 10_000 },
    )
    .toBeGreaterThan(50);
  await page.getByTestId("wb-run-btn").click();
  const output = page.getByTestId("wb-output");
  await expect(output).toContainText(/exited with code \d+/i, { timeout: PER_NOTEBOOK_MS });
  const text = (await output.textContent()) ?? "";
  const code = Number(text.match(/exited with code (\d+)/i)?.[1] ?? "-1");
  const tail = text.slice(-400).replace(/\s+/g, " ");
  return { code, tail };
}

const SLOW_NOTEBOOKS = new Set<string>([
  "notebook_42_deepagent",
  "notebook_52_cognitive_router",
  "notebook_57_research_workflow",
]);
const SLOW_MULTIPLIER = 3;

const guard = PROFILE ? test : test.skip;

test.describe.configure({ mode: "parallel" });

for (const entry of runnable) {
  guard(`#${String(entry.number).padStart(2, "0")} ${entry.id}`, async ({ page }) => {
    const budget = SLOW_NOTEBOOKS.has(entry.id)
      ? PER_NOTEBOOK_MS * SLOW_MULTIPLIER
      : PER_NOTEBOOK_MS;
    test.setTimeout(budget + 60_000);
    if (STAGGER_MS > 0) await page.waitForTimeout(Math.random() * STAGGER_MS);
    await configureOCI(page);
    const { code, tail } = await runOne(page, entry.id);
    expect(code, tail).toBe(0);
  });
}
