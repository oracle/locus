/**
 * Per-notebook Anthropic sweep — one playwright test per non-stdin
 * notebook, run in parallel workers. Each test configures Anthropic in
 * its own browser context, then drives a single notebook through the
 * workbench UI and asserts exit 0.
 *
 * Catalog is fetched synchronously via curl at module load so we can
 * generate the test() entries at top level (no top-level-await dance).
 *
 *   ANTHROPIC_API_KEY=sk-ant-... \
 *   ANTHROPIC_MODEL=claude-haiku-4-5 \
 *     npx playwright test tests/all-anthropic.spec.ts \
 *       --headed --workers=4
 *
 * Skipped entirely if ANTHROPIC_API_KEY isn't set.
 */
import { test, expect, type Page } from "@playwright/test";
import { execSync } from "node:child_process";

const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
const MODEL = process.env.ANTHROPIC_MODEL ?? "claude-sonnet-4-6";
const MODEL_B = process.env.ANTHROPIC_MODEL_B ?? "";
const MODEL_C = process.env.ANTHROPIC_MODEL_C ?? "";
const PER_NOTEBOOK_MS = Number(process.env.PER_NOTEBOOK_MS ?? 360_000);

// Anthropic and are skipped from this sweep.
const OCI_ONLY = new Set<string>([
  "notebook_42_deepagent",        // >10 min with subagents; covered by CLI tests
  "notebook_50_audio_response",
  "notebook_51_audio_chat",
  // RAG notebooks call OpenAIEmbeddings internally.
  "notebook_23_rag_basics",
  "notebook_25_rag_agents",
]);
// Stagger gap between notebooks inside a worker. Anthropic tier 1 caps
// at 50 RPM for sonnet — with N workers each firing back-to-back we can
// flood. A small sleep gives the model a breather between notebooks.
const STAGGER_MS = Number(process.env.STAGGER_MS ?? 4_000);
const BFF = process.env.BFF_URL ?? "http://127.0.0.1:3101";

test.use({ video: "off", trace: "off", screenshot: "off" });

type CatalogEntry = { id: string; number: number; title: string; needs_stdin?: boolean };

const catalog: CatalogEntry[] = ANTHROPIC_KEY
  ? JSON.parse(execSync(`curl -sf ${BFF}/api/notebooks`).toString())
  : [];
const runnable = catalog.filter((t) => !t.needs_stdin && !OCI_ONLY.has(t.id));

async function configureAnthropic(page: Page): Promise<void> {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByTestId("settings-btn").click();
  await page.getByTestId("cfg-provider").selectOption("anthropic");
  await page.getByTestId("cfg-apikey").fill(ANTHROPIC_KEY ?? "");
  await expect(async () => {
    const opts = await page.getByTestId("cfg-model").locator("option").allTextContents();
    expect(opts.includes(MODEL)).toBe(true);
  }).toPass({ timeout: 15_000 });
  await page.getByTestId("cfg-model").selectOption(MODEL);
  // The B/C dropdowns are populated by the same setModelOptions pass
  // as A but the option list may not have flushed to the WebDriver
  // layer by the time selectOption fires — poll for the target option
  // to exist before attempting to set it.
  for (const [tid, value] of [
    ["cfg-model-b", MODEL_B] as const,
    ["cfg-model-c", MODEL_C] as const,
  ]) {
    if (!value) continue;
    await expect(async () => {
      const opts = await page.getByTestId(tid).locator("option").allTextContents();
      expect(opts.includes(value)).toBe(true);
    }).toPass({ timeout: 15_000 });
    await page.getByTestId(tid).selectOption(value);
  }
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

const guard = ANTHROPIC_KEY ? test : test.skip;

test.describe.configure({ mode: "parallel" });

for (const entry of runnable) {
  guard(`#${String(entry.number).padStart(2, "0")} ${entry.id}`, async ({ page }) => {
    const budget = SLOW_NOTEBOOKS.has(entry.id)
      ? PER_NOTEBOOK_MS * SLOW_MULTIPLIER
      : PER_NOTEBOOK_MS;
    test.setTimeout(budget + 60_000);
    if (STAGGER_MS > 0) await page.waitForTimeout(Math.random() * STAGGER_MS);
    await configureAnthropic(page);
    const { code, tail } = await runOne(page, entry.id);
    expect(code, tail).toBe(0);
  });
}
