/**
 * Per-notebook OpenAI sweep — one playwright test per non-stdin
 * notebook, run in parallel workers. Each test configures OpenAI in
 * its own browser context, then drives a single notebook through the
 * workbench UI and asserts exit 0.
 *
 *   OPENAI_API_KEY=sk-proj-... \
 *   OPENAI_MODEL=gpt-5.5 \
 *   OPENAI_MODEL_B=gpt-5-nano \
 *     npx playwright test tests/all-openai.spec.ts \
 *       --headed --workers=3
 *
 * Skipped entirely if OPENAI_API_KEY isn't set.
 */
import { test, expect, type Page } from "@playwright/test";
import { execSync } from "node:child_process";

const OPENAI_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.OPENAI_MODEL ?? "gpt-5.5";
const MODEL_B = process.env.OPENAI_MODEL_B ?? "";
const MODEL_C = process.env.OPENAI_MODEL_C ?? "";
const PER_NOTEBOOK_MS = Number(process.env.PER_NOTEBOOK_MS ?? 360_000);

const OCI_ONLY = new Set<string>([
  "notebook_42_deepagent",        // >10 min with subagents; covered by CLI tests
  "notebook_50_audio_response",
  "notebook_51_audio_chat",
]);

// Stagger gap so N workers don't all hit OpenAI at the same instant.
const STAGGER_MS = Number(process.env.STAGGER_MS ?? 4_000);
const BFF = process.env.BFF_URL ?? "http://127.0.0.1:3101";

test.use({ video: "off", trace: "off", screenshot: "off" });

type CatalogEntry = { id: string; number: number; title: string; needs_stdin?: boolean };

const catalog: CatalogEntry[] = OPENAI_KEY
  ? JSON.parse(execSync(`curl -sf ${BFF}/api/notebooks`).toString())
  : [];
const runnable = catalog.filter((t) => !t.needs_stdin && !OCI_ONLY.has(t.id));

async function configureOpenAI(page: Page): Promise<void> {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByTestId("settings-btn").click();
  await page.getByTestId("cfg-provider").selectOption("openai");
  await page.getByTestId("cfg-apikey").fill(OPENAI_KEY ?? "");
  await expect(async () => {
    const opts = await page.getByTestId("cfg-model").locator("option").allTextContents();
    expect(opts.includes(MODEL)).toBe(true);
  }).toPass({ timeout: 15_000 });
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

const guard = OPENAI_KEY ? test : test.skip;

test.describe.configure({ mode: "parallel" });

for (const entry of runnable) {
  guard(`#${String(entry.number).padStart(2, "0")} ${entry.id}`, async ({ page }) => {
    const budget = SLOW_NOTEBOOKS.has(entry.id)
      ? PER_NOTEBOOK_MS * SLOW_MULTIPLIER
      : PER_NOTEBOOK_MS;
    test.setTimeout(budget + 60_000);
    if (STAGGER_MS > 0) await page.waitForTimeout(Math.random() * STAGGER_MS);
    await configureOpenAI(page);
    const { code, tail } = await runOne(page, entry.id);
    expect(code, tail).toBe(0);
  });
}
