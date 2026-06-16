import { test, expect, type Page } from "@playwright/test";

const COMPARTMENT =
  process.env.OCI_COMPARTMENT ??
  "ocid1.compartment.oc1..aaaaaaaa00000000000000000000000000000000000000000000000000000000";
const PROFILE = process.env.OCI_PROFILE ?? "MY_PROFILE";
const REGION = process.env.OCI_REGION ?? "us-chicago-1";

const OPENAI_KEY = process.env.OPENAI_API_KEY;
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;

async function configureOCI(page: Page, opts: { transport?: "auto" | "v1" | "sdk"; model?: string } = {}) {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByTestId("settings-btn").click();
  await page.getByTestId("cfg-provider").selectOption("oci-session");
  await page.getByTestId("cfg-profile").fill(PROFILE);
  await page.getByTestId("cfg-region").fill(REGION);
  await page.getByTestId("cfg-compartment").fill(COMPARTMENT);
  if (opts.transport) await page.getByTestId("cfg-transport").selectOption(opts.transport);
  const model = opts.model ?? "openai.gpt-5.5-2026-04-23";
  await expect(async () => {
    const m = await page.getByTestId("cfg-model").locator("option").allTextContents();
    expect(m.includes(model)).toBe(true);
  }).toPass({ timeout: 30_000 });
  await page.getByTestId("cfg-model").selectOption(model);
  await page.getByTestId("settings-save").click();
}

test.describe("locus workbench · main", () => {
  test("loads notebook catalog from BFF", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".app__brand-mark")).toHaveText("locus");
    await expect(page.getByTestId("side-notebooks").locator(".side__item").first()).toBeVisible({
      timeout: 30_000,
    });
    const items = page.getByTestId("side-notebooks").locator(".side__item");
    expect(await items.count()).toBeGreaterThan(20);
  });

  test("provider settings round-trip via localStorage", async ({ page }) => {
    await configureOCI(page);
    await page.reload();
    await page.getByTestId("settings-btn").click();
    await expect(page.getByTestId("cfg-profile")).toHaveValue(PROFILE);
    await expect(page.getByTestId("cfg-compartment")).toHaveValue(COMPARTMENT);
  });

  test("settings modal lists models for the live OCI connection", async ({ page }) => {
    await page.goto("/");
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await page.getByTestId("settings-btn").click();
    await expect.poll(
      async () => {
        const opts = await page.getByTestId("cfg-model").locator("option").allTextContents();
        return opts.some((m) => m.startsWith("openai.gpt-5.5-2026-04-23"));
      },
      { timeout: 30_000 },
    ).toBe(true);
  });

  test("runs an edited notebook subprocess against OCI gpt-5.5", async ({ page }) => {
    test.setTimeout(180_000);
    await configureOCI(page);
    await expect.poll(async () => page.evaluate(() => Boolean((window as any).__wb)), { timeout: 10_000 }).toBe(true);
    await expect.poll(
      async () => page.evaluate(() => ((window as any).__wb?.getSource?.() ?? "").length),
      { timeout: 15_000 },
    ).toBeGreaterThan(100);
    await page.evaluate(() => {
      const wb = (window as any).__wb as { setSource: (s: string) => void };
      wb.setSource("print('locus-workbench-marker-7421')\n");
    });
    await page.getByTestId("wb-run-btn").click();
    await expect(page.getByTestId("wb-output")).toContainText("locus-workbench-marker-7421", {
      timeout: 60_000,
    });
    await expect(page.getByTestId("wb-output")).toContainText(/exited with code 0/i, { timeout: 60_000 });
  });

  test("agent notebook emits Terminate event chip via the callback handler", async ({ page }) => {
    test.setTimeout(240_000);
    await configureOCI(page);
    await expect.poll(async () => page.evaluate(() => Boolean((window as any).__wb)), { timeout: 10_000 }).toBe(true);
    // Tiny notebook that builds an Agent and prints its reply. The
    // bootstrap injects a callback_handler so we should see a Terminate
    // chip in the output even though the notebook only calls run_sync.
    await page.evaluate(() => {
      const wb = (window as any).__wb as { setSource: (s: string) => void };
      wb.setSource(`from config import get_model
from locus.agent import Agent, AgentConfig
agent = Agent(config=AgentConfig(model=get_model(), system_prompt="Reply with 'pong' and nothing else.", max_iterations=2))
print(agent.run_sync("ping").message)
`);
    });
    await page.getByTestId("wb-run-btn").click();
    await expect(page.getByTestId("wb-output")).toContainText(/exited with code 0/i, { timeout: 120_000 });
    const chips = await page.locator(".wb-output .event__kind").allTextContents();
    expect(chips.some((c) => c.toLowerCase().includes("terminate"))).toBe(true);
  });

  test("theme toggle switches data-theme + persists in localStorage", async ({ page }) => {
    await page.goto("/");
    await page.evaluate(() => localStorage.removeItem("locus.workbench.theme"));
    await page.reload();
    const html = page.locator("html");
    // Click toggle, expect data-theme to flip and persist across reload.
    const before = await html.getAttribute("data-theme");
    await page.getByTestId("theme-btn").click();
    const after = await html.getAttribute("data-theme");
    expect(after).not.toBe(before);
    expect(["light", "dark"]).toContain(after);
    await page.reload();
    await expect(html).toHaveAttribute("data-theme", after as string);
  });

  test("split divider is draggable and persists", async ({ page }) => {
    await page.setViewportSize({ width: 1400, height: 900 });
    await page.goto("/");
    await page.evaluate(() => localStorage.removeItem("locus.workbench.split"));
    await page.reload();
    const editorCard = page.getByTestId("wb-editor-card");
    const handle = page.getByTestId("wb-resize");
    const before = (await editorCard.boundingBox())!.width;
    const hbox = (await handle.boundingBox())!;
    // Drag the divider 200px to the right — editor should grow.
    await page.mouse.move(hbox.x + hbox.width / 2, hbox.y + hbox.height / 2);
    await page.mouse.down();
    await page.mouse.move(hbox.x + hbox.width / 2 + 200, hbox.y + hbox.height / 2, { steps: 10 });
    await page.mouse.up();
    const after = (await editorCard.boundingBox())!.width;
    expect(after).toBeGreaterThan(before + 100);
    // Persisted across reload.
    await page.reload();
    const persisted = (await editorCard.boundingBox())!.width;
    expect(Math.abs(persisted - after)).toBeLessThan(24);
  });

  test("prev/next buttons walk the notebook catalog", async ({ page }) => {
    await page.goto("/");
    await expect.poll(async () => page.evaluate(() => Boolean((window as any).__wb)), { timeout: 10_000 }).toBe(true);
    // First notebook loaded (notebook 01); prev should be disabled.
    await expect(page.getByTestId("wb-prev-btn")).toBeDisabled();
    await expect(page.getByTestId("wb-next-btn")).toBeEnabled();
    const titleBefore = await page.locator("#wb-title").textContent();
    await page.getByTestId("wb-next-btn").click();
    await expect.poll(async () => page.locator("#wb-title").textContent(), { timeout: 10_000 }).not.toBe(titleBefore);
    // Now prev should be enabled too.
    await expect(page.getByTestId("wb-prev-btn")).toBeEnabled();
    // Step back, original title returns.
    await page.getByTestId("wb-prev-btn").click();
    await expect(page.locator("#wb-title")).toHaveText(titleBefore as string);
  });

  test("prev/next disabled while a run is in flight, re-enabled after", async ({ page }) => {
    await configureOCI(page);
    await expect.poll(async () => page.evaluate(() => Boolean((window as any).__wb)), { timeout: 10_000 }).toBe(true);
    // Sleep just long enough that the test can observe disabled state mid-run.
    await page.evaluate(() => {
      const wb = (window as any).__wb as { setSource: (s: string) => void };
      wb.setSource("import time\ntime.sleep(2)\nprint('done')\n");
    });
    // Step to notebook 02 first so both prev and next start enabled.
    await page.getByTestId("wb-next-btn").click();
    await expect(page.getByTestId("wb-prev-btn")).toBeEnabled();
    await expect(page.getByTestId("wb-next-btn")).toBeEnabled();
    // Refresh source to our sleep stub (selectNotebook just overwrote it).
    await page.evaluate(() => {
      const wb = (window as any).__wb as { setSource: (s: string) => void };
      wb.setSource("import time\ntime.sleep(2)\nprint('done')\n");
    });
    await page.getByTestId("wb-run-btn").click();
    // Mid-run both nav buttons must be disabled.
    await expect(page.getByTestId("wb-prev-btn")).toBeDisabled();
    await expect(page.getByTestId("wb-next-btn")).toBeDisabled();
    // After the subprocess finishes, prev should be enabled (we're on
    // notebook 02), next too.
    await expect(page.getByTestId("wb-output")).toContainText("done", { timeout: 30_000 });
    await expect(page.getByTestId("wb-prev-btn")).toBeEnabled();
    await expect(page.getByTestId("wb-next-btn")).toBeEnabled();
  });

  test("brand uses the locus mark + slogan from the docs site", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".app__brand-mark")).toHaveText("locus");
    await expect(page.locator(".app__brand-tag")).toContainText(/multi-agent reasoning orchestrator/i);
    await expect(page.locator(".app__brand-icon")).toBeVisible();
    // Oracle / OCI lockup on the right links to the GenAI page.
    const oracle = page.locator(".app__oracle");
    await expect(oracle).toBeVisible();
    await expect(oracle).toHaveAttribute("href", /oracle\.com.*generative-ai/i);
  });

  test("default theme is dark when no preference is saved", async ({ page }) => {
    await page.goto("/");
    await page.evaluate(() => localStorage.removeItem("locus.workbench.theme"));
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  });

  test("clicking Run auto-enters full-screen with output-only view", async ({ page }) => {
    await configureOCI(page);
    await expect.poll(async () => page.evaluate(() => Boolean((window as any).__wb)), { timeout: 10_000 }).toBe(true);
    await page.evaluate(() => {
      const wb = (window as any).__wb as { setSource: (s: string) => void };
      wb.setSource("print('autofs-marker')\n");
    });
    await page.getByTestId("wb-run-btn").click();
    // Workbench should immediately go full-screen with the editor hidden.
    await expect(page.getByTestId("wb-root")).toHaveClass(/wb--full/);
    await expect(page.getByTestId("wb-root")).toHaveClass(/wb--auto/);
    await expect(page.getByTestId("wb-editor-card")).toBeHidden();
    await expect(page.getByTestId("wb-output-card")).toBeVisible();
    await expect(page.getByTestId("wb-output")).toContainText("autofs-marker", { timeout: 60_000 });
    // Esc restores the editor.
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("wb-root")).not.toHaveClass(/wb--full/);
    await expect(page.getByTestId("wb-editor-card")).toBeVisible();
  });

  test("workbench full-screen toggles via icon button + Escape", async ({ page }) => {
    await page.goto("/");
    const root = page.getByTestId("wb-root");
    const editorCard = page.getByTestId("wb-editor-card");
    const outputCard = page.getByTestId("wb-output-card");
    await page.getByTestId("wb-fullscreen-btn").click();
    await expect(root).toHaveClass(/wb--full/);
    // Both cards must remain visible inside the full-screen workbench.
    await expect(editorCard).toBeVisible();
    await expect(outputCard).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(root).not.toHaveClass(/wb--full/);
  });

  test("SDK transport works with cohere.command-r-plus", async ({ page }) => {
    test.setTimeout(180_000);
    await configureOCI(page, { transport: "sdk", model: "cohere.command-r-plus-08-2024" });
    await expect(page.locator("#wb-provider-pill")).toContainText("cohere.command-r-plus-08-2024");
    await expect(page.locator("#wb-provider-pill")).toContainText("sdk");
  });
});

const openaiTest = OPENAI_KEY ? test : test.skip;
test.describe("locus workbench · OpenAI", () => {
  openaiTest("workbench runs against OpenAI gpt-5", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto("/");
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await page.getByTestId("settings-btn").click();
    await page.getByTestId("cfg-provider").selectOption("openai");
    await page.getByTestId("cfg-apikey").fill(OPENAI_KEY ?? "");
    await expect(async () => {
      const opts = await page.getByTestId("cfg-model").locator("option").allTextContents();
      expect(opts.includes("gpt-5")).toBe(true);
    }).toPass({ timeout: 10_000 });
    await page.getByTestId("cfg-model").selectOption("gpt-5");
    await page.getByTestId("settings-save").click();
    await expect.poll(async () => page.evaluate(() => Boolean((window as any).__wb)), { timeout: 10_000 }).toBe(true);
    await page.evaluate(() => {
      const wb = (window as any).__wb as { setSource: (s: string) => void };
      wb.setSource("print('openai-marker-9001')\n");
    });
    await page.getByTestId("wb-run-btn").click();
    await expect(page.getByTestId("wb-output")).toContainText("openai-marker-9001", { timeout: 60_000 });
  });
});

const anthropicTest = ANTHROPIC_KEY ? test : test.skip;
test.describe("locus workbench · Anthropic", () => {
  anthropicTest("workbench runs against Anthropic claude-sonnet-4-6", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto("/");
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await page.getByTestId("settings-btn").click();
    await page.getByTestId("cfg-provider").selectOption("anthropic");
    await page.getByTestId("cfg-apikey").fill(ANTHROPIC_KEY ?? "");
    await expect(async () => {
      const opts = await page.getByTestId("cfg-model").locator("option").allTextContents();
      expect(opts.includes("claude-sonnet-4-6")).toBe(true);
    }).toPass({ timeout: 10_000 });
    await page.getByTestId("cfg-model").selectOption("claude-sonnet-4-6");
    await page.getByTestId("settings-save").click();
    await expect.poll(async () => page.evaluate(() => Boolean((window as any).__wb)), { timeout: 10_000 }).toBe(true);
    await page.evaluate(() => {
      const wb = (window as any).__wb as { setSource: (s: string) => void };
      wb.setSource("print('anthropic-marker-3030')\n");
    });
    await page.getByTestId("wb-run-btn").click();
    await expect(page.getByTestId("wb-output")).toContainText("anthropic-marker-3030", { timeout: 60_000 });
  });
});
