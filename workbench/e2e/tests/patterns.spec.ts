import { test, expect, type Page } from "@playwright/test";
const BFF = process.env.BFF_URL ?? "http://127.0.0.1:3101";
type P = { id: string; title: string; streamable: boolean };

async function fresh(page: Page) {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
}

/** Navigate to the Patterns tab and wait for it to become active. */
async function openPatternsTab(page: Page) {
  await fresh(page);
  await expect(page.getByTestId("side-tab-patterns")).toBeVisible({ timeout: 10_000 });
  await page.getByTestId("side-tab-patterns").click();
  await expect(page.getByTestId("side-tab-patterns")).toHaveAttribute("aria-selected", "true");
}

test.use({ video: "off", trace: "off" });
test.describe("workbench · Patterns tab", () => {
  test("tab visible and selectable", async ({ page }) => {
    await fresh(page);
    await expect(page.getByTestId("side-tab-patterns")).toBeVisible();
    await page.getByTestId("side-tab-patterns").click();
    await expect(page.getByTestId("side-tab-patterns")).toHaveAttribute("aria-selected", "true");
    await expect(page.getByTestId("side-tab-notebooks")).toHaveAttribute("aria-selected", "false");
    await expect(page.getByTestId("side-tab-skills")).toHaveAttribute("aria-selected", "false");
    await expect(page.getByTestId("side-tab-protocols")).toHaveAttribute("aria-selected", "false");
  });

  test("sidebar lists patterns from BFF", async ({ page, request }) => {
    const catalog = (await (await request.get(`${BFF}/api/patterns`)).json()) as P[];
    expect(catalog.length).toBeGreaterThanOrEqual(7);
    await openPatternsTab(page);
    // Wait for items to populate (bootstrapPatterns is async)
    await expect(page.getByTestId("side-patterns").locator(".side__item").first()).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("side-patterns").locator(".side__item")).toHaveCount(catalog.length);
    for (const p of catalog) await expect(page.getByTestId(`pattern-${p.id}`)).toBeVisible();
  });

  test("clicking pattern shows detail view", async ({ page, request }) => {
    const first = ((await (await request.get(`${BFF}/api/patterns`)).json()) as P[])[0];
    await openPatternsTab(page);
    await expect(page.getByTestId("side-patterns").locator(".side__item").first()).toBeVisible({ timeout: 15_000 });
    await page.getByTestId(`pattern-${first.id}`).click();
    await expect(page.getByTestId("patterns-view")).toBeVisible();
    await expect(page.getByTestId("wb-root")).toBeHidden();
    await expect(page.locator("#pattern-title")).toHaveText(first.title);
    await expect(page.getByTestId("pattern-run-btn")).toBeEnabled();
  });

  test("tab round-trip preserves selection", async ({ page, request }) => {
    const first = ((await (await request.get(`${BFF}/api/patterns`)).json()) as P[])[0];
    await openPatternsTab(page);
    await expect(page.getByTestId("side-patterns").locator(".side__item").first()).toBeVisible({ timeout: 15_000 });
    await page.getByTestId(`pattern-${first.id}`).click();
    await expect(page.locator("#pattern-title")).toHaveText(first.title);
    await page.getByTestId("side-tab-protocols").click();
    await expect(page.getByTestId("protocols-view")).toBeVisible();
    await page.getByTestId("side-tab-patterns").click();
    await expect(page.getByTestId("patterns-view")).toBeVisible();
    await expect(page.locator("#pattern-title")).toHaveText(first.title);
  });

  test("run panel DOM wired", async ({ page, request }) => {
    const first = ((await (await request.get(`${BFF}/api/patterns`)).json()) as P[])[0];
    await openPatternsTab(page);
    await expect(page.getByTestId("side-patterns").locator(".side__item").first()).toBeVisible({ timeout: 15_000 });
    await page.getByTestId(`pattern-${first.id}`).click();
    await expect(page.getByTestId("pattern-run-btn")).toBeEnabled();
    await expect(page.getByTestId("pattern-stop-btn")).toBeHidden();
    await expect(page.getByTestId("pattern-output")).toBeHidden();
    await expect(page.getByTestId("pattern-error")).toBeHidden();
  });

  test("memory_manager listed", async ({ page, request }) => {
    const catalog = (await (await request.get(`${BFF}/api/patterns`)).json()) as P[];
    const has = catalog.some((p) => p.id === "memory_manager");
    test.skip(!has, "Runner needs restart to include memory_manager");
    await openPatternsTab(page);
    await expect(page.getByTestId("pattern-memory_manager")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("pattern-memory_manager")).toContainText("Long-term memory");
    await page.getByTestId("pattern-memory_manager").click();
    await expect(page.locator("#pattern-title")).toHaveText("Long-term memory");
  });
});

test.describe("workbench · Cognitive routing pattern", () => {
  test("pattern listed once runner is current", async ({ page, request }) => {
    const catalog = (await (await request.get(`${BFF}/api/patterns`)).json()) as P[];
    const has = catalog.some((p) => p.id === "cognitive_routing");
    test.skip(
      !has,
      "Runner needs restart to include cognitive_routing (added with LLM picker feature)",
    );
    await openPatternsTab(page);
    await expect(page.getByTestId("pattern-cognitive_routing")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("pattern-cognitive_routing")).toContainText(
      "Cognitive routing",
    );
  });

  test("selection mode toggle visible only on this pattern", async ({ page, request }) => {
    const catalog = (await (await request.get(`${BFF}/api/patterns`)).json()) as P[];
    const has = catalog.some((p) => p.id === "cognitive_routing");
    test.skip(!has, "Runner needs restart to include cognitive_routing");

    await openPatternsTab(page);
    // Pick any other pattern first — toggle must be hidden.
    await page.getByTestId(`pattern-${catalog[0].id}`).click();
    await expect(page.getByTestId("pattern-routing-toggle")).toBeHidden();

    // Switch to cognitive_routing — toggle becomes visible with the two
    // radio segments.
    await page.getByTestId("pattern-cognitive_routing").click();
    await expect(page.getByTestId("pattern-routing-toggle")).toBeVisible();
    await expect(page.getByTestId("pattern-mode-rules")).toBeVisible();
    await expect(page.getByTestId("pattern-mode-llm")).toBeVisible();
  });

  test("mode segments are mutually exclusive radios", async ({ page, request }) => {
    const catalog = (await (await request.get(`${BFF}/api/patterns`)).json()) as P[];
    const has = catalog.some((p) => p.id === "cognitive_routing");
    test.skip(!has, "Runner needs restart to include cognitive_routing");

    await openPatternsTab(page);
    await page.getByTestId("pattern-cognitive_routing").click();

    const rulesRadio = page.locator('input[name="pattern-mode"][value="rules"]');
    const llmRadio = page.locator('input[name="pattern-mode"][value="llm"]');

    // Default state: rules selected, llm not selected.
    await expect(rulesRadio).toBeChecked();
    await expect(llmRadio).not.toBeChecked();

    // Click the LLM segment label → only llm becomes checked.
    await page.getByTestId("pattern-mode-llm").click();
    await expect(llmRadio).toBeChecked();
    await expect(rulesRadio).not.toBeChecked();
  });
});
