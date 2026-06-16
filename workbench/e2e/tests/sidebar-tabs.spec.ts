/**
 * Sidebar-tabs e2e — verifies the Notebooks / Skills / Protocols
 * three-tab layout. No model provider is needed: every endpoint
 * exercised here (BFF /api/skills, /api/protocols, /api/notebooks) is
 * read-only.
 *
 * What this spec proves end-to-end:
 *   1. Tab switcher renders all three buttons.
 *   2. Clicking each tab swaps the sidebar pane *and* the main view.
 *   3. Skills tab lists every SKILL.md package under examples/skills/.
 *   4. Protocols tab lists all 8 builtin protocols.
 *   5. Clicking a protocol shows its detail view: runtime_shape,
 *      cost/risk_max/latency pills, HANDLES + CANONICAL FOR chips.
 *   6. The HANDLES chip subset that overlaps with primary_for is
 *      visually flagged (`.tt-chip--primary`).
 */
import { test, expect, type Page } from "@playwright/test";

test.use({ video: "off", trace: "off" });

const BFF = process.env.BFF_URL ?? "http://127.0.0.1:3101";

type ProtocolSummary = {
  id: string;
  handles: string[];
  primary_for: string[];
  cost: string;
  risk_max: string;
  latency: string;
};

const EXPECTED_PROTOCOL_IDS = [
  "direct_response",
  "plan_execute_validate",
  "specialist_fanout",
  "debate",
  "codegen_test_validate",
  "approval_gated_execution",
  "a2a_delegate",
  "handoff_chain",
];

async function freshPage(page: Page): Promise<void> {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
}

test.describe("workbench · sidebar tabs", () => {
  test("tab switcher renders Notebooks / Skills / Protocols", async ({ page }) => {
    await freshPage(page);
    await expect(page.getByTestId("side-tab-notebooks")).toBeVisible();
    await expect(page.getByTestId("side-tab-skills")).toBeVisible();
    await expect(page.getByTestId("side-tab-protocols")).toBeVisible();
    // Notebooks starts active.
    await expect(page.getByTestId("side-tab-notebooks")).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  test("Skills tab lists every SKILL.md package", async ({ page, request }) => {
    await freshPage(page);
    await page.getByTestId("side-tab-skills").click();
    await expect(page.getByTestId("side-tab-skills")).toHaveAttribute(
      "aria-selected",
      "true",
    );
    // Sidebar pane swaps.
    await expect(page.getByTestId("side-skills")).toBeVisible();
    // Main view swaps too — workbench panel is hidden.
    await expect(page.getByTestId("wb-root")).toBeHidden();
    await expect(page.getByTestId("skills-view")).toBeVisible();

    // Number of cards in the sidebar must match what the BFF returned.
    const apiResp = await request.get(`${BFF}/api/skills`);
    expect(apiResp.ok()).toBe(true);
    const skills = (await apiResp.json()) as Array<{ id: string }>;
    expect(skills.length).toBeGreaterThanOrEqual(4);
    for (const s of skills) {
      await expect(page.getByTestId(`skill-${s.id}`)).toBeVisible();
    }
  });

  test("Protocols tab lists all 8 builtin protocols", async ({ page, request }) => {
    await freshPage(page);
    await page.getByTestId("side-tab-protocols").click();
    await expect(page.getByTestId("side-tab-protocols")).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await expect(page.getByTestId("side-protocols")).toBeVisible();
    await expect(page.getByTestId("protocols-view")).toBeVisible();
    await expect(page.getByTestId("wb-root")).toBeHidden();
    await expect(page.getByTestId("skills-view")).toBeHidden();

    const apiResp = await request.get(`${BFF}/api/protocols`);
    expect(apiResp.ok()).toBe(true);
    const protocols = (await apiResp.json()) as ProtocolSummary[];
    expect(protocols.map((p) => p.id).sort()).toEqual(
      [...EXPECTED_PROTOCOL_IDS].sort(),
    );
    for (const id of EXPECTED_PROTOCOL_IDS) {
      await expect(page.getByTestId(`protocol-${id}`)).toBeVisible();
    }
  });

  test("clicking a protocol loads its detail view", async ({ page, request }) => {
    await freshPage(page);
    await page.getByTestId("side-tab-protocols").click();
    await page.getByTestId("protocol-specialist_fanout").click();

    // Title + crumbs reflect the selection.
    await expect(page.locator("#protocol-title")).toHaveText("specialist_fanout");
    await expect(page.locator("#crumbs")).toContainText("specialist_fanout");

    // Three pills (cost / risk_max / latency) are visible and carry the
    // exact values from the BFF.
    const apiResp = await request.get(`${BFF}/api/protocols/specialist_fanout`);
    const detail = (await apiResp.json()) as ProtocolSummary & {
      runtime_shape: string;
    };
    await expect(page.locator("#protocol-cost-pill")).toContainText(detail.cost);
    await expect(page.locator("#protocol-risk-pill")).toContainText(detail.risk_max);
    await expect(page.locator("#protocol-latency-pill")).toContainText(detail.latency);

    // Runtime-shape callout has the same string the structural-audit
    // tests pin in tests/unit/test_router_compiled_shape.py.
    await expect(page.getByTestId("protocol-shape")).toHaveText(detail.runtime_shape);

    // HANDLES row contains every task type the BFF declared.
    const handlesRow = page.getByTestId("protocol-handles");
    for (const tt of detail.handles) {
      await expect(handlesRow.locator(".tt-chip", { hasText: tt })).toBeVisible();
    }

    // CANONICAL FOR row contains every primary_for entry, *and* those
    // chips are the highlighted variant.
    const primaryRow = page.getByTestId("protocol-primary");
    for (const tt of detail.primary_for) {
      await expect(primaryRow.locator(`.tt-chip--primary`, { hasText: tt })).toBeVisible();
    }

    // Sanity check on the highlight rule itself: every chip in the
    // CANONICAL FOR row must be marked primary.
    const canonicalChips = primaryRow.locator(".tt-chip");
    const totalCanonical = await canonicalChips.count();
    const primaryCanonical = await primaryRow.locator(".tt-chip--primary").count();
    expect(primaryCanonical).toBe(totalCanonical);
  });

  test("filter input narrows the protocol list", async ({ page }) => {
    await freshPage(page);
    await page.getByTestId("side-tab-protocols").click();
    // Pre-filter — every protocol visible.
    await expect(page.getByTestId(`protocol-direct_response`)).toBeVisible();
    await expect(page.getByTestId(`protocol-debate`)).toBeVisible();
    await expect(page.getByTestId(`protocol-handoff_chain`)).toBeVisible();

    await page.locator("#protocol-search").fill("debate");
    await expect(page.getByTestId(`protocol-debate`)).toBeVisible();
    await expect(page.getByTestId(`protocol-direct_response`)).toBeHidden();
    await expect(page.getByTestId(`protocol-handoff_chain`)).toBeHidden();
  });

  test("each tab is independently re-selectable", async ({ page }) => {
    // Round-trip: notebooks → skills → protocols → notebooks. The
    // workbench view must be visible at the end (and skills/protocols
    // hidden).
    await freshPage(page);
    await page.getByTestId("side-tab-skills").click();
    await page.getByTestId("side-tab-protocols").click();
    await page.getByTestId("side-tab-notebooks").click();

    await expect(page.getByTestId("wb-root")).toBeVisible();
    await expect(page.getByTestId("skills-view")).toBeHidden();
    await expect(page.getByTestId("protocols-view")).toBeHidden();
    await expect(page.getByTestId("side-tab-notebooks")).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  test("notebook selection survives a tab round-trip", async ({ page }) => {
    // Pick a notebook → switch to Protocols → back to Notebooks. The
    // editor must still hold the notebook's source (not be wiped). We
    // wait until the BFF-fetched notebook body has clearly replaced
    // the boot placeholder before snapshotting.
    await freshPage(page);
    await expect
      .poll(
        () => page.evaluate(() => ((window as any).__wb?.getSource?.() ?? "").length),
        { timeout: 15_000 },
      )
      .toBeGreaterThan(500); // notebook code is much longer than the placeholder
    const original = await page.evaluate(() => (window as any).__wb.getSource());
    expect(original.length).toBeGreaterThan(500);

    await page.getByTestId("side-tab-protocols").click();
    await page.getByTestId("protocol-debate").click();
    await expect(page.locator("#protocol-title")).toHaveText("debate");

    await page.getByTestId("side-tab-notebooks").click();
    await expect(page.getByTestId("wb-root")).toBeVisible();
    const after = await page.evaluate(() => (window as any).__wb.getSource());
    expect(after).toBe(original);
  });
});

// ---------------------------------------------------------------------------
// Per-protocol detail loop — one test per builtin protocol id. Each
// asserts that clicking the sidebar entry loads the detail view with
// fields that match what /api/protocols/:pid returned. Catches drift
// between the BFF contract and the rendered view.
// ---------------------------------------------------------------------------

const ALL_PROTOCOL_IDS = [
  "direct_response",
  "plan_execute_validate",
  "specialist_fanout",
  "debate",
  "codegen_test_validate",
  "approval_gated_execution",
  "a2a_delegate",
  "handoff_chain",
] as const;

test.describe("workbench · per-protocol detail", () => {
  for (const pid of ALL_PROTOCOL_IDS) {
    test(`${pid} detail view matches BFF metadata`, async ({ page, request }) => {
      const apiResp = await request.get(`${BFF}/api/protocols/${pid}`);
      expect(apiResp.ok()).toBe(true);
      const detail = (await apiResp.json()) as {
        id: string;
        description: string;
        cost: string;
        risk_max: string;
        latency: string;
        runtime_shape: string;
        handles: string[];
        primary_for: string[];
      };

      await freshPage(page);
      await page.getByTestId("side-tab-protocols").click();
      await page.getByTestId(`protocol-${pid}`).click();

      // Title + description.
      await expect(page.locator("#protocol-title")).toHaveText(detail.id);
      await expect(page.locator("#protocol-sub")).toHaveText(detail.description);

      // Pill triple — exact text match including the label prefix.
      await expect(page.locator("#protocol-cost-pill")).toHaveText(
        `cost: ${detail.cost}`,
      );
      await expect(page.locator("#protocol-risk-pill")).toHaveText(
        `risk_max: ${detail.risk_max}`,
      );
      await expect(page.locator("#protocol-latency-pill")).toHaveText(
        `latency: ${detail.latency}`,
      );

      // Runtime-shape callout — the load-bearing claim of the panel.
      await expect(page.getByTestId("protocol-shape")).toHaveText(detail.runtime_shape);

      // HANDLES row contains every entry from the BFF.
      const handlesRow = page.getByTestId("protocol-handles");
      for (const tt of detail.handles) {
        await expect(handlesRow.locator(".tt-chip", { hasText: tt })).toBeVisible();
      }

      // CANONICAL FOR row — matches BFF, may be empty for opt-in
      // protocols (a2a_delegate). When non-empty, every chip carries
      // the highlighted variant.
      const primaryRow = page.getByTestId("protocol-primary");
      if (detail.primary_for.length === 0) {
        await expect(primaryRow.locator(".tt-chip")).toHaveCount(0);
      } else {
        for (const tt of detail.primary_for) {
          await expect(
            primaryRow.locator(".tt-chip--primary", { hasText: tt }),
          ).toBeVisible();
        }
      }
    });
  }
});
