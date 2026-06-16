/**
 * End-to-end Playwright coverage for the OCI Responses transport.
 *
 * Drives the workbench UI:
 *   - Configure OCI provider with transport=responses
 *   - Verify the Project OCID input becomes visible
 *   - Run the basic-agent pattern against the live OCI Responses endpoint
 *   - Assert a non-empty reply lands in the UI
 *
 * Required env (skipped silently otherwise):
 *   OCI_PROFILE       — profile in ~/.oci/config (defaults to BOAT-OC1)
 *   OCI_COMPARTMENT   — compartment OCID with GenAI Responses access
 *
 * Optional:
 *   OCI_REGION        — defaults to us-chicago-1
 *   OCI_MODEL         — defaults to openai.gpt-5 (Responses-capable)
 *   OCI_AUTH          — "session" (default) or "apikey"
 */
import { expect, test, type Page } from "@playwright/test";

const PROFILE = process.env.OCI_PROFILE;
const COMPARTMENT = process.env.OCI_COMPARTMENT;
const REGION = process.env.OCI_REGION ?? "us-chicago-1";
const MODEL = process.env.OCI_MODEL ?? "openai.gpt-5";
const AUTH = (process.env.OCI_AUTH ?? "session").toLowerCase();
const PROVIDER = AUTH === "apikey" ? "oci-apikey" : "oci-session";

test.skip(!PROFILE || !COMPARTMENT, "OCI_PROFILE + OCI_COMPARTMENT required");

async function configureResponses(page: Page): Promise<void> {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByTestId("settings-btn").click();
  await page.getByTestId("cfg-provider").selectOption(PROVIDER);
  await page.getByTestId("cfg-profile").fill(PROFILE!);
  await page.getByTestId("cfg-region").fill(REGION);
  await page.getByTestId("cfg-compartment").fill(COMPARTMENT!);
  await page.getByTestId("cfg-transport").selectOption("responses");

  // The Project OCID input appears only when transport=responses.
  await expect(page.getByTestId("cfg-project-ocid")).toBeVisible();

  // Wait for the live model list to populate, then pick a Responses-capable one.
  await expect(async () => {
    const opts = await page.getByTestId("cfg-model").locator("option").allTextContents();
    expect(opts.length).toBeGreaterThan(0);
  }).toPass({ timeout: 30_000 });

  // Some lists might not include gpt-5 by exact name — accept any matching prefix
  // ("openai.gpt-5"…) and fall back to whatever the dropdown chose first.
  const models = await page.getByTestId("cfg-model").locator("option").allTextContents();
  const match = models.find((m) => m === MODEL) ?? models.find((m) => m.startsWith("openai.gpt-5"));
  if (match) {
    await page.getByTestId("cfg-model").selectOption(match);
  }

  await page.getByTestId("settings-save").click();
}

test.setTimeout(300_000);

test("OCI Responses transport: configure + run basic agent", async ({ page }) => {
  await configureResponses(page);

  // Switch to the Patterns sidebar tab and wait for the list to populate.
  await page.getByTestId("side-tab-patterns").click();
  await expect(page.getByTestId("side-tab-patterns")).toHaveAttribute("aria-selected", "true");
  await expect(
    page.getByTestId("side-patterns").locator(".side__item").first(),
  ).toBeVisible({ timeout: 15_000 });

  // Select a non-streamable pattern. The streaming patterns hit a
  // pre-existing workbench wiring bug (api.ts streamPattern signature
  // doesn't match the callback shape patterns.ts expects), unrelated
  // to OCIResponsesModel — use the sync runPattern path to validate
  // the model class end-to-end.
  await page.getByTestId("pattern-composition").click();
  await expect(page.getByTestId("patterns-view")).toBeVisible();
  await expect(page.getByTestId("pattern-run-btn")).toBeEnabled();

  await page.getByTestId("pattern-prompt").fill("Say hi in three words.");

  // Capture browser console + network failures from the moment we click.
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error" || m.type() === "warning") {
      errors.push(`[${m.type()}] ${m.text()}`);
    }
  });
  page.on("pageerror", (e) => errors.push(`[pageerror] ${e.message}`));
  page.on("requestfailed", (r) => errors.push(`[requestfailed] ${r.method()} ${r.url()} - ${r.failure()?.errorText}`));
  page.on("response", (r) => {
    if (r.url().includes("/api/run")) errors.push(`[response] ${r.status()} ${r.url()}`);
  });

  await page.getByTestId("pattern-run-btn").click();

  // Quick poll: after 5 seconds, log whatever the page state is for diagnosis.
  await page.waitForTimeout(5000);
  const state = await page.evaluate(() => {
    const out = document.querySelector('[data-testid="pattern-output"]') as HTMLElement;
    const err = document.querySelector('[data-testid="pattern-error"]') as HTMLElement;
    return {
      outputText: out?.textContent ?? "",
      outputDisplay: out?.style.display ?? "",
      errorText: err?.textContent ?? "",
      errorDisplay: err?.style.display ?? "",
      runBtnDisplay: (document.querySelector("#pattern-run-btn") as HTMLElement)?.style.display ?? "",
    };
  });
  console.log("PAGE STATE 5s after click:", JSON.stringify(state, null, 2));
  console.log("ERRORS:", errors.join("\n"));

  // Wait for output to become visible (it's hidden until the first token).
  // OCI Responses + gpt-5 + auth + first-token latency can take a while.
  const output = page.getByTestId("pattern-output");
  const error = page.getByTestId("pattern-error");

  // Race: either the output produces text, or pattern-error shows a failure.
  await Promise.race([
    expect(output).not.toBeEmpty({ timeout: 240_000 }),
    expect(error).toBeVisible({ timeout: 240_000 }).then(async () => {
      const errText = (await error.textContent()) ?? "";
      throw new Error(`pattern-error surfaced: ${errText}`);
    }),
  ]);

  const text = (await output.textContent()) ?? "";
  expect(text.trim().length).toBeGreaterThan(0);
});
