// Capture a screenshot + a short MP4 of the workbench for the README.
// Run against a workbench you already have running locally:
//
//   docker run --rm -p 5173:5173 -p 3101:3101 -p 8100:8100 locus-workbench
//   # in another shell:
//   node workbench/e2e/scripts/capture-demo.mjs
//
// Outputs:
//   docs/img/workbench.png
//   docs/img/workbench.mp4   (convert to gif with ffmpeg, see README)
//
// Override the target with WORKBENCH_URL=http://localhost:5273 if you
// booted the container on alt ports.

import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const REPO_ROOT = resolve(__dirname, "..", "..", "..");
const OUT_DIR = resolve(REPO_ROOT, "docs", "img");

const URL = process.env.WORKBENCH_URL ?? "http://localhost:5173";
const VIEWPORT = { width: 1440, height: 900 };

await mkdir(OUT_DIR, { recursive: true });

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: VIEWPORT,
  recordVideo: { dir: OUT_DIR, size: VIEWPORT },
});
const page = await context.newPage();

console.log(`[capture] navigating to ${URL}`);
await page.goto(URL, { waitUntil: "networkidle" });

// Best-effort wait for the notebooks sidebar to populate (depends on the
// BFF responding with the catalog). Continue even if it doesn't show up.
await page
  .waitForSelector('[data-testid="side-notebooks"] .side__item', {
    timeout: 8_000,
  })
  .catch(() => console.log("[capture] sidebar slow to populate; continuing"));
await page.waitForTimeout(1500);

console.log("[capture] taking still screenshot");
await page.screenshot({
  path: resolve(OUT_DIR, "workbench.png"),
  fullPage: false,
});

// Brief animated walk-through: hover the catalog, open Provider settings,
// close it again. Keeps the recording short (≈8s) so the gif stays small.
console.log("[capture] recording walk-through");
await page.waitForTimeout(800);
const items = page.locator('[data-testid="side-notebooks"] .side__item');
const count = Math.min(await items.count(), 6);
for (let i = 0; i < count; i++) {
  await items.nth(i).hover();
  await page.waitForTimeout(220);
}
await page.locator('[data-testid="settings-btn"]').click().catch(() => {});
await page.waitForTimeout(1500);
await page.locator('[data-testid="settings-cancel"]').click().catch(() => {});
await page.waitForTimeout(600);

await page.close();
const videoPath = await page.video()?.path();
await context.close();
await browser.close();

if (videoPath) {
  // Rename the auto-generated webm to a stable name for the README.
  const { rename } = await import("node:fs/promises");
  const stable = resolve(OUT_DIR, "workbench.webm");
  await rename(videoPath, stable);
  console.log(`[capture] video at ${stable}`);
}

console.log(`[capture] still at ${resolve(OUT_DIR, "workbench.png")}`);
