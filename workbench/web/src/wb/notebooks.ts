/** Notebook sidebar: catalog fetch, filter, render, prev/next nav. */
import {
  getNotebook,
  listNotebookCategories,
  listNotebooks,
  type CategoryInfo,
  type Notebook,
  type NotebookDetail,
} from "../api";
import { $ } from "./dom";
import { setEditorContent } from "./editor";
import { showEmptyState } from "./output";

let notebooks: Notebook[] = [];
let categories: CategoryInfo[] = [];
let current: NotebookDetail | null = null;

export function getCurrent(): NotebookDetail | null {
  return current;
}

export function getNotebooks(): Notebook[] {
  return notebooks;
}

function sideNotebooks(): HTMLElement {
  return $("#side-notebooks");
}
function search(): HTMLInputElement {
  return $<HTMLInputElement>("#notebook-search");
}

export async function bootstrapNotebooks(): Promise<void> {
  try {
    // Categories load is best-effort — if it fails the sidebar still
    // renders, just without section headers.
    [notebooks, categories] = await Promise.all([
      listNotebooks(),
      listNotebookCategories().catch((err) => {
        console.warn("[wb/notebooks] categories load failed", err);
        return [] as CategoryInfo[];
      }),
    ]);
    console.info(
      `[wb/notebooks] loaded ${notebooks.length} notebooks in ${categories.length} categories`,
    );
    renderList("");
    if (notebooks.length) {
      // Default selection: notebook 01 in the catalog. Falls back to
      // whatever sorted first if the canonical id ever moves.
      const first =
        notebooks.find((t) => t.id === "notebook_01_oci_transports") ?? notebooks[0];
      await selectNotebook(first.id);
    }
  } catch (err) {
    console.error("[wb/notebooks] catalog load failed", err);
    sideNotebooks().innerHTML = `<div style="color: var(--or-red-deep); font-size:0.8rem; padding: 0.5rem">${(err as Error).message}</div>`;
  }
  search().addEventListener("input", () => renderList(search().value));
  installNavButtons();
}

export async function selectNotebook(id: string): Promise<void> {
  console.info("[wb/notebooks] select", id);
  try {
    current = await getNotebook(id);
  } catch (err) {
    console.error("[wb/notebooks] failed to load", id, err);
    return;
  }
  $("#wb-title").textContent = current.title;
  $("#wb-sub").textContent = current.summary || current.filename;
  setEditorContent(current.source);
  showEmptyState();
  $("#wb-output-pill").style.display = "none";
  $("#wb-status").textContent = `loaded ${current.filename}`;
  $("#crumbs").textContent = `Workbench · Notebook ${current.number}`;
  renderList(search().value);
  renderNavState();
  document
    .querySelector<HTMLElement>(`[data-testid="notebook-${current.id}"]`)
    ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function renderList(filter: string): void {
  const sidebar = sideNotebooks();
  sidebar.innerHTML = "";
  const q = filter.trim().toLowerCase();

  // Index categories by id for quick lookup; notebooks are pre-sorted
  // by the backend (category position, then category_order, then number)
  // so a single pass with a "previous category" sentinel produces
  // correctly ordered section headers.
  const catById: Map<string, CategoryInfo> = new Map(categories.map((c) => [c.id, c]));
  let lastCategory: string | null = null;

  for (const t of notebooks) {
    if (q && !`${t.number} ${t.title} ${t.id}`.toLowerCase().includes(q)) continue;

    const catId = t.category ?? "misc";
    if (catId !== lastCategory) {
      const meta = catById.get(catId);
      const header = document.createElement("div");
      header.className = "side__category";
      header.dataset.testid = `notebook-category-${catId}`;
      header.innerHTML = `
        <div class="side__category-name">${meta?.name ?? catId}</div>
        ${meta?.description ? `<div class="side__category-desc">${meta.description}</div>` : ""}
      `;
      sidebar.appendChild(header);
      lastCategory = catId;
    }

    const item = document.createElement("div");
    item.className = `side__item${current?.id === t.id ? " side__item--active" : ""}`;
    item.dataset.testid = `notebook-${t.id}`;
    const stdinBadge = t.needs_stdin
      ? `<span class="needs-stdin-badge" title="uses interrupt() — pops a modal for human input" data-testid="needs-stdin-badge">↩</span>`
      : "";
    item.innerHTML = `
      <span style="font-family: var(--mono); font-size: 0.7rem; color: var(--or-text-mute); min-width: 1.6rem">${String(t.number).padStart(2, "0")}</span>
      <span style="font-size: 0.82rem; flex: 1">${t.title.replace(/^Notebook \d+:\s*/i, "")}</span>
      ${stdinBadge}
    `;
    item.addEventListener("click", () => void selectNotebook(t.id));
    sidebar.appendChild(item);
  }
}

function installNavButtons(): void {
  const prev = $<HTMLButtonElement>("#wb-prev-btn");
  const next = $<HTMLButtonElement>("#wb-next-btn");
  const step = (delta: number) => {
    if (!current) return;
    const cid = current.id;
    const idx = notebooks.findIndex((t) => t.id === cid);
    const target = notebooks[idx + delta];
    if (target) void selectNotebook(target.id);
  };
  prev.addEventListener("click", () => step(-1));
  next.addEventListener("click", () => step(+1));
}

export function renderNavState(): void {
  const prev = $<HTMLButtonElement>("#wb-prev-btn");
  const next = $<HTMLButtonElement>("#wb-next-btn");
  const cur = current;
  const idx = cur ? notebooks.findIndex((t) => t.id === cur.id) : -1;
  prev.disabled = idx <= 0;
  next.disabled = idx === -1 || idx >= notebooks.length - 1;
}
