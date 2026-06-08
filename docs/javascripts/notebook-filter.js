// Notebook filter for /notebooks/ index page.
//
// Live-filters the notebook tables on docs/notebooks/index.md as the user
// types. Hides rows that don't match, then hides whole sections (the
// preceding h2 + intro paragraphs) when none of their rows survive.
// Modeled on the workbench sidebar filter UX.

(function () {
  function init() {
    const filter = document.querySelector("#notebook-filter-input");
    if (!filter) return;

    const tables = Array.from(
      document.querySelectorAll("article.md-content__inner table"),
    );
    if (!tables.length) return;

    // Pre-compute the searchable text per row + cache the preceding
    // section heading (h2) so we can hide the whole section when every
    // row in it disappears.
    const rows = [];
    for (const table of tables) {
      const bodyRows = Array.from(table.querySelectorAll("tbody tr"));
      for (const tr of bodyRows) {
        rows.push({
          el: tr,
          text: tr.textContent.toLowerCase(),
          table,
        });
      }
    }

    // For each table, walk backwards through the DOM until we hit an h2
    // — that's the section header we'll toggle alongside the table.
    const sectionsByTable = new Map();
    for (const table of tables) {
      const block = [table];
      let prev = table.previousElementSibling;
      while (prev && prev.tagName !== "H2") {
        block.unshift(prev);
        prev = prev.previousElementSibling;
      }
      if (prev) block.unshift(prev);
      sectionsByTable.set(table, block);
    }

    function apply() {
      const q = filter.value.trim().toLowerCase();
      const visibleTables = new Set();
      for (const row of rows) {
        const match = !q || row.text.includes(q);
        row.el.style.display = match ? "" : "none";
        if (match) visibleTables.add(row.table);
      }
      for (const table of tables) {
        const blockEls = sectionsByTable.get(table) || [];
        const show = visibleTables.has(table);
        for (const el of blockEls) {
          el.style.display = show ? "" : "none";
        }
      }
    }

    filter.addEventListener("input", apply);
    // Keyboard ergonomic: Cmd/Ctrl+K focuses the filter when on this page.
    document.addEventListener("keydown", (ev) => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
        ev.preventDefault();
        filter.focus();
        filter.select();
      }
      if (ev.key === "Escape" && document.activeElement === filter) {
        filter.value = "";
        apply();
        filter.blur();
      }
    });
  }

  // mkdocs Material with navigation.instant uses pjax — listen for the
  // theme's content-swap event so the filter rebinds after navigation.
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(() => init());
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
