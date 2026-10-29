// Stacked tables on one-column screens.
//
// A Markdown table cannot always give every column a readable width on a
// phone. Below the one-column breakpoint this measures each table and marks
// it with data-stack when it cannot be read as a table: it still overflows its
// column after compressing (first with its code chips kept whole, then with
// them allowed to wrap, data-wrap-code), it has three or more prose columns
// that would each be narrower than about 150px, or it has two columns and one
// of them is squeezed into a ribbon of one or two words per line. The
// stylesheet then lays its rows out as cards: the first cell is the row title,
// every other cell shows its column header as a caption, copied here into
// data-label. Tables with the same headers on one page stack together, so a
// catalogue does not alternate between the two presentations. A table that
// fits stays a table, as does the notebooks index (it has its own
// narrow-number layout). Reads and writes are batched so a page with a hundred
// tables (the API reference) costs a few layout passes, not one per table.
// Re-runs on resize and on Material's instant-navigation page changes.

(function () {
  const NARROW = window.matchMedia("(max-width: 48em)");
  const MIN_COLUMN_PX = 150;
  const SQUEEZED_COLUMN_PX = 150;
  const MATRIX_MAX_CHARS = 14;
  const FIRST_COLUMN_LABEL_CHARS = 28;
  const PHONE_COLUMN_PX = 500;
  const PROSE_CELL_CHARS = 40;
  const DASH = /^[—–-]$/;

  function headers(table) {
    const row = table.querySelector(":scope > thead > tr");
    return row ? Array.from(row.children) : [];
  }

  function text(el) {
    return el.textContent.replace(/\s+/g, " ").replace(/:$/, "").trim();
  }

  // One-time annotation: column count, data-label on every cell, and the
  // shape facts the decision needs (short matrix, index column, long first
  // column, prose cells).
  function label(table) {
    if (table.dataset.cols) return;
    const heads = headers(table);
    if (heads.length < 2) return;
    if (table.querySelector("[colspan], [rowspan]")) return;
    const article = table.closest("article");
    if (article && article.querySelector(".notebook-filter")) return;
    const names = heads.map(text);
    const codeHeads = heads.map(function (th) {
      return th.children.length === 1 && th.firstElementChild.tagName === "CODE" && text(th) === text(th.firstElementChild);
    });
    let longest = 0;
    let longestFirst = 0;
    let firstAllNumbers = true;
    const firstLengths = [];
    table.querySelectorAll(":scope > tbody > tr").forEach(function (tr) {
      Array.from(tr.children).forEach(function (td, i) {
        if (names[i]) td.setAttribute("data-label", names[i]);
        if (codeHeads[i]) td.setAttribute("data-label-code", "");
        const t = td.textContent.trim();
        if (DASH.test(t)) td.setAttribute("data-dash", "");
        longest = Math.max(longest, t.length);
        if (i === 0) {
          firstLengths.push(t.length);
          longestFirst = Math.max(longestFirst, t.length);
          if (!/^\d{1,2}$/.test(t)) firstAllNumbers = false;
        }
      });
    });
    firstLengths.sort(function (a, b) { return a - b; });
    const medianFirst = firstLengths.length ? firstLengths[Math.floor(firstLengths.length / 2)] : 0;
    table.dataset.cols = String(heads.length);
    table.dataset.key = names.join("|");
    table.dataset.matrix = longest <= MATRIX_MAX_CHARS ? "1" : "";
    table.dataset.prose = longest > PROSE_CELL_CHARS ? "1" : "";
    table.dataset.firstLong = medianFirst > FIRST_COLUMN_LABEL_CHARS ? "1" : "";
    // Recorded under its own name: data-index-col is the applied state and
    // is cleared on every run.
    if (firstLengths.length && firstAllNumbers && longestFirst <= 2 && heads.length >= 3) table.dataset.indexFirst = "1";
  }

  function wrapper(table) {
    return table.closest(".md-typeset__scrollwrap") || table.parentElement;
  }

  // The width a table may take without running past the text column: the
  // scroll wrapper bleeds 0.4rem into each margin, so its own width is too
  // generous by that padding.
  function columnWidth(table) {
    const inner = table.closest(".md-typeset__table") || table.parentElement;
    const style = getComputedStyle(inner);
    return inner.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
  }

  function overflows(table) {
    const wrap = wrapper(table);
    return table.scrollWidth > table.clientWidth + 1 ||
      table.getBoundingClientRect().width > columnWidth(table) + 1 ||
      wrap.scrollWidth > wrap.clientWidth + 1;
  }

  // A two-column table with prose whose narrower column is a ribbon. Only a
  // phone-width column counts: at tablet width the other column is wide.
  function squeezed(table) {
    if (Number(table.dataset.cols) !== 2 || !table.dataset.prose) return false;
    if (columnWidth(table) >= PHONE_COLUMN_PX) return false;
    const row = table.querySelector(":scope > tbody > tr");
    if (!row) return false;
    const widths = Array.from(row.children, function (td) { return td.getBoundingClientRect().width; });
    return widths.length === 2 && Math.min.apply(null, widths) < SQUEEZED_COLUMN_PX;
  }

  function apply() {
    const tables = Array.from(document.querySelectorAll(".md-typeset table:not([class])"));
    if (!tables.length) return;
    tables.forEach(function (table) {
      table.removeAttribute("data-stack");
      table.removeAttribute("data-wrap-code");
      table.removeAttribute("data-index-col");
    });
    if (!NARROW.matches) return;
    tables.forEach(label);
    const live = tables.filter(function (table) { return table.dataset.cols; });

    // Pass 1: measured as tables with code chips kept whole.
    const decision = new Map();
    const retry = [];
    live.forEach(function (table) {
      const available = columnWidth(table);
      const cols = Number(table.dataset.cols);
      const crowded = cols >= 3 && !table.dataset.matrix && cols * MIN_COLUMN_PX > available;
      // A sentence-length first column only forces cards on a phone-width
      // column; at tablet width the two columns read fine as a table.
      const firstLong = cols === 2 && table.dataset.firstLong === "1" && available < PHONE_COLUMN_PX;
      if (crowded || squeezed(table) || firstLong) {
        decision.set(table, true);
      } else if (overflows(table)) {
        retry.push(table);
      } else {
        decision.set(table, false);
      }
    });

    // Pass 2: a table that only overflows because of a code chip gets its
    // code allowed to wrap; it stacks if that is not enough, or if wrapping
    // the chips left one of two columns a ribbon.
    if (retry.length) {
      retry.forEach(function (table) { table.setAttribute("data-wrap-code", ""); });
      retry.forEach(function (table) { decision.set(table, overflows(table) || squeezed(table)); });
    }

    // Same headers on one page: stack all or none.
    const byKey = new Map();
    live.forEach(function (table) {
      const k = table.dataset.key;
      byKey.set(k, (byKey.get(k) || false) || decision.get(table));
    });
    live.forEach(function (table) {
      if (byKey.get(table.dataset.key)) {
        table.setAttribute("data-stack", "");
        if (table.dataset.indexFirst === "1") table.setAttribute("data-index-col", "");
      }
    });
  }

  let timer = null;
  window.addEventListener("resize", function () {
    clearTimeout(timer);
    timer = setTimeout(apply, 150);
  });

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(apply);
  } else {
    document.addEventListener("DOMContentLoaded", apply);
  }
})();
