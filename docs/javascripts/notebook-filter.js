// Examples filter for the /notebooks/ index page.
//
// Live-filters the example lists on docs/notebooks/index.md as the user
// types: hides the entries that do not match, then hides a whole section
// (its h2 and anything between the h2 and the list) when none of its
// entries survive. Cmd+K on macOS or Ctrl+K elsewhere focuses the filter;
// Escape clears it. The shortcut hint is appended to the placeholder from
// here, on devices with a keyboard-style pointer, so it is only advertised
// where the binding exists and can be used.

(function () {
  const platform =
    (navigator.userAgentData && navigator.userAgentData.platform) ||
    navigator.platform ||
    "";
  const SHORTCUT = /mac|iphone|ipad|ipod/i.test(platform) ? "⌘K" : "Ctrl+K";
  const HAS_KEYBOARD = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

  function currentFilter() {
    return document.querySelector("#notebook-filter-input");
  }

  function init() {
    const filter = currentFilter();
    if (!filter || filter.dataset.filterReady) return;

    // The catalog: every list that is a direct child of the article and
    // follows the filter box. Lists inside admonitions are not entries.
    const article = filter.closest("article") || document.body;
    const lists = Array.from(article.querySelectorAll(":scope > ul")).filter(
      (ul) =>
        filter.compareDocumentPosition(ul) & Node.DOCUMENT_POSITION_FOLLOWING,
    );
    if (!lists.length) return;
    filter.dataset.filterReady = "1";
    if (HAS_KEYBOARD) {
      filter.placeholder = filter.placeholder.trim() + " press " + SHORTCUT;
    }

    // Pre-compute the searchable text per entry and, per list, the block
    // of elements back to the section heading, so a section whose entries
    // all disappear hides as one.
    const entries = [];
    const sections = new Map();
    for (const list of lists) {
      for (const li of list.querySelectorAll(":scope > li")) {
        entries.push({ el: li, text: li.textContent.toLowerCase(), list });
      }
      const block = [list];
      let prev = list.previousElementSibling;
      while (prev && prev.tagName !== "H2") {
        block.unshift(prev);
        prev = prev.previousElementSibling;
      }
      if (prev) block.unshift(prev);
      sections.set(list, block);
    }

    function apply() {
      const q = filter.value.trim().toLowerCase();
      const alive = new Set();
      for (const entry of entries) {
        const match = !q || entry.text.includes(q);
        entry.el.style.display = match ? "" : "none";
        if (match) alive.add(entry.list);
      }
      for (const [list, block] of sections) {
        const show = alive.has(list);
        for (const el of block) el.style.display = show ? "" : "none";
      }
    }

    filter.addEventListener("input", apply);
  }

  // One document-level listener that looks the input up at event time, so
  // it keeps working across instant-navigation page swaps.
  document.addEventListener("keydown", (ev) => {
    const filter = currentFilter();
    if (!filter) return;
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
      ev.preventDefault();
      filter.focus();
      filter.select();
    } else if (ev.key === "Escape" && document.activeElement === filter) {
      filter.value = "";
      filter.dispatchEvent(new Event("input"));
      filter.blur();
    }
  });

  // mkdocs Material with navigation.instant swaps the page content; the
  // theme's document$ fires after each swap so the filter rebinds.
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(() => init());
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
