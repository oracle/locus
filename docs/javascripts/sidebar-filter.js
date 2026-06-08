// Sidebar live filter for notebook entries.
//
// Injects a search input at the top of the primary sidebar on pages whose
// sidebar lists notebooks (the Notebooks tab). Hides non-matching
// notebook rows as you type and collapses entire sections when all of
// their notebooks are hidden. Other top-level items (Quickstart, Guides)
// are left alone.

(function () {
  const FILTER_ID = "sidebar-notebook-filter-input";

  function init() {
    const root = document.querySelector(
      ".md-sidebar--primary .md-nav--primary > .md-nav__list",
    );
    if (!root) return;

    // Only inject when the sidebar actually lists notebooks.
    if (!root.querySelector('a.md-nav__link[href*="/notebooks/notebook_"]')) {
      return;
    }

    let input = document.getElementById(FILTER_ID);
    if (!input) {
      const wrap = document.createElement("li");
      wrap.className = "md-nav__item sidebar-filter";
      const inp = document.createElement("input");
      inp.id = FILTER_ID;
      inp.className = "sidebar-filter__input";
      inp.type = "search";
      inp.placeholder = "Filter notebooks…";
      inp.autocomplete = "off";
      inp.spellcheck = false;
      inp.setAttribute("autocorrect", "off");
      wrap.appendChild(inp);
      root.insertBefore(wrap, root.firstChild);
      input = inp;
    }

    // Collect every notebook entry grouped by its enclosing section.
    const sections = Array.from(
      root.querySelectorAll("li.md-nav__item--section"),
    );
    const groups = [];
    for (const section of sections) {
      const items = Array.from(
        section.querySelectorAll("li.md-nav__item"),
      ).filter((li) =>
        li.querySelector('a.md-nav__link[href*="/notebooks/notebook_"]'),
      );
      if (items.length) groups.push({ section, items });
    }
    if (!groups.length) return;

    function apply() {
      const q = input.value.trim().toLowerCase();
      for (const { section, items } of groups) {
        let anyVisible = false;
        for (const li of items) {
          const text = (li.textContent || "").toLowerCase();
          const match = !q || text.includes(q);
          li.style.display = match ? "" : "none";
          if (match) anyVisible = true;
        }
        section.style.display = anyVisible ? "" : "none";
      }
    }

    if (!input.dataset.bound) {
      input.addEventListener("input", apply);
      input.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape") {
          input.value = "";
          apply();
          input.blur();
        }
      });
      input.dataset.bound = "1";
    }
    apply();
  }

  // mkdocs Material with navigation.instant swaps content via pjax —
  // listen for the theme's content-swap event so the filter rebinds.
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(() => init());
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
