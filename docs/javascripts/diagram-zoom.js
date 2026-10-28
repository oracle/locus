// Diagram expand — every inlined diagram (figure.tl-dia, written by the build
// hook) gets an "Expand" button and opens a copy of its SVG in a full-width
// overlay, so a dense drawing is readable without widening the page column.
// The overlay panel carries the tl-dia class so the SVG keeps its theme styles.

(function () {
  const ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 3h6v6h-2V6.41l-3.29 3.3-1.42-1.42L17.59 5H15V3zM3 15h2v2.59l3.29-3.3 1.42 1.42L6.41 19H9v2H3v-6z"/></svg>';

  function open(figure) {
    const svg = figure.querySelector("svg");
    if (!svg) return;
    const overlay = document.createElement("div");
    overlay.className = "tl-dia-overlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    const title = svg.querySelector("title");
    if (title) overlay.setAttribute("aria-label", title.textContent);
    const panel = document.createElement("div");
    panel.className = "tl-dia-overlay__panel tl-dia md-typeset";
    panel.appendChild(svg.cloneNode(true));
    const close = document.createElement("button");
    close.type = "button";
    close.className = "tl-dia-overlay__close";
    close.textContent = "Close";
    overlay.appendChild(panel);
    overlay.appendChild(close);
    document.body.appendChild(overlay);
    const previous = document.activeElement;
    function dismiss() {
      overlay.remove();
      document.removeEventListener("keydown", onKey);
      if (previous && previous.focus) previous.focus();
    }
    function onKey(event) {
      if (event.key === "Escape") dismiss();
    }
    overlay.addEventListener("click", function (event) {
      if (!panel.contains(event.target)) dismiss();
    });
    close.addEventListener("click", dismiss);
    document.addEventListener("keydown", onKey);
    close.focus();
  }

  function init() {
    document.querySelectorAll("figure.tl-dia").forEach(function (figure) {
      if (figure.querySelector(".tl-dia__expand")) return;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "tl-dia__expand";
      button.setAttribute("aria-label", "Expand diagram");
      button.innerHTML = ICON + "<span>Expand</span>";
      button.addEventListener("click", function () {
        open(figure);
      });
      figure.appendChild(button);
    });
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
