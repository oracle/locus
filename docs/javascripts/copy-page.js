// Copy page — puts the page's Markdown twin on the clipboard.
//
// hooks/llms_txt.py writes every page a second time as Markdown at
// `<page-url>index.md` (and the site as /llms.txt and /llms-full.txt). This
// adds a "Copy page" button beside the title that fetches that twin and copies
// it, with a small menu for the raw file and the llms.txt files. Runs again on
// every instant-navigation page change via Material's `document$`.

(function () {
  const COPY_ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1zm3 4H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H8V7h11v14z"/></svg>';
  const CHEVRON_ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7.41 8.59 12 13.17l4.59-4.58L18 10l-6 6-6-6z"/></svg>';

  function siteRoot() {
    const logo = document.querySelector(".md-header__button.md-logo");
    return new URL(logo ? logo.getAttribute("href") : "/", location.href);
  }

  function init() {
    const inner = document.querySelector(".md-content__inner");
    if (!inner || inner.querySelector(".locus-copy-page")) return;
    // The landing page is a marketing page, not a document to copy.
    if (inner.querySelector(".locus-hero")) return;
    const h1 = inner.querySelector("h1");
    if (!h1) return;

    const mdUrl = new URL("index.md", location.href);
    const root = siteRoot();

    const wrap = document.createElement("div");
    wrap.className = "locus-copy-page";
    wrap.innerHTML =
      '<button type="button" class="locus-copy-page__btn" aria-label="Copy this page as Markdown">' +
      COPY_ICON +
      "<span>Copy page</span></button>" +
      '<button type="button" class="locus-copy-page__more" aria-haspopup="menu" aria-expanded="false" aria-label="Other formats">' +
      CHEVRON_ICON +
      "</button>" +
      '<ul class="locus-copy-page__menu" role="menu" hidden>' +
      '<li role="none"><a role="menuitem" href="' + mdUrl.pathname + '">View as Markdown</a></li>' +
      '<li role="none"><a role="menuitem" href="' + root.pathname + 'llms.txt">llms.txt</a></li>' +
      '<li role="none"><a role="menuitem" href="' + root.pathname + 'llms-full.txt">llms-full.txt</a></li>' +
      "</ul>";
    inner.insertBefore(wrap, h1);

    const btn = wrap.querySelector(".locus-copy-page__btn");
    const label = btn.querySelector("span");
    const more = wrap.querySelector(".locus-copy-page__more");
    const menu = wrap.querySelector(".locus-copy-page__menu");

    function flash(state, text) {
      btn.classList.remove("is-done", "is-failed");
      if (state) btn.classList.add(state);
      label.textContent = text;
      window.setTimeout(function () {
        btn.classList.remove("is-done", "is-failed");
        label.textContent = "Copy page";
      }, 1800);
    }

    btn.addEventListener("click", function () {
      fetch(mdUrl.href, { credentials: "same-origin" })
        .then(function (r) {
          if (!r.ok) throw new Error(String(r.status));
          return r.text();
        })
        .then(function (text) {
          return navigator.clipboard.writeText(text);
        })
        .then(function () {
          flash("is-done", "Copied");
        })
        .catch(function () {
          flash("is-failed", "Copy failed");
        });
    });

    function setOpen(open) {
      menu.hidden = !open;
      more.setAttribute("aria-expanded", open ? "true" : "false");
    }
    more.addEventListener("click", function (event) {
      event.stopPropagation();
      setOpen(menu.hidden);
    });
    document.addEventListener("click", function (event) {
      if (!wrap.contains(event.target)) setOpen(false);
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") setOpen(false);
    });
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
