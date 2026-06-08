// Make the header title ("tulip agents SDK") behave like the logo:
// click → docs home. Uses event delegation on `document` so it keeps
// working across mkdocs-material `navigation.instant` page swaps.
//
// `site_url` is the production apex (tuliplabs.ai), so the rendered logo
// anchor is absolute to that origin. We honour it when we're actually
// served from that origin, but fall back to the current origin's root
// for local dev / previews — otherwise "home" would jump off to the
// production domain.
document.addEventListener("click", (e) => {
  const target = e.target;
  if (!(target instanceof Element)) return;
  // Only the title slot — never hijack the logo, repo links, or icons.
  if (!target.closest(".md-header__title") || target.closest("a")) return;
  e.preventDefault();
  const logo = document.querySelector(".md-header__button.md-logo");
  let dest = logo && logo.href ? logo.href : window.location.origin + "/";
  try {
    if (new URL(dest).origin !== window.location.origin) {
      dest = window.location.origin + "/";
    }
  } catch {
    dest = window.location.origin + "/";
  }
  window.location.href = dest;
});
