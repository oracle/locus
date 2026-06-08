// Make the header title text behave like the logo: click → docs home.
// Inherit the URL from the mkdocs-material-rendered `.md-logo` anchor
// rather than hard-coding "/" — the deployed site lives at
// `https://tuliplabs.ai/`, so "/" would resolve to the wrong
// place if we ever moved off the apex. `.md-logo` carries the correct
// site-relative href on every page.
document.addEventListener("DOMContentLoaded", () => {
  const title = document.querySelector(".md-header__ellipsis");
  const logo = document.querySelector(".md-header__button.md-logo");
  if (!title || !logo) return;
  title.style.cursor = "pointer";
  title.addEventListener("click", () => {
    window.location.href = logo.href;
  });
});
