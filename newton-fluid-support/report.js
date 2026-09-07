(() => {
  const root = document.documentElement;
  const toggle = document.querySelector(".theme-toggle");
  const label = document.querySelector(".theme-label");
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const saved = localStorage.getItem("report-theme");

  function applyTheme(theme) {
    root.dataset.theme = theme;
    const next = theme === "dark" ? "light" : "dark";
    toggle.setAttribute("aria-label", `Switch to ${next} theme`);
    label.textContent = next[0].toUpperCase() + next.slice(1);
  }

  applyTheme(saved || (media.matches ? "dark" : "light"));
  toggle.addEventListener("click", () => {
    const theme = root.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem("report-theme", theme);
    applyTheme(theme);
  });
})();
