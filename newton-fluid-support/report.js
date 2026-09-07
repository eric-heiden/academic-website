(function () {
  const root = document.documentElement;
  const themeButton = document.getElementById("theme-toggle");
  const themeIcon = themeButton?.querySelector(".theme-icon");
  const themeLabel = themeButton?.querySelector(".theme-label");

  function updateThemeControl() {
    const dark = root.dataset.theme === "dark";
    if (!themeButton) return;
    themeButton.setAttribute("aria-pressed", String(dark));
    themeButton.setAttribute("aria-label", dark ? "Switch to bright theme" : "Switch to dark theme");
    if (themeIcon) themeIcon.textContent = dark ? "☾" : "☀";
    if (themeLabel) themeLabel.textContent = dark ? "Dark" : "Bright";
  }

  updateThemeControl();
  themeButton?.addEventListener("click", function () {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("report-theme", root.dataset.theme); } catch (_) {}
    updateThemeControl();
  });

  document.querySelectorAll(".copy-button").forEach(function (button) {
    button.addEventListener("click", async function () {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.innerText);
      } catch (_) {
        const textarea = document.createElement("textarea");
        textarea.value = target.innerText;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
      const original = button.textContent;
      button.textContent = "Copied";
      setTimeout(function () { button.textContent = original; }, 1400);
    });
  });

  const sectionLinks = Array.from(document.querySelectorAll("[data-section-link]"));
  const sections = sectionLinks
    .map(function (link) { return document.querySelector(link.getAttribute("href")); })
    .filter(Boolean);

  if ("IntersectionObserver" in window && sections.length) {
    const observer = new IntersectionObserver(function (entries) {
      const visible = entries
        .filter(function (entry) { return entry.isIntersecting; })
        .sort(function (a, b) { return b.intersectionRatio - a.intersectionRatio; });
      if (!visible.length) return;
      sectionLinks.forEach(function (link) {
        const active = link.getAttribute("href") === "#" + visible[0].target.id;
        if (active) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      });
    }, { rootMargin: "-25% 0px -62%", threshold: [0.05, 0.25, 0.6] });
    sections.forEach(function (section) { observer.observe(section); });
  }
})();
