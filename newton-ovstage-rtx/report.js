// Theme, navigation and code controls adapted from the Aperture template.
(function () {
  const root = document.documentElement;
  const themeButton = document.getElementById("theme-toggle");
  function updateThemeControl() {
    const dark = root.dataset.theme === "dark";
    themeButton.setAttribute("aria-pressed", String(dark));
    themeButton.setAttribute("aria-label", dark ? "Switch to bright theme" : "Switch to dark theme");
    themeButton.querySelector(".theme-icon").textContent = dark ? "☾" : "☀";
    themeButton.querySelector(".theme-label").textContent = dark ? "Dark" : "Bright";
  }
  updateThemeControl();
  themeButton.addEventListener("click", function () {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("report-theme", root.dataset.theme); } catch (_) {}
    updateThemeControl();
    window.dispatchEvent(new CustomEvent("report-theme-change"));
  });

  document.querySelectorAll(".copy-button").forEach(function (button) {
    button.addEventListener("click", async function () {
      const target = document.getElementById(button.dataset.copyTarget);
      const text = target.textContent;
      let copied = false;
      try {
        await navigator.clipboard.writeText(text);
        copied = true;
      } catch (_) {
        const field = document.createElement("textarea");
        field.value = text;
        field.style.position = "fixed";
        field.style.opacity = "0";
        document.body.appendChild(field);
        field.select();
        copied = document.execCommand("copy");
        field.remove();
        button.focus();
      }
      button.textContent = copied ? "Copied" : "Select code to copy";
      setTimeout(() => { button.textContent = "Copy code"; }, 1600);
    });
  });

  const links = Array.from(document.querySelectorAll("[data-section-link]"));
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(function (entries) {
      const entry = entries.filter(e => e.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!entry) return;
      links.forEach(link => {
        if (link.hash === "#" + entry.target.id) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      });
    }, { rootMargin: "-15% 0px -60%", threshold: [0, .1, .3] });
    document.querySelectorAll(".report-section").forEach(section => observer.observe(section));
  }

  const scene = document.getElementById("scene-select");
  scene.addEventListener("change", function () {
    document.querySelectorAll(".comparison").forEach(panel => {
      panel.hidden = panel.id !== "compare-" + scene.value;
    });
  });
  document.querySelectorAll(".comparison").forEach(panel => {
    const button = panel.querySelector(".comparison-toggle");
    const slider = panel.querySelector(".slider-control");
    button.addEventListener("click", function () {
      const enabled = panel.classList.toggle("overlay");
      button.setAttribute("aria-pressed", String(enabled));
      button.textContent = enabled ? "Side-by-side comparison" : "Overlay comparison";
      slider.hidden = !enabled;
    });
    slider.querySelector("input").addEventListener("input", function (event) {
      panel.style.setProperty("--reveal", event.target.value + "%");
    });
  });
})();
