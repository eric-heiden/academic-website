(function () {
  const root = document.documentElement;
  const validTemplates = new Set(["prism", "signal", "aperture"]);
  const picker = document.getElementById("template-picker");
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
    window.dispatchEvent(new CustomEvent("report-theme-change"));
  });

  if (picker) {
    picker.value = validTemplates.has(root.dataset.template) ? root.dataset.template : "aperture";
    picker.addEventListener("change", function () {
      if (!validTemplates.has(picker.value)) return;
      root.dataset.template = picker.value;
      const url = new URL(window.location.href);
      url.searchParams.set("template", picker.value);
      history.replaceState({}, "", url);
      window.dispatchEvent(new CustomEvent("report-theme-change"));
    });
  }

  document.querySelectorAll(".copy-button").forEach(function (button) {
    button.addEventListener("click", async function () {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      const text = target.innerText;
      try {
        await navigator.clipboard.writeText(text);
      } catch (_) {
        const textarea = document.createElement("textarea");
        textarea.value = text;
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

(function () {
  const selector = document.getElementById("channel-select");
  const image = document.getElementById("channel-image");
  const detail = document.getElementById("channel-detail");
  if (!selector || !image || !detail) return;
  const channels = {
    "sensor-color": ["CPU sensor color; 384 × 256 pixels, 48° vertical field of view. The camera is at (3, −4, 2.8) m.", "CPU sensor color observation of a red sphere, blue cube, and yellow capsule above a ground plane."],
    "sensor-albedo": ["CPU sensor albedo at the first camera pose. Albedo shows surface color without illumination shading.", "Albedo image with flat red sphere, blue cube and yellow capsule colors."],
    "sensor-camera-two": ["CPU sensor color from a second camera at (−3, −3.2, 2) m; target (0, 0.15, 0.4) m, vertical field of view 48°. The simulated state is unchanged.", "The same scene observed from the opposite side, changing occlusion and the relative screen positions of the sphere, box and capsule."],
    "sensor-contacts": ["Six actual collision-pipeline contacts are marked in magenta at physical surface midpoints. This capture uses the always-visible overlay policy.", "Sensor image with six magenta markers indicating generated physical contact surface midpoints."],
    "sensor-forward-depth": ["Forward depth in grayscale: near = 255, far = 50, misses = 0; display range 3.390–122.766 m. Raw depth remains metric and is not the PNG intensity.", "Grayscale forward-depth observation of the sphere, box, capsule and ground."],
    "sensor-normal": ["World-space normal components from −1 to 1 are mapped to RGB values from 0 to 255; missed rays are black.", "RGB normal visualization with surface orientations encoded by color."],
    "sensor-shape-index": ["Shape identifiers use a deterministic display palette; missed rays are black. The numerical IDs are separate from the displayed colors.", "Flat false-color shape-identifier image separating the sphere, box, capsule and ground."],
    "viewer-color": ["Actual attached CPU ViewerGL capture: native 512 × 384 framebuffer, requested 384 × 256 output and matching requested camera aspect ratio.", "Attached ViewerGL color observation of the same three primitive shapes."],
    "viewer-wireframe": ["Actual attached ViewerGL mesh wireframe. Sensor observations do not support this channel.", "ViewerGL wireframe rendering revealing the triangle edges of the sphere, box and capsule."],
    "default-sensor-attached": ["Sensor remains the default when ViewerGL is attached. Six actual contacts use the always-visible overlay policy.", "Sensor capture with contact markers produced while an interactive ViewerGL is attached."]
  };
  selector.addEventListener("change", function () {
    const value = channels[selector.value];
    if (!value) return;
    image.src = "assets/" + selector.value + ".png";
    image.alt = value[1];
    detail.textContent = value[0];
  });
})();
