import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.17.2/dist/mermaid.esm.min.mjs";

let renderVersion = 0;

async function renderDiagrams() {
  const version = ++renderVersion;
  const root = document.documentElement;
  const styles = getComputedStyle(root);
  const dark = root.dataset.theme === "dark";
  const colors = {
    background: styles.getPropertyValue("--surface").trim(),
    primaryColor: styles.getPropertyValue("--accent-soft").trim(),
    primaryTextColor: styles.getPropertyValue("--text").trim(),
    primaryBorderColor: styles.getPropertyValue("--accent").trim(),
    secondaryColor: styles.getPropertyValue("--surface-2").trim(),
    secondaryTextColor: styles.getPropertyValue("--text").trim(),
    tertiaryColor: styles.getPropertyValue("--surface-3").trim(),
    tertiaryTextColor: styles.getPropertyValue("--text").trim(),
    lineColor: styles.getPropertyValue("--line-strong").trim(),
    textColor: styles.getPropertyValue("--text").trim(),
    clusterBkg: styles.getPropertyValue("--surface-2").trim(),
    clusterBorder: styles.getPropertyValue("--line").trim(),
    fontFamily: styles.getPropertyValue("--sans").trim(),
    darkMode: dark
  };
  mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: "base", themeVariables: colors, flowchart: { htmlLabels: false, curve: "linear", nodeSpacing: 22, rankSpacing: 34 } });

  for (const node of document.querySelectorAll("[data-mermaid-source]")) {
    const original = node.dataset.mermaidSource;
    const source = window.innerWidth < 620 ? original.replace("flowchart LR", "flowchart TB") : original;
    if (!source) continue;
    try {
      const result = await mermaid.render("newton-architecture-" + version + "-" + Math.random().toString(36).slice(2), source);
      if (version !== renderVersion) return;
      node.innerHTML = result.svg;
    } catch (error) {
      node.textContent = "The architecture diagram could not be rendered.";
      console.error(error);
    }
  }
}

window.addEventListener("report-theme-change", renderDiagrams);
window.addEventListener("load", renderDiagrams, { once: true });

let resizeTimer;
window.addEventListener("resize", function () {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderDiagrams, 160);
});
