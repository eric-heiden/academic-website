document.addEventListener("DOMContentLoaded", function () {
  if (typeof window.renderMathInElement !== "function") return;

  const report = document.querySelector(".report");
  if (!report) return;

  window.renderMathInElement(report, {
    delimiters: [
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false }
    ],
    throwOnError: false
  });
});
