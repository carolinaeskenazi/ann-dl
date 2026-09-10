window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  // Quem faz o typeset é o document$ abaixo, em toda carga de página. Se o MathJax
  // também fizesse o dele na inicialização, as fórmulas sairiam duplicadas.
  startup: { typeset: false }
};

// Re-typeset after each page load (Material SPA). Sem limpar o cache, a navegação
// instantânea reaproveita o CSS dos caracteres da página anterior e as fórmulas
// aparecem vazias (só as bordas de \boxed, sem os símbolos).
document$.subscribe(() => {
  if (!window.MathJax?.startup?.promise) return;
  MathJax.startup.promise.then(() => {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    return MathJax.typesetPromise();
  });
});
