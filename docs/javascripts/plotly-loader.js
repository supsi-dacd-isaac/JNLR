// Re-render Plotly plots on instant navigation
document$.subscribe(() => {
  // Find all script tags that contain Plotly.newPlot or Plotly.react
  const scripts = document.querySelectorAll('script:not([src])');
  scripts.forEach(script => {
    const content = script.textContent;
    if (content && (content.includes('Plotly.newPlot') || content.includes('Plotly.react'))) {
      try {
        // Execute the Plotly script
        eval(content);
      } catch (e) {
        console.warn('Error re-executing Plotly script:', e);
      }
    }
  });
})

