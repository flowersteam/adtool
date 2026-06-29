export function createPageRouter({
    analysis,
    discoveryMap,
    elements,
    preview,
}) {
    function showPage(pageName) {
        const isAnalysis = pageName === "analysis";

        elements.viewerPage.classList.toggle("active", !isAnalysis);
        elements.analysisPage.classList.toggle("active", isAnalysis);
        elements.discoveriesTab.classList.toggle("active", !isAnalysis);
        elements.analysisTab.classList.toggle("active", isAnalysis);

        if (isAnalysis) {
            preview.hide();
            analysis.load();
        } else {
            discoveryMap.resizeRenderer();
        }
    }

    return { showPage };
}
