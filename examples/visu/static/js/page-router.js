export function createPageRouter({
    coverage,
    discoveryMap,
    elements,
    preview,
}) {
    function showPage(pageName) {
        const isCoverage = pageName === "coverage";

        elements.viewerPage.classList.toggle("active", !isCoverage);
        elements.coveragePage.classList.toggle("active", isCoverage);
        elements.discoveriesTab.classList.toggle("active", !isCoverage);
        elements.coverageTab.classList.toggle("active", isCoverage);

        if (isCoverage) {
            preview.hide();
            coverage.load();
        } else {
            discoveryMap.resizeRenderer();
        }
    }

    return { showPage };
}
