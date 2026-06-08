function byId(id) {
    const element = document.getElementById(id);
    if (!element) {
        throw new Error(`Missing required element #${id}`);
    }
    return element;
}

export function getDom() {
    return {
        app: byId("app"),
        statusLine: byId("statusLine"),
        emptyState: byId("emptyState"),
        discoveryTotal: byId("discoveryTotal"),
        selectionTotal: byId("selectionTotal"),
        entriesList: byId("entriesList"),
        searchInput: byId("searchInput"),
        displayLimitSelect: byId("displayLimitSelect"),
        displayLimitCustom: byId("displayLimitCustom"),
        displayLimitApplyButton: byId("displayLimitApplyButton"),
        fitViewButton: byId("fitViewButton"),
        refreshButton: byId("refreshButton"),
        recomputeLayoutButton: byId("recomputeLayoutButton"),
        clearSelectionButton: byId("clearSelectionButton"),
        exportButton: byId("exportButton"),
        previewSizeSlider: byId("previewSizeSlider"),
        previewSizeValue: byId("previewSizeValue"),
        discoveriesTab: byId("discoveriesTab"),
        coverageTab: byId("coverageTab"),
        viewerPage: byId("viewerPage"),
        coveragePage: byId("coveragePage"),
        reloadCoverageButton: byId("reloadCoverageButton"),
        coverageSubtitle: byId("coverageSubtitle"),
        coverageStats: byId("coverageStats"),
        coverageGrid: byId("coverageGrid"),
        coverageEmpty: byId("coverageEmpty"),
        previewCard: byId("previewCard"),
        previewVideo: byId("hoverVideo"),
        previewImage: byId("hoverImage"),
        previewMeta: byId("previewMeta"),
        graphLightbox: byId("graphLightbox"),
        graphLightboxTitle: byId("graphLightboxTitle"),
        graphLightboxImage: byId("graphLightboxImage"),
        graphLightboxClose: byId("graphLightboxClose"),
    };
}
