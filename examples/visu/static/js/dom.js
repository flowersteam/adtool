function byId(id) {
    const element = document.getElementById(id);
    if (!element) {
        throw new Error(`Missing required element #${id}`);
    }
    return element;
}

const DOM_IDS = {
    app: "app",
    statusLine: "statusLine",
    emptyState: "emptyState",
    discoveryTotal: "discoveryTotal",
    selectionTotal: "selectionTotal",
    entriesList: "entriesList",
    searchInput: "searchInput",
    displayLimitSelect: "displayLimitSelect",
    displayLimitCustom: "displayLimitCustom",
    displayLimitApplyButton: "displayLimitApplyButton",
    viewModeControl: "viewModeControl",
    projectionControl: "projectionControl",
    projectionMethodSelect: "projectionMethodSelect",
    projectionAxisXInput: "projectionAxisXInput",
    projectionAxisYInput: "projectionAxisYInput",
    projectionApplyButton: "projectionApplyButton",
    coverageActionsBody: "coverageActionsBody",
    coverageActionsToggle: "coverageActionsToggle",
    randomConfigPath: "randomConfigPath",
    randomIterationsInput: "randomIterationsInput",
    randomSeedInput: "randomSeedInput",
    randomRunButton: "randomRunButton",
    coverageComparePath: "coverageComparePath",
    coverageConfigPath: "coverageConfigPath",
    coverageLabelA: "coverageLabelA",
    coverageLabelB: "coverageLabelB",
    coverageCompareButton: "coverageCompareButton",
    fitViewButton: "fitViewButton",
    refreshButton: "refreshButton",
    recomputeLayoutButton: "recomputeLayoutButton",
    clearSelectionButton: "clearSelectionButton",
    exportButton: "exportButton",
    previewSizeSlider: "previewSizeSlider",
    previewSizeValue: "previewSizeValue",
    discoveriesTab: "discoveriesTab",
    coverageTab: "coverageTab",
    viewerPage: "viewerPage",
    coveragePage: "coveragePage",
    reloadCoverageButton: "reloadCoverageButton",
    coverageSubtitle: "coverageSubtitle",
    coverageStats: "coverageStats",
    coverageGrid: "coverageGrid",
    coverageEmpty: "coverageEmpty",
    previewCard: "previewCard",
    previewVideo: "hoverVideo",
    previewImage: "hoverImage",
    previewMeta: "previewMeta",
    graphLightbox: "graphLightbox",
    graphLightboxTitle: "graphLightboxTitle",
    graphLightboxImage: "graphLightboxImage",
    graphLightboxClose: "graphLightboxClose",
};

export function getDom() {
    return Object.fromEntries(
        Object.entries(DOM_IDS).map(([key, id]) => [key, byId(id)]),
    );
}
