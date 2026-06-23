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
    runtimeStatusText: "runtimeStatusText",
    pauseExperimentButton: "pauseExperimentButton",
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
    renderSettingsControl: "renderSettingsControl",
    stickerPreviewSizeInput: "stickerPreviewSizeInput",
    stickerPreviewSizeValue: "stickerPreviewSizeValue",
    highlightRulesSection: "highlightRulesSection",
    highlightRulesList: "highlightRulesList",
    computeHighlightFiltersButton: "computeHighlightFiltersButton",
    highlightRulesToggleButton: "highlightRulesToggleButton",
    analysisPanelBody: "analysisPanelBody",
    analysisPanelToggle: "analysisPanelToggle",
    randomConfigPath: "randomConfigPath",
    randomIterationsInput: "randomIterationsInput",
    randomSeedInput: "randomSeedInput",
    randomRunButton: "randomRunButton",
    analysisTargetPath: "analysisTargetPath",
    analysisComparisonList: "analysisComparisonList",
    addAnalysisComparisonButton: "addAnalysisComparisonButton",
    analysisConfigPath: "analysisConfigPath",
    analysisLabelA: "analysisLabelA",
    analysisLabelB: "analysisLabelB",
    runAnalysisButton: "runAnalysisButton",
    fitViewButton: "fitViewButton",
    refreshButton: "refreshButton",
    recomputeLayoutButton: "recomputeLayoutButton",
    clearSelectionButton: "clearSelectionButton",
    exportButton: "exportButton",
    previewSizeSlider: "previewSizeSlider",
    previewSizeValue: "previewSizeValue",
    discoveriesTab: "discoveriesTab",
    analysisTab: "analysisTab",
    viewerPage: "viewerPage",
    analysisPage: "analysisPage",
    reloadAnalysisButton: "reloadAnalysisButton",
    analysisSubtitle: "analysisSubtitle",
    analysisStats: "analysisStats",
    analysisGrid: "analysisGrid",
    analysisEmpty: "analysisEmpty",
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
