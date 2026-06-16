import { createAnalysisActions } from "./js/analysis-actions.js";
import { createAnalysisController } from "./js/analysis.js";
import { createDiscoveryActions } from "./js/discovery-actions.js";
import { createDiscoveryMap } from "./js/discovery-map.js";
import { createDisplayLimitController } from "./js/display-limit.js";
import { getDom } from "./js/dom.js";
import { createGraphLightbox } from "./js/lightbox.js";
import { createPageRouter } from "./js/page-router.js";
import { createPreviewController } from "./js/preview.js";
import { createProjectionController } from "./js/projection.js";
import { createRenderSettingsController } from "./js/render-settings.js";

const elements = getDom();

function updateStatus(text) {
    elements.statusLine.textContent = text;
}

const preview = createPreviewController(elements);
const lightbox = createGraphLightbox(elements);
const analysis = createAnalysisController({ elements, lightbox });
const discoveryMap = createDiscoveryMap({ elements, preview, updateStatus });
const router = createPageRouter({ analysis, discoveryMap, elements, preview });
const displayLimit = createDisplayLimitController({
    elements,
    refreshDiscoveries: discoveryMap.refreshDiscoveries,
    updateStatus,
});
const projection = createProjectionController({
    elements,
    refreshDiscoveries: discoveryMap.refreshDiscoveries,
    updateStatus,
});
const renderSettings = createRenderSettingsController({
    discoveryMap,
    elements,
    updateStatus,
});
const discoveryActions = createDiscoveryActions({ discoveryMap, elements, updateStatus });
const analysisActions = createAnalysisActions({
    analysis,
    elements,
    showPage: router.showPage,
    updateStatus,
});

function bindEvents() {
    elements.discoveriesTab.addEventListener("click", () => router.showPage("discoveries"));
    elements.analysisTab.addEventListener("click", () => router.showPage("analysis"));
    elements.reloadAnalysisButton.addEventListener("click", analysis.load);
    elements.fitViewButton.addEventListener("click", discoveryMap.fitView);
    elements.refreshButton.addEventListener("click", () => discoveryMap.refreshDiscoveries(false));
    elements.recomputeLayoutButton.addEventListener("click", discoveryActions.recomputeLayout);
    elements.clearSelectionButton.addEventListener("click", discoveryMap.clearSelection);
    elements.exportButton.addEventListener("click", discoveryActions.exportEntries);
    elements.searchInput.addEventListener("input", discoveryMap.applyFilter);
    elements.displayLimitSelect.addEventListener("change", displayLimit.syncCustomInputVisibility);
    elements.displayLimitApplyButton.addEventListener("click", displayLimit.apply);
    elements.viewModeControl.addEventListener("click", async (event) => {
        const button = event.target.closest("[data-render-mode]");
        if (button) {
            await discoveryMap.setRenderMode(button.dataset.renderMode);
            renderSettings.setHybridActive(button.dataset.renderMode === "hybrid");
        }
    });
    elements.projectionMethodSelect.addEventListener("change", projection.syncAxisVisibility);
    elements.projectionApplyButton.addEventListener("click", projection.apply);
    elements.stickerPreviewSizeInput.addEventListener("input", renderSettings.scheduleApply);
    elements.analysisPanelToggle.addEventListener("click", analysisActions.toggleAnalysisPanel);
    elements.randomRunButton.addEventListener("click", analysisActions.launchRandomRun);
    elements.runAnalysisButton.addEventListener("click", analysisActions.launchAnalysis);
    elements.previewSizeSlider.addEventListener("input", (event) => {
        preview.applyScale(event.target.value);
    });
    elements.graphLightboxClose.addEventListener("click", lightbox.close);
    elements.graphLightbox.addEventListener("click", (event) => {
        if (event.target === elements.graphLightbox) {
            lightbox.close();
        }
    });

    window.addEventListener("resize", discoveryMap.resizeRenderer);
    window.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            lightbox.close();
            preview.hide();
            elements.searchInput.blur();
        }
    });

    new ResizeObserver(discoveryMap.resizeRenderer).observe(elements.app);
}

async function initialize() {
    bindEvents();
    preview.applyScale(elements.previewSizeSlider.value);
    analysis.initializeNavigation();
    await displayLimit.initialize();
    await projection.initialize();
    await renderSettings.initialize();
    renderSettings.setHybridActive(false);
    discoveryMap.resizeRenderer();
    discoveryMap.refreshDiscoveries(true).then(() => {
        discoveryMap.markLiveRefreshNow();
    });
    discoveryMap.connectWebsocket();
    discoveryMap.startAnimation();
}

initialize();
