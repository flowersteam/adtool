import { exportDiscoveries, requestLayoutRecompute } from "./js/api.js";
import { createCoverageController } from "./js/coverage.js";
import { createDiscoveryMap } from "./js/discovery-map.js";
import { createDisplayLimitController } from "./js/display-limit.js";
import { getDom } from "./js/dom.js";
import { createGraphLightbox } from "./js/lightbox.js";
import { createPreviewController } from "./js/preview.js";

const elements = getDom();

function updateStatus(text) {
    elements.statusLine.textContent = text;
}

const preview = createPreviewController(elements);
const lightbox = createGraphLightbox(elements);
const coverage = createCoverageController({ elements, lightbox });
const discoveryMap = createDiscoveryMap({ elements, preview, updateStatus });
const displayLimit = createDisplayLimitController({
    elements,
    refreshDiscoveries: discoveryMap.refreshDiscoveries,
    updateStatus,
});

function showPage(pageName) {
    const isCoverage = pageName === "coverage";
    if (isCoverage && !coverage.isEnabled()) {
        return;
    }

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

async function recomputeLayout() {
    elements.recomputeLayoutButton.disabled = true;
    elements.refreshButton.disabled = true;
    try {
        updateStatus("Recomputing clustered layout...");
        await requestLayoutRecompute();
        await discoveryMap.refreshDiscoveries(true);
        updateStatus("Clustered layout recomputed.");
    } catch {
        updateStatus("Failed to recompute clustered layout.");
    } finally {
        elements.recomputeLayoutButton.disabled = false;
        elements.refreshButton.disabled = false;
    }
}

async function exportEntries() {
    const selectedEntries = discoveryMap.selectedEntries();
    if (selectedEntries.length === 0) {
        updateStatus("No entries selected for export.");
        return;
    }

    elements.exportButton.disabled = true;
    updateStatus("Exporting selected entries...");
    try {
        const payload = await exportDiscoveries(selectedEntries);
        updateStatus(`Export complete: ${payload.new_dir}`);
    } catch {
        updateStatus("Export failed. Check server logs.");
    } finally {
        elements.exportButton.disabled = false;
    }
}

function bindEvents() {
    elements.discoveriesTab.addEventListener("click", () => showPage("discoveries"));
    elements.coverageTab.addEventListener("click", () => showPage("coverage"));
    elements.reloadCoverageButton.addEventListener("click", coverage.load);
    elements.fitViewButton.addEventListener("click", discoveryMap.fitView);
    elements.refreshButton.addEventListener("click", () => discoveryMap.refreshDiscoveries(false));
    elements.recomputeLayoutButton.addEventListener("click", recomputeLayout);
    elements.clearSelectionButton.addEventListener("click", discoveryMap.clearSelection);
    elements.exportButton.addEventListener("click", exportEntries);
    elements.searchInput.addEventListener("input", discoveryMap.applyFilter);
    elements.displayLimitSelect.addEventListener("change", displayLimit.syncCustomInputVisibility);
    elements.displayLimitApplyButton.addEventListener("click", displayLimit.apply);
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
    coverage.initializeNavigation(() => showPage("discoveries"));
    displayLimit.initialize();
    discoveryMap.resizeRenderer();
    discoveryMap.refreshDiscoveries(true).then(() => {
        discoveryMap.markLiveRefreshNow();
    });
    discoveryMap.connectWebsocket();
    discoveryMap.startAnimation();
}

initialize();
