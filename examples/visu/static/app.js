import {
    exportDiscoveries,
    requestLayoutRecompute,
    runCoverageComparison,
    runRandomRun,
} from "./js/api.js";
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

function trimmedValue(element) {
    return element.value.trim();
}

async function launchRandomRun() {
    const configPath = trimmedValue(elements.randomConfigPath);
    if (!configPath) {
        updateStatus("Random config path is required.");
        elements.randomConfigPath.focus();
        return;
    }

    elements.randomRunButton.disabled = true;
    updateStatus("Running random baseline...");
    try {
        const payload = await runRandomRun({
            config_file: configPath,
            nb_iterations: elements.randomIterationsInput.value,
            seed: elements.randomSeedInput.value,
        });
        elements.coverageComparePath.value = payload.discoveries_dir;
        updateStatus(`Random run complete: ${payload.discoveries_dir}`);
    } catch (error) {
        updateStatus(error.message || "Random run failed. Check server logs.");
    } finally {
        elements.randomRunButton.disabled = false;
    }
}

async function launchCoverageComparison() {
    const comparisonPath = trimmedValue(elements.coverageComparePath);
    if (!comparisonPath) {
        updateStatus("Compare path is required.");
        elements.coverageComparePath.focus();
        return;
    }

    const configFile = trimmedValue(elements.coverageConfigPath);
    const resolvedConfigFile = configFile.toLowerCase() === "none" ? "" : configFile;
    const labelA = trimmedValue(elements.coverageLabelA) || "IMGEP";
    const labelB = trimmedValue(elements.coverageLabelB) || "baseline";

    elements.coverageCompareButton.disabled = true;
    elements.reloadCoverageButton.disabled = true;
    updateStatus("Running coverage comparison...");
    try {
        const payload = await runCoverageComparison({
            path: comparisonPath,
            config_file: resolvedConfigFile || null,
            label_a: labelA,
            label_b: labelB,
        });
        coverage.setEnabled(true);
        updateStatus(`Coverage comparison complete: ${payload.run_dir}`);
        showPage("coverage");
    } catch (error) {
        updateStatus(error.message || "Coverage comparison failed. Check server logs.");
    } finally {
        elements.coverageCompareButton.disabled = false;
        elements.reloadCoverageButton.disabled = false;
    }
}

function toggleCoverageActions() {
    const collapsed = elements.coverageActionsBody.hidden;
    elements.coverageActionsBody.hidden = !collapsed;
    elements.coverageActionsToggle.setAttribute("aria-expanded", String(collapsed));
    elements.coverageActionsToggle.textContent = collapsed ? "Hide" : "Show";
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
    elements.coverageActionsToggle.addEventListener("click", toggleCoverageActions);
    elements.randomRunButton.addEventListener("click", launchRandomRun);
    elements.coverageCompareButton.addEventListener("click", launchCoverageComparison);
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
