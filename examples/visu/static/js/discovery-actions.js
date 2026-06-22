import {
    exportDiscoveries,
    materializeDiscoveryFilters,
    requestLayoutRecompute,
} from "./api.js";

export function createDiscoveryActions({
    discoveryMap,
    elements,
    updateStatus,
}) {
    async function recomputeLayout() {
        elements.recomputeLayoutButton.disabled = true;
        elements.refreshButton.disabled = true;
        try {
            updateStatus("Recomputing layout...");
            await requestLayoutRecompute();
            await discoveryMap.refreshDiscoveries(true);
            updateStatus("Layout recomputed.");
        } catch (error) {
            updateStatus(error.message || "Failed to recompute layout.");
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

    async function computeHighlightFilters() {
        elements.computeHighlightFiltersButton.disabled = true;
        elements.recomputeLayoutButton.disabled = true;
        elements.refreshButton.disabled = true;
        try {
            updateStatus("Computing discovery filters...");
            const payload = await materializeDiscoveryFilters();
            await discoveryMap.refreshDiscoveries(false);
            updateStatus(`Discovery filters computed for ${payload.updated_count} discoveries.`);
        } catch (error) {
            updateStatus(error.message || "Failed to compute discovery filters.");
        } finally {
            elements.computeHighlightFiltersButton.disabled = false;
            elements.recomputeLayoutButton.disabled = false;
            elements.refreshButton.disabled = false;
        }
    }

    return {
        computeHighlightFilters,
        exportEntries,
        recomputeLayout,
    };
}
