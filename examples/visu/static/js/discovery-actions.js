import { exportDiscoveries, requestLayoutRecompute } from "./api.js";

export function createDiscoveryActions({
    discoveryMap,
    elements,
    updateStatus,
}) {
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

    return {
        exportEntries,
        recomputeLayout,
    };
}
