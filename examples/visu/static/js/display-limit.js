import { getDisplayLimit, setDisplayLimit } from "./api.js";

export function createDisplayLimitController({
    elements,
    refreshDiscoveries,
    updateStatus,
}) {
    function updateInputs(limit) {
        const value = String(limit);
        const hasPreset = Array.from(elements.displayLimitSelect.options)
            .some((option) => option.value === value);
        elements.displayLimitSelect.value = hasPreset ? value : "custom";
        elements.displayLimitCustom.hidden = hasPreset;
        elements.displayLimitCustom.value = value;
    }

    function selectedLimit() {
        const rawValue = elements.displayLimitSelect.value === "custom"
            ? elements.displayLimitCustom.value
            : elements.displayLimitSelect.value;
        const limit = Number.parseInt(rawValue, 10);
        return Number.isFinite(limit) ? limit : null;
    }

    async function initialize() {
        try {
            const payload = await getDisplayLimit();
            updateInputs(payload.limit || 500);
        } catch {
            updateInputs(500);
        }
    }

    async function apply() {
        const limit = selectedLimit();
        if (limit === null) {
            updateStatus("Display limit must be a number.");
            return;
        }

        elements.displayLimitApplyButton.disabled = true;
        try {
            updateStatus(`Updating display limit to ${limit} discoveries...`);
            await setDisplayLimit(limit);
            updateInputs(limit);
            await refreshDiscoveries(true);
            updateStatus(`Displaying up to ${limit} discoveries.`);
        } catch (error) {
            updateStatus(error.message || "Failed to update display limit.");
        } finally {
            elements.displayLimitApplyButton.disabled = false;
        }
    }

    function syncCustomInputVisibility() {
        elements.displayLimitCustom.hidden = elements.displayLimitSelect.value !== "custom";
    }

    return {
        apply,
        initialize,
        syncCustomInputVisibility,
    };
}
