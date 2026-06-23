import { getProjection, setProjection } from "./api.js";

const DEFAULT_PROJECTION = {
    method: "umap",
    axes: [0, 1],
};

export function createProjectionController({
    discoveryMap,
    elements,
    onProjectionApplied = async () => {},
    refreshDiscoveries,
    updateStatus,
}) {
    function syncAxisVisibility() {
        const showAxes = elements.projectionMethodSelect.value === "axis";
        elements.projectionAxisXInput.hidden = !showAxes;
        elements.projectionAxisYInput.hidden = !showAxes;
        elements.projectionControl.classList.toggle("axesVisible", showAxes);
    }

    function updateInputs(payload = DEFAULT_PROJECTION) {
        const method = payload.method || DEFAULT_PROJECTION.method;
        const axes = Array.isArray(payload.axes) ? payload.axes : DEFAULT_PROJECTION.axes;
        elements.projectionMethodSelect.value = method;
        elements.projectionAxisXInput.value = `${axes[0] ?? 0}`;
        elements.projectionAxisYInput.value = `${axes[1] ?? 1}`;
        syncAxisVisibility();
    }

    function selectedProjection() {
        const method = elements.projectionMethodSelect.value;
        const payload = { method };

        if (method !== "axis") {
            return {
                error: "",
                payload,
            };
        }

        const xAxis = Number.parseInt(elements.projectionAxisXInput.value, 10);
        const yAxis = Number.parseInt(elements.projectionAxisYInput.value, 10);

        if (!Number.isFinite(xAxis) || !Number.isFinite(yAxis)) {
            return { error: "Axis ids must be integers." };
        }
        if (xAxis < 0 || yAxis < 0) {
            return { error: "Axis ids must be non-negative." };
        }
        if (method === "axis" && xAxis === yAxis) {
            return { error: "Axis ids must be different." };
        }

        return {
            error: "",
            payload: {
                ...payload,
                axes: [xAxis, yAxis],
            },
        };
    }

    async function initialize() {
        try {
            updateInputs(await getProjection());
        } catch {
            updateInputs(DEFAULT_PROJECTION);
        }
    }

    async function apply() {
        const selection = selectedProjection();
        if (selection.error) {
            updateStatus(selection.error);
            return;
        }

        elements.projectionApplyButton.disabled = true;
        elements.recomputeLayoutButton.disabled = true;
        try {
            updateStatus(`Applying ${selection.payload.method.toUpperCase()} projection...`);
            const payload = await setProjection(
                selection.payload,
                discoveryMap.selectedEntries(),
            );
            updateInputs(payload);
            await refreshDiscoveries(true);
            await onProjectionApplied(payload);
            updateStatus(`Projection set to ${payload.method.toUpperCase()}.`);
        } catch (error) {
            updateStatus(error.message || "Failed to update projection.");
        } finally {
            elements.projectionApplyButton.disabled = false;
            elements.recomputeLayoutButton.disabled = false;
        }
    }

    return {
        apply,
        initialize,
        syncAxisVisibility,
    };
}
