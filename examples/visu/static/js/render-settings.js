import { getRenderSettings, setRenderSettings } from "./api.js";

const APPLY_DEBOUNCE_MS = 120;
const DEFAULT_RENDER_SETTINGS = {
    sticker_preview_world_height: 0.6,
};

export function createRenderSettingsController({
    discoveryMap,
    elements,
    updateStatus,
}) {
    let hybridActive = false;
    let applyTimerId = null;
    let applyVersion = 0;

    function syncLabels() {
        const previewHeight = Number.parseFloat(elements.stickerPreviewSizeInput.value);

        elements.stickerPreviewSizeValue.textContent = Number.isFinite(previewHeight)
            ? previewHeight.toFixed(2)
            : "";
    }

    function updateBounds(payload) {
        if (Number.isFinite(Number(payload.min_sticker_preview_world_height))) {
            elements.stickerPreviewSizeInput.min = `${payload.min_sticker_preview_world_height}`;
        }
        if (Number.isFinite(Number(payload.max_sticker_preview_world_height))) {
            elements.stickerPreviewSizeInput.max = `${payload.max_sticker_preview_world_height}`;
        }
    }

    function updateInputs(payload = DEFAULT_RENDER_SETTINGS) {
        updateBounds(payload);
        elements.stickerPreviewSizeInput.value = `${payload.sticker_preview_world_height ?? DEFAULT_RENDER_SETTINGS.sticker_preview_world_height}`;
        syncLabels();
    }

    function selectedSettings() {
        const stickerPreviewWorldHeight = Number.parseFloat(elements.stickerPreviewSizeInput.value);

        if (!Number.isFinite(stickerPreviewWorldHeight)) {
            return { error: "Sticker preview size must be a number." };
        }

        return {
            error: "",
            payload: {
                sticker_preview_world_height: stickerPreviewWorldHeight,
            },
        };
    }

    async function applyCurrentSettings(version) {
        const selection = selectedSettings();
        if (selection.error) {
            updateStatus(selection.error);
            return;
        }

        await discoveryMap.setRenderSettings(selection.payload);
        if (version !== applyVersion) {
            return;
        }

        try {
            await setRenderSettings(selection.payload);
        } catch (error) {
            updateStatus(error.message || "Failed to update sticker settings.");
        }
    }

    function scheduleApply() {
        syncLabels();
        if (!hybridActive) {
            return;
        }

        if (applyTimerId !== null) {
            clearTimeout(applyTimerId);
        }

        const version = ++applyVersion;
        applyTimerId = window.setTimeout(() => {
            applyTimerId = null;
            applyCurrentSettings(version);
        }, APPLY_DEBOUNCE_MS);
    }

    function setHybridActive(active) {
        hybridActive = active;
        elements.renderSettingsControl.hidden = !hybridActive;
        if (!hybridActive && applyTimerId !== null) {
            clearTimeout(applyTimerId);
            applyTimerId = null;
        }
    }

    async function initialize() {
        try {
            const payload = await getRenderSettings();
            updateInputs(payload);
            await discoveryMap.setRenderSettings(payload);
        } catch {
            updateInputs(DEFAULT_RENDER_SETTINGS);
            await discoveryMap.setRenderSettings(DEFAULT_RENDER_SETTINGS);
        }
    }

    return {
        initialize,
        scheduleApply,
        setHybridActive,
        syncLabels,
    };
}
