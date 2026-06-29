import {
    getRuntimeStatus,
    setExperimentControl,
} from "./api.js";

const RUNTIME_STATUS_POLL_MS = 5000;

export function createRuntimeControls({
    discoveryMap,
    elements,
    updateStatus,
}) {
    let currentStatus = null;
    let pollTimerId = null;
    let isUpdatingPause = false;

    function runtimeLabel(status) {
        if (status.refresh) {
            if (status.paused) {
                return "Experiment paused";
            }
            return `Online update ${status.point_update_interval_seconds}s · full recompute ${status.full_recompute_interval_seconds}s`;
        }
        return "Manual update mode";
    }

    function applyStatus(status) {
        currentStatus = status;
        elements.runtimeStatusText.hidden = false;
        elements.runtimeStatusText.textContent = runtimeLabel(status);
        elements.pauseExperimentButton.hidden = !status.pause_supported;
        elements.pauseExperimentButton.textContent = status.paused
            ? "Resume Experiment"
            : "Pause Experiment";
        if (status.refresh) {
            discoveryMap.setLiveRefreshCooldown(status.point_update_interval_seconds * 1000);
        }
    }

    async function refreshStatus() {
        const status = await getRuntimeStatus();
        applyStatus(status);
    }

    async function togglePause() {
        if (!currentStatus?.pause_supported || isUpdatingPause) {
            return;
        }

        isUpdatingPause = true;
        elements.pauseExperimentButton.disabled = true;
        try {
            const nextPaused = !currentStatus.paused;
            const payload = await setExperimentControl({ paused: nextPaused });
            applyStatus({
                ...currentStatus,
                paused: payload.paused,
            });
            updateStatus(payload.paused ? "Experiment paused." : "Experiment resumed.");
        } catch (error) {
            updateStatus(error.message || "Experiment control update failed.");
        } finally {
            isUpdatingPause = false;
            elements.pauseExperimentButton.disabled = false;
        }
    }

    async function initialize() {
        try {
            await refreshStatus();
        } catch (error) {
            elements.runtimeStatusText.hidden = false;
            elements.runtimeStatusText.textContent = "Runtime status unavailable";
            updateStatus(error.message || "Runtime status unavailable.");
        }

        pollTimerId = window.setInterval(async () => {
            try {
                await refreshStatus();
            } catch {
                // Ignore transient polling failures.
            }
        }, RUNTIME_STATUS_POLL_MS);
    }

    return {
        initialize,
        togglePause,
    };
}
