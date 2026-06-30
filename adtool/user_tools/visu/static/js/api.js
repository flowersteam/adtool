import { LOAD_RETRY_MS } from "./config.js";
import { readDiscoveryHighlights as readHighlightsPayload } from "./highlights/api.js";
import { sleep } from "./utils.js";

export async function readDiscoveries(maxWaitMs = 0, onWaiting = () => {}) {
    const deadline = performance.now() + maxWaitMs;
    let lastError = null;

    while (true) {
        try {
            const response = await fetch("/static/discoveries.json", { cache: "no-store" });
            if (!response.ok) {
                throw new Error("discoveries.json unavailable");
            }

            const pointsData = await response.json();
            if (!Array.isArray(pointsData)) {
                throw new Error("invalid discoveries format");
            }

            if (pointsData.length > 0 || performance.now() >= deadline) {
                return pointsData;
            }
        } catch (error) {
            lastError = error;
            if (performance.now() >= deadline) {
                throw lastError;
            }
        }

        onWaiting();
        await sleep(LOAD_RETRY_MS);
    }
}

export async function readDiscoveryHighlights() {
    return readHighlightsPayload();
}

export async function getAnalysisStatus() {
    return getJson("/analysis_status", "analysis status unavailable");
}

export async function getAnalysisRuns() {
    return fetch("/analysis_runs", { cache: "no-store" });
}

export async function responseErrorMessage(response, fallback = "Analysis summary could not be loaded.") {
    try {
        const payload = await response.json();
        if (typeof payload.detail === "string") {
            return payload.detail;
        }
    } catch {
        // Fall through to the generic message below.
    }
    return fallback;
}

async function getJson(url, fallbackMessage) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
        throw new Error(await responseErrorMessage(response, fallbackMessage));
    }
    return response.json();
}

async function postJson(url, payload, fallbackMessage) {
    const request = {
        method: "POST",
    };
    if (payload !== undefined) {
        request.headers = { "Content-Type": "application/json" };
        request.body = JSON.stringify(payload);
    }

    const response = await fetch(url, request);

    if (!response.ok) {
        throw new Error(await responseErrorMessage(response, fallbackMessage));
    }

    return response.json();
}

export async function getDisplayLimit() {
    return getJson("/display_limit", "display limit unavailable");
}

export async function getRuntimeStatus() {
    return getJson("/runtime_status", "runtime status unavailable");
}

export async function getExperimentControl() {
    return getJson("/experiment_control", "experiment control unavailable");
}

export async function setExperimentControl(payload) {
    return postJson("/experiment_control", payload, "experiment control update failed");
}

export async function getGoalTargeting() {
    return getJson("/goal_targeting", "goal targeting unavailable");
}

export async function setGoalTargeting(payload) {
    return postJson("/goal_targeting", payload, "goal targeting update failed");
}

export async function setDisplayLimit(limit, selectedSources = []) {
    return postJson("/display_limit", { limit, selected_sources: selectedSources }, "display limit update failed");
}

export async function getProjection() {
    return getJson("/projection", "projection settings unavailable");
}

export async function setProjection(payload, selectedSources = []) {
    return postJson(
        "/projection",
        { ...payload, selected_sources: selectedSources },
        "projection update failed",
    );
}

export async function getRenderSettings() {
    return getJson("/render_settings", "render settings unavailable");
}

export async function setRenderSettings(payload) {
    return postJson("/render_settings", payload, "render settings update failed");
}

export async function requestLayoutRecompute(selectedSources = []) {
    return postJson(
        "/recompute_layout",
        { selected_sources: selectedSources },
        "layout recompute failed",
    );
}

export async function materializeDiscoveryFilters(selectedSources = []) {
    return postJson(
        "/discovery_highlights/materialize",
        { selected_sources: selectedSources },
        "highlight filter computation failed",
    );
}

export async function exportDiscoveries(files) {
    return postJson("/export", files, "export failed");
}

export async function runRandomRun(payload) {
    return postJson("/analysis/random_run", payload, "random run failed");
}

export async function runAnalysis(payload) {
    return postJson("/analysis/run", payload, "analysis run failed");
}
