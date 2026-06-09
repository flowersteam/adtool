import { LOAD_RETRY_MS } from "./config.js";
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

export async function getCoverageStatus() {
    return getJson("/coverage_status", "coverage status unavailable");
}

export async function getCoverageRuns() {
    return fetch("/coverage_runs", { cache: "no-store" });
}

export async function responseErrorMessage(response, fallback = "Coverage summary could not be loaded.") {
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

export async function setDisplayLimit(limit) {
    return postJson("/display_limit", { limit }, "display limit update failed");
}

export async function getProjection() {
    return getJson("/projection", "projection settings unavailable");
}

export async function setProjection(payload) {
    return postJson("/projection", payload, "projection update failed");
}

export async function requestLayoutRecompute() {
    return postJson("/recompute_layout", undefined, "layout recompute failed");
}

export async function exportDiscoveries(files) {
    return postJson("/export", files, "export failed");
}

export async function runRandomRun(payload) {
    return postJson("/analysis/random_run", payload, "random run failed");
}

export async function runCoverageComparison(payload) {
    return postJson("/analysis/coverage_comparison", payload, "coverage comparison failed");
}
