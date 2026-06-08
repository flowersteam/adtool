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
    const response = await fetch("/coverage_status", { cache: "no-store" });
    if (!response.ok) {
        throw new Error("coverage status unavailable");
    }
    return response.json();
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

async function postJson(url, payload, fallbackMessage) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        throw new Error(await responseErrorMessage(response, fallbackMessage));
    }

    return response.json();
}

export async function getDisplayLimit() {
    const response = await fetch("/display_limit", { cache: "no-store" });
    if (!response.ok) {
        throw new Error("display limit unavailable");
    }
    return response.json();
}

export async function setDisplayLimit(limit) {
    const response = await fetch("/display_limit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit }),
    });

    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "display limit update failed");
    }

    return response.json();
}

export async function requestLayoutRecompute() {
    const response = await fetch("/recompute_layout", { method: "POST" });
    if (!response.ok) {
        throw new Error("layout recompute failed");
    }
    return response.json();
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
