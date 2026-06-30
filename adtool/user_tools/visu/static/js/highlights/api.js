const EMPTY_HIGHLIGHTS = {
    fields: [],
    filters_detected: false,
    rules: [],
    storage_key: "",
};

export async function readDiscoveryHighlights() {
    try {
        const response = await fetch("/static/discovery_highlights.json", { cache: "no-store" });
        if (!response.ok) {
            return EMPTY_HIGHLIGHTS;
        }

        const payload = await response.json();
        return {
            fields: Array.isArray(payload.fields) ? payload.fields : [],
            filters_detected: Boolean(payload.filters_detected),
            rules: Array.isArray(payload.rules) ? payload.rules : [],
            storage_key: typeof payload.storage_key === "string" ? payload.storage_key : "",
        };
    } catch {
        return EMPTY_HIGHLIGHTS;
    }
}
