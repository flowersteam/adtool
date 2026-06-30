const STORAGE_VERSION = 1;
const STORAGE_PREFIX = "adtool.discovery_highlights";

function storageBucket(storageKey) {
    return `${STORAGE_PREFIX}:${storageKey || window.location.origin}`;
}

export function loadStoredHighlightRules(storageKey) {
    try {
        const payload = window.localStorage.getItem(storageBucket(storageKey));
        if (!payload) {
            return [];
        }

        const parsed = JSON.parse(payload);
        if (parsed.version !== STORAGE_VERSION || !Array.isArray(parsed.rules)) {
            return [];
        }
        return parsed.rules;
    } catch {
        return [];
    }
}

export function saveStoredHighlightRules(storageKey, rules) {
    try {
        window.localStorage.setItem(
            storageBucket(storageKey),
            JSON.stringify({
                version: STORAGE_VERSION,
                rules,
            }),
        );
    } catch {
        // Ignore storage failures and keep the in-memory state.
    }
}
