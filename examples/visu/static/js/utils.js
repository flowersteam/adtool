const VIDEO_SUFFIX = /\.mp4$/i;

export function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

export function sleep(ms) {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

export function normalizeVisualPath(path) {
    return String(path || "").replace(/^\/+/, "");
}

export function mediaUrl(visualPath) {
    return `/discoveries/${normalizeVisualPath(visualPath)}`;
}

export function prettifyEntryLabel(src) {
    const normalized = src.replace(/^\/discoveries\/+/, "");
    const parts = normalized.split("/");
    if (parts.length <= 1) {
        return normalized;
    }
    return `${parts[0]} / ${parts[parts.length - 1]}`;
}

function replaceVideoSuffix(path, replacement) {
    return String(path).replace(VIDEO_SUFFIX, replacement);
}

function previewImagePath(path, replacement) {
    return VIDEO_SUFFIX.test(path) ? replaceVideoSuffix(path, replacement) : path;
}

export function visualToPreviewImage(visualPath) {
    const normalized = normalizeVisualPath(visualPath);
    return previewImagePath(normalized, ".jpg");
}

export function fallbackPreviewImage(visualPath) {
    const normalized = normalizeVisualPath(visualPath);
    return previewImagePath(normalized, ".png");
}

export function selectedEntryPreviewPath(sourcePath) {
    return previewImagePath(sourcePath, ".jpg");
}

export function selectedEntryPreviewFallback(sourcePath) {
    return previewImagePath(sourcePath, ".png");
}

export function isVideoPath(path) {
    return VIDEO_SUFFIX.test(path);
}

export function formatNumber(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return "0";
    }
    return new Intl.NumberFormat().format(value);
}

export function formatRange(bounds) {
    if (!Array.isArray(bounds) || bounds.length < 2) {
        return "";
    }

    const [min, max] = bounds;
    if (!Number.isFinite(Number(min)) || !Number.isFinite(Number(max))) {
        return "";
    }

    return `${Number(min).toPrecision(3)} to ${Number(max).toPrecision(3)}`;
}

function regexMatcher(regex) {
    return (label) => {
        regex.lastIndex = 0;
        return regex.test(label);
    };
}

export function buildDiscoveryMatcher(rawQuery) {
    const query = rawQuery.trim();
    if (query === "") {
        return {
            error: "",
            matcher: () => true,
            mode: "empty",
        };
    }

    try {
        if (query.startsWith("re:")) {
            const regex = new RegExp(query.slice(3), "i");
            return { error: "", matcher: regexMatcher(regex), mode: "regex" };
        }

        if (query.length > 2 && query.startsWith("/") && query.lastIndexOf("/") > 0) {
            const lastSlash = query.lastIndexOf("/");
            const pattern = query.slice(1, lastSlash);
            const flags = query.slice(lastSlash + 1) || "i";
            const regex = new RegExp(pattern, flags.includes("i") ? flags : `${flags}i`);
            return { error: "", matcher: regexMatcher(regex), mode: "regex" };
        }
    } catch (error) {
        return {
            error: error.message || "Invalid search pattern.",
            matcher: () => true,
            mode: "invalid",
        };
    }

    const text = query.toLowerCase();
    return {
        error: "",
        matcher: (label) => label.includes(text),
        mode: "text",
    };
}
