import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const SCALE_FACTOR = 30;
const POINT_OPACITY = 0.9;
const HOVER_OPACITY = 1.0;
const LOAD_RETRY_MS = 800;
const DISCOVERY_LOAD_GRACE_MS = 30000;
const COVERAGE_LOAD_GRACE_MS = 30000;
const LIVE_REFRESH_COOLDOWN_MS = 15000;
const cameraDepthBounds = { min: 2.2, max: 150.0 };

const app = document.getElementById("app");
const statusLine = document.getElementById("statusLine");
const emptyState = document.getElementById("emptyState");
const discoveryTotal = document.getElementById("discoveryTotal");
const selectionTotal = document.getElementById("selectionTotal");
const entriesList = document.getElementById("entriesList");
const searchInput = document.getElementById("searchInput");
const fitViewButton = document.getElementById("fitViewButton");
const refreshButton = document.getElementById("refreshButton");
const recomputeLayoutButton = document.getElementById("recomputeLayoutButton");
const clearSelectionButton = document.getElementById("clearSelectionButton");
const exportButton = document.getElementById("exportButton");
const previewSizeSlider = document.getElementById("previewSizeSlider");
const previewSizeValue = document.getElementById("previewSizeValue");

const discoveriesTab = document.getElementById("discoveriesTab");
const coverageTab = document.getElementById("coverageTab");
const viewerPage = document.getElementById("viewerPage");
const coveragePage = document.getElementById("coveragePage");
const reloadCoverageButton = document.getElementById("reloadCoverageButton");
const coverageSubtitle = document.getElementById("coverageSubtitle");
const coverageStats = document.getElementById("coverageStats");
const coverageGrid = document.getElementById("coverageGrid");
const coverageEmpty = document.getElementById("coverageEmpty");

const previewCard = document.getElementById("previewCard");
const previewVideo = document.getElementById("hoverVideo");
const previewImage = document.getElementById("hoverImage");
const previewMeta = document.getElementById("previewMeta");
const graphLightbox = document.getElementById("graphLightbox");
const graphLightboxTitle = document.getElementById("graphLightboxTitle");
const graphLightboxImage = document.getElementById("graphLightboxImage");
const graphLightboxClose = document.getElementById("graphLightboxClose");

const scene = new THREE.Scene();
scene.background = new THREE.Color("#eef0ec");

const camera = new THREE.PerspectiveCamera(54, 1, 0.01, 800);
camera.position.set(0, 0, 26);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
app.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enableRotate = false;
controls.screenSpacePanning = true;
controls.zoomSpeed = 1.05;
controls.panSpeed = 0.85;
controls.maxDistance = cameraDepthBounds.max;
controls.minDistance = cameraDepthBounds.min;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const textureLoader = new THREE.TextureLoader();

const planes = [];
const planesBySource = new Map();
const selectedEntries = new Set();
const selectedNodes = new Map();

let hoveredPlane = null;
let isRefreshing = false;
let pendingRefresh = false;
let coverageRequestId = 0;
let coverageEnabled = false;
let pointerDown = null;
let liveRefreshTimerId = null;
let lastLiveRefreshTimestamp = 0;

function updateStatus(text) {
    statusLine.textContent = text;
}

function setTotals() {
    const visibleCount = planes.filter((plane) => plane.visible).length;
    discoveryTotal.textContent = `${visibleCount}`;
    selectionTotal.textContent = `${selectedEntries.size}`;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function applyPreviewScale(value) {
    const scalePercent = clamp(Number(value) || 100, 70, 250);
    const width = Math.round(430 * (scalePercent / 100));
    const height = Math.round(270 * (scalePercent / 100));
    document.documentElement.style.setProperty("--preview-width", `${width}px`);
    document.documentElement.style.setProperty("--preview-height", `${height}px`);
    previewSizeSlider.value = `${scalePercent}`;
    previewSizeValue.textContent = `${scalePercent}%`;
}

function sleep(ms) {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

function normalizeVisualPath(path) {
    return String(path || "").replace(/^\/+/, "");
}

function mediaUrl(visualPath) {
    return `/discoveries/${normalizeVisualPath(visualPath)}`;
}

function prettifyEntryLabel(src) {
    const normalized = src.replace(/^\/discoveries\/+/, "");
    const parts = normalized.split("/");
    if (parts.length <= 1) {
        return normalized;
    }
    return `${parts[0]} / ${parts[parts.length - 1]}`;
}

function visualToPreviewImage(visualPath) {
    const normalized = normalizeVisualPath(visualPath);
    if (/\.mp4$/i.test(normalized)) {
        return normalized.replace(/\.mp4$/i, ".jpg");
    }
    return normalized;
}

function fallbackPreviewImage(visualPath) {
    const normalized = normalizeVisualPath(visualPath);
    if (/\.mp4$/i.test(normalized)) {
        return normalized.replace(/\.mp4$/i, ".png");
    }
    return normalized;
}

function isVideoPath(path) {
    return /\.mp4$/i.test(path);
}

function disposePlane(plane) {
    scene.remove(plane);
    if (plane.material?.map) {
        plane.material.map.dispose();
    }
    plane.material?.dispose();
    plane.geometry?.dispose();
}

function clearPlanes() {
    for (const plane of planes) {
        disposePlane(plane);
    }
    planes.length = 0;
    planesBySource.clear();
    hoveredPlane = null;
    selectedEntries.clear();
    selectedNodes.clear();
    entriesList.innerHTML = "";
    hidePreview();
    setTotals();
}

function updatePlaneStyle(plane) {
    const selected = plane.userData.selected;
    const hovered = plane === hoveredPlane;
    plane.material.color.set(selected ? "#bc6c25" : hovered ? "#255f56" : "#ffffff");
    plane.material.opacity = hovered ? HOVER_OPACITY : POINT_OPACITY;
    plane.userData.scaleBoost = selected ? 1.24 : hovered ? 1.16 : 1.0;
}

function updateAllPlaneStyles() {
    for (const plane of planes) {
        updatePlaneStyle(plane);
    }
}

function setEmptyState(
    visible,
    message = "Start or refresh an experiment to populate this map.",
    title = "No discoveries available",
) {
    emptyState.hidden = !visible;
    const heading = emptyState.querySelector("h2");
    const text = emptyState.querySelector("p");
    if (heading) {
        heading.textContent = title;
    }
    if (text) {
        text.textContent = message;
    }
}

function setCoverageEmpty(visible, title = "No coverage run found", message = "") {
    coverageEmpty.hidden = !visible;
    const heading = coverageEmpty.querySelector("h3");
    const text = coverageEmpty.querySelector("p");
    if (heading) {
        heading.textContent = title;
    }
    if (text) {
        text.textContent = message;
    }
}

async function initializeCoverageNavigation() {
    try {
        const response = await fetch("/coverage_status", { cache: "no-store" });
        if (!response.ok) {
            throw new Error("coverage status unavailable");
        }

        const status = await response.json();
        coverageEnabled = Boolean(status.enabled);
    } catch {
        coverageEnabled = false;
    }

    coverageTab.hidden = !coverageEnabled;
    if (!coverageEnabled && coveragePage.classList.contains("active")) {
        showPage("discoveries");
    }
}

function showPage(pageName) {
    const isCoverage = pageName === "coverage";
    if (isCoverage && !coverageEnabled) {
        return;
    }

    viewerPage.classList.toggle("active", !isCoverage);
    coveragePage.classList.toggle("active", isCoverage);
    discoveriesTab.classList.toggle("active", !isCoverage);
    coverageTab.classList.toggle("active", isCoverage);

    if (isCoverage) {
        hidePreview();
        loadCoverageSummary();
    } else {
        resizeRenderer();
    }
}

function fitView() {
    const visiblePlanes = planes.filter((plane) => plane.visible);
    if (visiblePlanes.length === 0) {
        camera.position.set(0, 0, 26);
        controls.target.set(0, 0, 0);
        controls.update();
        return;
    }

    const box = new THREE.Box3();
    for (const plane of visiblePlanes) {
        box.expandByPoint(plane.position);
    }

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxSpan = Math.max(size.x, size.y, 1);
    const fov = camera.fov * (Math.PI / 180);
    const distance = clamp((maxSpan / 2) / Math.tan(fov / 2) + 5, cameraDepthBounds.min, cameraDepthBounds.max);

    controls.target.set(center.x, center.y, 0);
    camera.position.set(center.x, center.y, distance);
    controls.update();
}

function addEntryToList(src) {
    if (selectedNodes.has(src)) {
        return;
    }

    const li = document.createElement("li");
    li.dataset.src = src;

    const entryMain = document.createElement("div");
    entryMain.className = "entryMain";

    const preview = document.createElement("img");
    preview.className = "entryPreview";
    preview.alt = "Selected discovery preview";
    preview.loading = "lazy";
    preview.decoding = "async";
    const previewSrc = selectedEntryPreviewPath(src);
    preview.src = previewSrc;
    const fallbackSrc = selectedEntryPreviewFallback(src);
    if (fallbackSrc !== previewSrc) {
        preview.addEventListener("error", () => {
            preview.src = fallbackSrc;
        }, { once: true });
    }

    preview.addEventListener("mouseenter", () => showPreviewForSource(src, null, preview));
    preview.addEventListener("focus", () => showPreviewForSource(src, null, preview));
    preview.addEventListener("mouseleave", hidePreview);
    preview.addEventListener("blur", hidePreview);

    const span = document.createElement("span");
    span.className = "entryLabel";
    span.textContent = prettifyEntryLabel(src);
    span.title = span.textContent;

    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.textContent = "Remove";
    removeButton.addEventListener("click", () => unselectEntry(src));

    entryMain.appendChild(preview);
    entryMain.appendChild(span);
    li.appendChild(entryMain);
    li.appendChild(removeButton);
    entriesList.appendChild(li);
    selectedNodes.set(src, li);
}

function selectedEntryPreviewPath(sourcePath) {
    if (/\.mp4$/i.test(sourcePath)) {
        return sourcePath.replace(/\.mp4$/i, ".jpg");
    }
    return sourcePath;
}

function selectedEntryPreviewFallback(sourcePath) {
    if (/\.mp4$/i.test(sourcePath)) {
        return sourcePath.replace(/\.mp4$/i, ".png");
    }
    return sourcePath;
}

function selectEntry(src) {
    if (selectedEntries.has(src)) {
        return;
    }
    selectedEntries.add(src);
    addEntryToList(src);
    setTotals();
}

function unselectEntry(src) {
    if (!selectedEntries.has(src)) {
        return;
    }
    selectedEntries.delete(src);
    const node = selectedNodes.get(src);
    if (node) {
        node.remove();
        selectedNodes.delete(src);
    }

    for (const plane of planes) {
        if (plane.userData.sourcePath === src) {
            plane.userData.selected = false;
            updatePlaneStyle(plane);
        }
    }
    setTotals();
}

function toggleEntryForPlane(plane) {
    const src = plane.userData.sourcePath;
    plane.userData.selected = !plane.userData.selected;
    if (plane.userData.selected) {
        selectEntry(src);
    } else {
        unselectEntry(src);
    }
    updatePlaneStyle(plane);
}

function clearSelection() {
    selectedEntries.clear();
    selectedNodes.clear();
    entriesList.innerHTML = "";
    for (const plane of planes) {
        plane.userData.selected = false;
        updatePlaneStyle(plane);
    }
    setTotals();
    updateStatus("Selection cleared.");
}

function hidePreview() {
    previewCard.style.display = "none";
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.style.display = "none";
    previewImage.removeAttribute("src");
    previewImage.style.display = "none";
}

function openGraphLightbox(src, title) {
    graphLightboxTitle.textContent = title;
    graphLightboxImage.src = src;
    graphLightboxImage.alt = `Expanded coverage graph for ${title}`;
    graphLightbox.hidden = false;
}

function closeGraphLightbox() {
    graphLightbox.hidden = true;
    graphLightboxImage.removeAttribute("src");
}

function placePreviewAt(x, y) {
    const margin = 14;
    const rect = previewCard.getBoundingClientRect();
    const width = Math.min(rect.width || 430, window.innerWidth - margin * 2);
    const height = Math.min(rect.height || 310, window.innerHeight - margin * 2);
    const maxLeft = Math.max(margin, window.innerWidth - width - margin);
    const maxTop = Math.max(margin, window.innerHeight - height - margin);
    const left = clamp(x + 18, margin, maxLeft);
    const top = clamp(y + 14, margin, maxTop);
    previewCard.style.left = `${left}px`;
    previewCard.style.top = `${top}px`;
}

function placePreviewNearElement(element) {
    const rect = element.getBoundingClientRect();
    placePreviewAt(rect.right, rect.top);
}

function schedulePreviewPlacement(pointerEvent = null, anchorElement = null) {
    requestAnimationFrame(() => {
        if (previewCard.style.display === "none") {
            return;
        }

        if (pointerEvent) {
            placePreviewAt(pointerEvent.clientX, pointerEvent.clientY);
        } else if (anchorElement) {
            placePreviewNearElement(anchorElement);
        }
    });
}

function showPreviewForSource(src, pointerEvent = null, anchorElement = null) {
    previewMeta.textContent = prettifyEntryLabel(src);
    previewCard.style.display = "block";

    previewVideo.pause();
    previewVideo.style.display = "none";
    previewImage.style.display = "none";

    if (isVideoPath(src)) {
        previewVideo.src = src;
        previewVideo.style.display = "block";
        previewVideo.currentTime = 0;
        previewVideo.play().catch(() => {});
    } else {
        previewImage.src = src;
        previewImage.style.display = "block";
    }

    schedulePreviewPlacement(pointerEvent, anchorElement);
}

function showPreviewForPlane(plane, event) {
    showPreviewForSource(plane.userData.sourcePath, event, null);
}

function setPointerFromEvent(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
}

function pickPlaneAtPointer(event) {
    setPointerFromEvent(event);
    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObjects(planes.filter((plane) => plane.visible));
    return intersects.length > 0 ? intersects[0].object : null;
}

function updateHoverState(event) {
    const nextHovered = pickPlaneAtPointer(event);
    if (nextHovered !== hoveredPlane) {
        const previous = hoveredPlane;
        hoveredPlane = nextHovered;
        if (previous) {
            updatePlaneStyle(previous);
        }
        if (hoveredPlane) {
            updatePlaneStyle(hoveredPlane);
        }
    }

    if (hoveredPlane) {
        renderer.domElement.style.cursor = "pointer";
        showPreviewForPlane(hoveredPlane, event);
    } else {
        renderer.domElement.style.cursor = "grab";
        hidePreview();
    }
}

function updatePlaneFromPoint(plane, point) {
    const visual = normalizeVisualPath(point.visual);
    const sourcePath = mediaUrl(visual);
    plane.position.set(SCALE_FACTOR * Number(point.x || 0), SCALE_FACTOR * Number(point.y || 0), 0);
    plane.userData.sourcePath = sourcePath;
    plane.userData.label = prettifyEntryLabel(sourcePath).toLowerCase();
    plane.userData.selected = selectedEntries.has(sourcePath);
    updatePlaneStyle(plane);
}

async function loadDiscoveryPoint(point) {
    const visual = normalizeVisualPath(point.visual);
    const previewImagePath = visualToPreviewImage(visual);
    const previewFallbackPath = fallbackPreviewImage(visual);

    let texture;
    try {
        texture = await textureLoader.loadAsync(mediaUrl(previewImagePath));
    } catch {
        if (previewFallbackPath === previewImagePath) {
            return null;
        }
        try {
            texture = await textureLoader.loadAsync(mediaUrl(previewFallbackPath));
        } catch {
            return null;
        }
    }

    const width = texture.image.width || 256;
    const height = texture.image.height || 256;
    const ratio = width / Math.max(1, height);
    const baseHeight = 0.42;
    const baseWidth = baseHeight * ratio;

    const material = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: POINT_OPACITY,
    });

    const plane = new THREE.Mesh(new THREE.PlaneGeometry(baseWidth, baseHeight), material);
    plane.userData.scaleBoost = 1.0;
    updatePlaneFromPoint(plane, point);
    return plane;
}

async function readDiscoveries(maxWaitMs = 0) {
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

        updateStatus("Waiting for discovery data...");
        await sleep(LOAD_RETRY_MS);
    }
}

function removePlaneBySource(sourcePath) {
    const plane = planesBySource.get(sourcePath);
    if (!plane) {
        return;
    }

    const idx = planes.indexOf(plane);
    if (idx >= 0) {
        planes.splice(idx, 1);
    }
    if (hoveredPlane === plane) {
        hoveredPlane = null;
        hidePreview();
    }
    planesBySource.delete(sourcePath);
    disposePlane(plane);
}

async function syncPoints(pointsData, shouldFitInitialView = false) {
    const existingCount = planes.length;
    const expectedSources = new Set();
    const newPoints = [];
    let addedCount = 0;

    for (const point of pointsData) {
        const sourcePath = mediaUrl(point.visual);
        expectedSources.add(sourcePath);

        const existingPlane = planesBySource.get(sourcePath);
        if (existingPlane) {
            updatePlaneFromPoint(existingPlane, point);
            continue;
        }

        newPoints.push(point);
    }

    const loadedPlanes = await Promise.all(newPoints.map(loadDiscoveryPoint));
    for (const plane of loadedPlanes) {
        if (!plane) {
            continue;
        }

        planes.push(plane);
        planesBySource.set(plane.userData.sourcePath, plane);
        scene.add(plane);
        addedCount += 1;
    }

    for (const sourcePath of Array.from(planesBySource.keys())) {
        if (!expectedSources.has(sourcePath)) {
            unselectEntry(sourcePath);
            removePlaneBySource(sourcePath);
        }
    }

    applyFilter();
    setEmptyState(
        planes.length === 0,
        "No usable preview images were found for these discoveries.",
        "No previews available",
    );

    if (shouldFitInitialView || existingCount === 0) {
        fitView();
    }

    return addedCount;
}

async function loadPoints(maxWaitMs = 0, shouldFitInitialView = false) {
    const pointsData = await readDiscoveries(maxWaitMs);

    if (pointsData.length === 0) {
        setEmptyState(
            true,
            "Start or refresh an experiment to populate this map.",
            "No discoveries available",
        );
        updateStatus("No discoveries found yet.");
        return;
    }

    const previousCount = planes.length;
    const addedCount = await syncPoints(pointsData, shouldFitInitialView);
    const shown = planes.filter((plane) => plane.visible).length;
    const suffix = addedCount > 0 && previousCount > 0 ? `, ${addedCount} new` : "";
    updateStatus(`${shown}/${pointsData.length} discoveries shown${suffix}.`);
}

async function refreshDiscoveries(shouldFitInitialView = false) {
    if (isRefreshing) {
        pendingRefresh = true;
        return;
    }

    isRefreshing = true;
    refreshButton.disabled = true;
    try {
        updateStatus("Updating discoveries...");
        setEmptyState(false);
        await loadPoints(DISCOVERY_LOAD_GRACE_MS, shouldFitInitialView);
    } catch {
        setEmptyState(
            true,
            "Discovery coordinates are not ready yet. Try refreshing after recompute.",
            "Discoveries unavailable",
        );
        updateStatus("Failed to load discoveries.");
    } finally {
        isRefreshing = false;
        refreshButton.disabled = false;
        if (pendingRefresh) {
            pendingRefresh = false;
            refreshDiscoveries(shouldFitInitialView);
        }
    }
}

function scheduleLiveRefresh() {
    if (liveRefreshTimerId !== null) {
        return;
    }

    const elapsed = performance.now() - lastLiveRefreshTimestamp;
    const delay = Math.max(0, LIVE_REFRESH_COOLDOWN_MS - elapsed);

    liveRefreshTimerId = window.setTimeout(async () => {
        liveRefreshTimerId = null;
        lastLiveRefreshTimestamp = performance.now();
        await refreshDiscoveries(false);
    }, delay);
}

async function recomputeLayout() {
    recomputeLayoutButton.disabled = true;
    refreshButton.disabled = true;
    try {
        updateStatus("Recomputing clustered layout...");
        const response = await fetch("/recompute_layout", { method: "POST" });
        if (!response.ok) {
            throw new Error("layout recompute failed");
        }
        await refreshDiscoveries(true);
        updateStatus("Clustered layout recomputed.");
    } catch {
        updateStatus("Failed to recompute clustered layout.");
    } finally {
        recomputeLayoutButton.disabled = false;
        refreshButton.disabled = false;
    }
}

function applyFilter() {
    const search = buildDiscoveryMatcher(searchInput.value);
    if (search.error) {
        updateStatus(`Invalid search pattern: ${search.error}`);
        return;
    }

    let firstVisible = null;
    for (const plane of planes) {
        const visible = search.matcher(plane.userData.label);
        plane.visible = visible;
        if (visible && !firstVisible) {
            firstVisible = plane;
        }
    }

    if (hoveredPlane && !hoveredPlane.visible) {
        hoveredPlane = null;
        hidePreview();
    }

    setTotals();
    updateAllPlaneStyles();
    if (planes.length > 0 && !firstVisible) {
        updateStatus("No discoveries match the filter.");
    } else if (planes.length > 0) {
        const shown = planes.filter((plane) => plane.visible).length;
        const modeLabel = search.mode === "empty" ? "" : ` (${search.mode})`;
        updateStatus(`${shown}/${planes.length} discoveries shown${modeLabel}.`);
    }
}

async function exportEntries() {
    if (selectedEntries.size === 0) {
        updateStatus("No entries selected for export.");
        return;
    }

    exportButton.disabled = true;
    updateStatus("Exporting selected entries...");
    try {
        const response = await fetch("/export", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(Array.from(selectedEntries)),
        });
        if (!response.ok) {
            throw new Error("export failed");
        }
        const payload = await response.json();
        updateStatus(`Export complete: ${payload.new_dir}`);
    } catch {
        updateStatus("Export failed. Check server logs.");
    } finally {
        exportButton.disabled = false;
    }
}

function connectWebsocket() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

    ws.onmessage = async (event) => {
        if (event.data === "refresh") {
            scheduleLiveRefresh();
        }
    };

    ws.onclose = () => {
        setTimeout(connectWebsocket, 1000);
    };
}

function formatNumber(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return "0";
    }
    return new Intl.NumberFormat().format(value);
}

function formatRange(bounds) {
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

function buildDiscoveryMatcher(rawQuery) {
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

function statBox(value, label) {
    const node = document.createElement("div");
    node.className = "statBox";
    const valueNode = document.createElement("span");
    valueNode.className = "statValue";
    valueNode.textContent = value;
    const labelNode = document.createElement("span");
    labelNode.className = "statLabel";
    labelNode.textContent = label;
    node.appendChild(valueNode);
    node.appendChild(labelNode);
    return node;
}

function renderCoverage(summary) {
    coverageGrid.innerHTML = "";
    coverageStats.innerHTML = "";
    setCoverageEmpty(false);

    coverageSubtitle.textContent = summary.run_name
        ? `Showing ${summary.run_name}`
        : "Showing latest coverage run";

    coverageStats.appendChild(statBox(formatNumber(summary.random_count), "random samples"));
    coverageStats.appendChild(statBox(formatNumber(summary.tool_count), "tool discoveries"));
    coverageStats.appendChild(statBox(formatNumber(summary.dim_count), "embedding dimensions"));
    coverageStats.appendChild(statBox(formatNumber((summary.images || []).length), "graphs"));

    const images = Array.isArray(summary.images) ? summary.images : [];
    const labels = Array.isArray(summary.labels) ? summary.labels : [];
    const bounds = Array.isArray(summary.bounds) ? summary.bounds : [];

    for (const [index, image] of images.entries()) {
        const imageInfo = typeof image === "string" ? { file: image, url: `/coverage/${image}` } : image;
        const title = labels[index] || imageInfo.title || imageInfo.file || `Graph ${index + 1}`;
        const card = document.createElement("article");
        card.className = "coverageCard";

        const header = document.createElement("div");
        header.className = "coverageCardHeader";

        const titleNode = document.createElement("div");
        titleNode.className = "coverageCardTitle";
        titleNode.textContent = title;
        titleNode.title = title;

        const metaNode = document.createElement("div");
        metaNode.className = "coverageCardMeta";
        metaNode.textContent = formatRange(bounds[index]);

        const img = document.createElement("img");
        img.loading = "lazy";
        img.decoding = "async";
        img.alt = `Coverage graph for ${title}`;
        img.src = imageInfo.url || `/coverage/${imageInfo.file}`;

        const imageButton = document.createElement("button");
        imageButton.className = "coverageImageButton";
        imageButton.type = "button";
        imageButton.setAttribute("aria-label", `Open ${title} larger`);
        imageButton.addEventListener("click", () => {
            openGraphLightbox(img.src, title);
        });

        header.appendChild(titleNode);
        header.appendChild(metaNode);
        card.appendChild(header);
        imageButton.appendChild(img);
        card.appendChild(imageButton);
        coverageGrid.appendChild(card);
    }

    if (images.length === 0) {
        setCoverageEmpty(
            true,
            "No coverage graphs found",
            "The coverage summary exists, but it does not list generated images.",
        );
    }
}

async function responseErrorMessage(response) {
    try {
        const payload = await response.json();
        if (typeof payload.detail === "string") {
            return payload.detail;
        }
    } catch {
        // Fall through to the generic message below.
    }
    return "Coverage summary could not be loaded.";
}

async function loadCoverageSummary() {
    const requestId = ++coverageRequestId;
    const deadline = performance.now() + COVERAGE_LOAD_GRACE_MS;
    let lastErrorMessage = "Coverage summary could not be loaded.";
    let fatalErrorMessage = "";
    reloadCoverageButton.disabled = true;
    coverageSubtitle.textContent = "Loading coverage summary...";
    coverageGrid.innerHTML = "";
    coverageStats.innerHTML = "";
    setCoverageEmpty(false);
    try {
        let summary = null;
        while (summary === null) {
            try {
                const response = await fetch("/coverage_summary", { cache: "no-store" });
                if (response.ok) {
                    summary = await response.json();
                    break;
                }

                lastErrorMessage = await responseErrorMessage(response);
                if (response.status === 422 || response.status === 400) {
                    fatalErrorMessage = lastErrorMessage;
                    throw new Error(fatalErrorMessage);
                }
            } catch (error) {
                if (fatalErrorMessage) {
                    throw error;
                }
                // Retry transient startup/load failures until the grace window expires.
            }

            if (performance.now() >= deadline) {
                throw new Error("coverage summary unavailable");
            }
            if (requestId !== coverageRequestId) {
                return;
            }
            await sleep(LOAD_RETRY_MS);
        }
        if (requestId !== coverageRequestId) {
            return;
        }
        renderCoverage(summary);
    } catch {
        if (requestId !== coverageRequestId) {
            return;
        }
        coverageGrid.innerHTML = "";
        coverageStats.innerHTML = "";
        coverageSubtitle.textContent = "Coverage could not be loaded.";
        setCoverageEmpty(
            true,
            "Coverage format issue",
            lastErrorMessage,
        );
    } finally {
        if (requestId === coverageRequestId) {
            reloadCoverageButton.disabled = false;
        }
    }
}

function resizeRenderer() {
    const width = Math.max(1, app.clientWidth);
    const height = Math.max(1, app.clientHeight);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
}

function updateViewAnimation() {
    requestAnimationFrame(updateViewAnimation);
    controls.update();

    for (const plane of planes) {
        const distance = Math.max(0.01, camera.position.z - plane.position.z);
        const scale = distance * 0.19 * (plane.userData.scaleBoost || 1.0);
        plane.scale.set(scale, scale, 1);
    }

    renderer.render(scene, camera);
}

renderer.domElement.addEventListener("pointerdown", (event) => {
    pointerDown = { x: event.clientX, y: event.clientY };
});

renderer.domElement.addEventListener("pointermove", updateHoverState);

renderer.domElement.addEventListener("pointerleave", () => {
    const previous = hoveredPlane;
    hoveredPlane = null;
    if (previous) {
        updatePlaneStyle(previous);
    }
    hidePreview();
});

renderer.domElement.addEventListener("click", (event) => {
    if (pointerDown) {
        const moved = Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y);
        if (moved > 5) {
            return;
        }
    }

    const plane = pickPlaneAtPointer(event);
    if (plane) {
        toggleEntryForPlane(plane);
        showPreviewForPlane(plane, event);
    }
});

window.addEventListener("resize", resizeRenderer);
window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        closeGraphLightbox();
        hidePreview();
        searchInput.blur();
    }
});

discoveriesTab.addEventListener("click", () => showPage("discoveries"));
coverageTab.addEventListener("click", () => showPage("coverage"));
reloadCoverageButton.addEventListener("click", loadCoverageSummary);
fitViewButton.addEventListener("click", fitView);
refreshButton.addEventListener("click", () => refreshDiscoveries(false));
recomputeLayoutButton.addEventListener("click", recomputeLayout);
clearSelectionButton.addEventListener("click", clearSelection);
exportButton.addEventListener("click", exportEntries);
searchInput.addEventListener("input", applyFilter);
previewSizeSlider.addEventListener("input", (event) => {
    applyPreviewScale(event.target.value);
});
graphLightboxClose.addEventListener("click", closeGraphLightbox);
graphLightbox.addEventListener("click", (event) => {
    if (event.target === graphLightbox) {
        closeGraphLightbox();
    }
});

new ResizeObserver(resizeRenderer).observe(app);

applyPreviewScale(previewSizeSlider.value);
initializeCoverageNavigation();
resizeRenderer();
refreshDiscoveries(true).then(() => {
    lastLiveRefreshTimestamp = performance.now();
});
connectWebsocket();
updateViewAnimation();
