import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const SCALE_FACTOR = 30;
const DEFAULT_POINT_OPACITY = 0.88;
const FOCUSED_OPACITY = 1.0;
const cameraDepthBounds = { min: 2.0, max: 150.0 };

const app = document.getElementById("app");
const statusLine = document.getElementById("statusLine");
const selectedCount = document.getElementById("selectedCount");
const entriesList = document.getElementById("entriesList");

const previewCard = document.getElementById("previewCard");
const previewVideo = document.getElementById("hoverVideo");
const previewImage = document.getElementById("hoverImage");
const previewMeta = document.getElementById("previewMeta");
const previewSizeSlider = document.getElementById("previewSizeSlider");
const previewSizeValue = document.getElementById("previewSizeValue");

const prevFocusButton = document.getElementById("prevFocusButton");
const nextFocusButton = document.getElementById("nextFocusButton");
const selectFocusedButton = document.getElementById("selectFocusedButton");
const scanToggleButton = document.getElementById("scanToggleButton");
const targetFocusButton = document.getElementById("targetFocusButton");
const targetPlaceModeButton = document.getElementById("targetPlaceModeButton");
const removeTargetButton = document.getElementById("removeTargetButton");
const mouthModeButton = document.getElementById("mouthModeButton");
const clearPreviewButton = document.getElementById("clearPreviewButton");
const clearSelectionButton = document.getElementById("clearSelectionButton");
const exportButton = document.getElementById("exportButton");

const scene = new THREE.Scene();
scene.background = new THREE.Color("#f2f7fc");

const camera = new THREE.PerspectiveCamera(
    65,
    window.innerWidth / window.innerHeight,
    0.01,
    800,
);
camera.position.set(0, 0, 28);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
app.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.rotateSpeed = 0.0;
controls.enableRotate = false;
controls.zoomSpeed = 1.0;
controls.panSpeed = 0.85;
controls.screenSpacePanning = true;
controls.maxDistance = cameraDepthBounds.max;
controls.minDistance = cameraDepthBounds.min;
controls.maxPolarAngle = Math.PI / 2;
controls.minPolarAngle = Math.PI / 2;

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

const planes = [];
const selectedEntries = new Set();
const selectedNodes = new Map();

const PREVIEW_BASE_WIDTH = 512;
const PREVIEW_BASE_HEIGHT = 316;
let previewScalePercent = 100;

let focusedIndex = -1;
let hoveredPlane = null;
let targetSprite = null;
let targetVisible = false;
let targetPlaceMode = false;
let scanMode = false;
let mouthMode = false;
let scanIntervalId = null;
let lastHoverPath = "";
let lastHoverTimestamp = 0;
let previewPinnedFromList = false;
let isRefreshing = false;
let pendingRefresh = false;

function updateStatus(text) {
    statusLine.textContent = text;
}

function updateSelectedCount() {
    selectedCount.textContent = `${selectedEntries.size} selected`;
}

function isUiEventTarget(target) {
    return Boolean(target.closest(".hud") || target.closest("#previewCard"));
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function getPreviewDimensions() {
    const scale = previewScalePercent / 100;
    return {
        width: Math.round(PREVIEW_BASE_WIDTH * scale),
        height: Math.round(PREVIEW_BASE_HEIGHT * scale),
    };
}

function applyPreviewScale(percent) {
    previewScalePercent = clamp(Number(percent) || 100, 50, 200);
    const dims = getPreviewDimensions();
    document.documentElement.style.setProperty("--preview-width", `${dims.width}px`);
    document.documentElement.style.setProperty("--preview-height", `${dims.height}px`);
    previewSizeSlider.value = `${previewScalePercent}`;
    previewSizeValue.textContent = `${previewScalePercent}%`;
}

function clearPlanes() {
    for (const plane of planes) {
        scene.remove(plane);
        if (plane.material?.map) {
            plane.material.map.dispose();
        }
        if (plane.material) {
            plane.material.dispose();
        }
        if (plane.geometry) {
            plane.geometry.dispose();
        }
    }
    planes.length = 0;
    hoveredPlane = null;
    focusedIndex = -1;
    selectedEntries.clear();
    selectedNodes.clear();
    entriesList.innerHTML = "";
    updateSelectedCount();
}

async function refreshDiscoveries() {
    if (isRefreshing) {
        pendingRefresh = true;
        return;
    }
    isRefreshing = true;

    try {
        updateStatus("Updating discoveries...");
        clearPlanes();
        hidePreview();
        await loadPoints();
        await loadTargetSprite();
        updateStatus(`${planes.length} discoveries loaded.`);
    } catch {
        updateStatus("Refresh failed. Try again.");
    } finally {
        isRefreshing = false;
        if (pendingRefresh) {
            pendingRefresh = false;
            refreshDiscoveries();
        }
    }
}

function prettifyEntryLabel(src) {
    const normalized = src.replace(/^\/discoveries\//, "");
    const parts = normalized.split("/");
    if (parts.length <= 1) {
        return normalized;
    }
    return `${parts[0]} / ${parts[parts.length - 1]}`;
}

function visualToPreviewImage(visualPath) {
    if (visualPath.endsWith(".mp4")) {
        return visualPath.replace(/\.mp4$/i, ".jpg");
    }
    return visualPath;
}

function fallbackPreviewImage(visualPath) {
    if (visualPath.endsWith(".mp4")) {
        return visualPath.replace(/\.mp4$/i, ".png");
    }
    return visualPath;
}

function isVideoPath(path) {
    return /\.mp4$/i.test(path);
}

function selectedEntryPreviewPath(sourcePath) {
    if (sourcePath.endsWith(".mp4")) {
        return sourcePath.replace(/\.mp4$/i, ".jpg");
    }
    return sourcePath;
}

function selectedEntryPreviewFallback(sourcePath) {
    if (sourcePath.endsWith(".mp4")) {
        return sourcePath.replace(/\.mp4$/i, ".png");
    }
    return sourcePath;
}

function updatePlaneStyle(plane) {
    const isSelected = plane.userData.selected;
    const isFocused = plane.userData.focused;

    if (isSelected) {
        plane.material.color.set("#f59e0b");
    } else if (isFocused) {
        plane.material.color.set("#2563eb");
    } else {
        plane.material.color.set("#ffffff");
    }

    plane.material.opacity = isFocused ? FOCUSED_OPACITY : DEFAULT_POINT_OPACITY;
    plane.userData.scaleBoost = isFocused ? 1.18 : 1.0;
}

function setFocusedIndex(index, shouldOpenPreview = true) {
    if (planes.length === 0) {
        focusedIndex = -1;
        return;
    }

    const wrapped = ((index % planes.length) + planes.length) % planes.length;

    if (focusedIndex >= 0 && focusedIndex < planes.length) {
        const oldFocused = planes[focusedIndex];
        oldFocused.userData.focused = false;
        updatePlaneStyle(oldFocused);
    }

    focusedIndex = wrapped;
    const focusedPlane = planes[focusedIndex];
    focusedPlane.userData.focused = true;
    updatePlaneStyle(focusedPlane);

    updateStatus(
        `Focused ${focusedIndex + 1}/${planes.length}: ${prettifyEntryLabel(focusedPlane.userData.sourcePath)}`,
    );

    if (shouldOpenPreview) {
        showPreviewForPlane(focusedPlane);
    }
}

function focusNext() {
    if (planes.length === 0) {
        return;
    }
    setFocusedIndex(focusedIndex + 1);
}

function focusPrevious() {
    if (planes.length === 0) {
        return;
    }
    setFocusedIndex(focusedIndex - 1);
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
    preview.alt = "Selected entry preview";
    preview.loading = "lazy";
    preview.decoding = "async";
    const previewSrc = selectedEntryPreviewPath(src);
    const fallbackSrc = selectedEntryPreviewFallback(src);
    preview.src = previewSrc;
    if (fallbackSrc !== previewSrc) {
        preview.addEventListener(
            "error",
            () => {
                if (preview.src.endsWith(".jpg")) {
                    preview.src = fallbackSrc;
                }
            },
            { once: true },
        );
    }

    const showPinnedPreview = () => {
        previewPinnedFromList = true;
        showPreviewForSource(src, null, preview);
    };

    const unpinPreview = () => {
        previewPinnedFromList = false;
        if (!hoveredPlane && !mouthMode) {
            hidePreview();
        }
    };

    preview.addEventListener("mouseenter", showPinnedPreview);
    preview.addEventListener("focus", showPinnedPreview);
    preview.addEventListener("mousemove", showPinnedPreview);
    preview.addEventListener("mouseleave", unpinPreview);
    preview.addEventListener("blur", unpinPreview);

    const span = document.createElement("span");
    span.className = "entryLabel";
    span.textContent = prettifyEntryLabel(src);

    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.textContent = "Remove";
    removeButton.title = "Remove this discovery from the selected entries list";
    removeButton.addEventListener("click", () => {
        unselectEntry(src);
    });

    entryMain.appendChild(preview);
    entryMain.appendChild(span);
    li.appendChild(entryMain);
    li.appendChild(removeButton);
    entriesList.appendChild(li);
    selectedNodes.set(src, li);
}

function removeEntryFromList(src) {
    const node = selectedNodes.get(src);
    if (!node) {
        return;
    }
    node.remove();
    selectedNodes.delete(src);
}

function selectEntry(src) {
    if (selectedEntries.has(src)) {
        return;
    }
    selectedEntries.add(src);
    addEntryToList(src);
    updateSelectedCount();
}

function unselectEntry(src) {
    if (!selectedEntries.has(src)) {
        return;
    }
    selectedEntries.delete(src);
    removeEntryFromList(src);

    for (const plane of planes) {
        if (plane.userData.sourcePath === src) {
            plane.userData.selected = false;
            updatePlaneStyle(plane);
        }
    }

    updateSelectedCount();
}

function toggleEntryForPlane(plane) {
    const src = plane.userData.sourcePath;
    if (selectedEntries.has(src)) {
        unselectEntry(src);
        plane.userData.selected = false;
        updatePlaneStyle(plane);
        return;
    }

    selectEntry(src);
    plane.userData.selected = true;
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

    updateSelectedCount();
    updateStatus("Selection cleared.");
}

function hidePreview() {
    previewCard.style.display = "none";
    previewVideo.pause();
    previewVideo.style.display = "none";
    previewImage.style.display = "none";
}

function placePreviewNearElement(element) {
    const rect = element.getBoundingClientRect();
    const x = rect.right + 10;
    const y = rect.top;
    placePreviewAt(x, y);
}

function placePreviewAt(x, y) {
    const margin = 14;
    const dims = getPreviewDimensions();
    const cardWidth = previewCard.offsetWidth || dims.width;
    const cardHeight = previewCard.offsetHeight || dims.height;

    const left = clamp(x + 18, margin, window.innerWidth - cardWidth - margin);
    const top = clamp(y + 14, margin, window.innerHeight - cardHeight - margin);

    previewCard.style.left = `${left}px`;
    previewCard.style.top = `${top}px`;
}

function showPreviewForSource(src, pointerEvent = null, anchorElement = null) {
    if (pointerEvent) {
        placePreviewAt(pointerEvent.clientX, pointerEvent.clientY);
    } else if (anchorElement) {
        placePreviewNearElement(anchorElement);
    } else {
        const dims = getPreviewDimensions();
        placePreviewAt(window.innerWidth - dims.width - 40, 90);
    }

    previewMeta.textContent = prettifyEntryLabel(src);
    previewCard.style.display = "block";

    previewVideo.pause();
    previewVideo.style.display = "none";
    previewImage.style.display = "none";

    if (isVideoPath(src)) {
        previewVideo.src = src;
        previewVideo.style.display = "block";
        previewVideo.currentTime = 0;
        previewVideo.play().catch(() => { });
    } else {
        previewImage.src = src;
        previewImage.style.display = "block";
    }
}

function showPreviewForPlane(plane, pointerEvent = null) {
    showPreviewForSource(plane.userData.sourcePath, pointerEvent, null);
}

function getMouseWorldPosition() {
    const point = new THREE.Vector3(mouse.x, mouse.y, 0.5);
    point.unproject(camera);
    const direction = point.sub(camera.position).normalize();
    const distance = -camera.position.z / direction.z;
    return camera.position.clone().add(direction.multiplyScalar(distance));
}

async function loadTargetSprite() {
    try {
        const textureLoader = new THREE.TextureLoader();
        const texture = await textureLoader.loadAsync("/static/target.png");
        const material = new THREE.SpriteMaterial({ map: texture });
        material.depthTest = false;
        material.depthWrite = false;
        targetSprite = new THREE.Sprite(material);
        targetSprite.renderOrder = 999;
        targetSprite.scale.set(0.7, 0.7, 1);
        targetSprite.visible = false;
        scene.add(targetSprite);

        const response = await fetch("/discoveries/target.json");
        if (!response.ok) {
            return;
        }
        const targetJson = await response.json();
        if (targetJson && typeof targetJson.x === "number" && typeof targetJson.y === "number") {
            targetSprite.position.set(targetJson.x * SCALE_FACTOR, targetJson.y * SCALE_FACTOR, 0);
            targetSprite.visible = true;
            targetVisible = true;
        }
    } catch {
        targetSprite = null;
    }
}

function setTargetAtWorldPosition(position) {
    if (!targetSprite) {
        return;
    }

    targetSprite.position.set(position.x, position.y, 0);
    targetSprite.visible = true;
    targetVisible = true;

    fetch("/update_target", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            x: position.x / SCALE_FACTOR,
            y: position.y / SCALE_FACTOR,
        }),
    }).catch(() => {
        updateStatus("Failed to update target on server.");
    });
}

function setTargetFromCanvasEvent(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    const nx = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera({ x: nx, y: ny }, camera);
    const placementPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const world = new THREE.Vector3();

    if (!raycaster.ray.intersectPlane(placementPlane, world)) {
        return false;
    }

    setTargetAtWorldPosition(world);
    return true;
}

function toggleTargetPlaceMode(force = null) {
    targetPlaceMode = force === null ? !targetPlaceMode : Boolean(force);
    targetPlaceModeButton.classList.toggle("active", targetPlaceMode);
    targetPlaceModeButton.textContent = `Place Target: ${targetPlaceMode ? "On" : "Off"}`;
    renderer.domElement.style.cursor = targetPlaceMode ? "crosshair" : "default";

    if (targetPlaceMode) {
        updateStatus("Place-target mode enabled: click anywhere on the map.");
    } else {
        updateStatus("Place-target mode disabled.");
    }
}

function clearTarget() {
    if (!targetSprite || !targetVisible) {
        return;
    }
    targetSprite.visible = false;
    targetVisible = false;
    fetch("/disable_target").catch(() => { });
}

function updateViewAnimation() {
    requestAnimationFrame(updateViewAnimation);
    controls.update();

    for (const plane of planes) {
        const distance = Math.max(0.01, camera.position.z - plane.position.z);
        const scale = distance * 0.21 * (plane.userData.scaleBoost || 1.0);
        plane.scale.set(scale, scale, 1);
    }

    if (targetSprite && targetVisible) {
        const targetDistance = Math.max(0.01, camera.position.z - targetSprite.position.z);
        const targetScale = targetDistance * 0.06;
        targetSprite.scale.set(targetScale, targetScale, 1);
    }

    renderer.render(scene, camera);
}

function pickPlaneAtPointer() {
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(planes);
    return intersects.length > 0 ? intersects[0].object : null;
}

function updateHoverState(event) {
    if (previewPinnedFromList && !event.target.closest("#entriesList")) {
        previewPinnedFromList = false;
    }

    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    const plane = pickPlaneAtPointer();
    hoveredPlane = plane;

    if (!plane) {
        lastHoverPath = "";
        if (!mouthMode && !previewPinnedFromList) {
            hidePreview();
        }
        return;
    }

    showPreviewForPlane(plane, event);

    if (!mouthMode) {
        return;
    }

    const hoveredPath = plane.userData.sourcePath;
    const now = performance.now();

    if (lastHoverPath !== hoveredPath) {
        lastHoverPath = hoveredPath;
        lastHoverTimestamp = now;
        return;
    }

    if (now - lastHoverTimestamp >= 900) {
        const idx = planes.indexOf(plane);
        if (idx >= 0) {
            setFocusedIndex(idx, false);
            lastHoverTimestamp = now + 10_000;
        }
    }
}

async function exportEntries() {
    if (selectedEntries.size === 0) {
        updateStatus("No entries selected for export.");
        return;
    }

    updateStatus("Exporting selected entries...");

    try {
        const response = await fetch("/export", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(Array.from(selectedEntries)),
        });

        if (!response.ok) {
            throw new Error("Export failed");
        }

        const payload = await response.json();
        updateStatus(`Export complete: ${payload.new_dir}`);
    } catch {
        updateStatus("Export failed. Check server logs.");
    }
}

function toggleScanMode(force = null) {
    scanMode = force === null ? !scanMode : Boolean(force);

    if (scanIntervalId) {
        clearInterval(scanIntervalId);
        scanIntervalId = null;
    }

    if (scanMode) {
        scanIntervalId = setInterval(() => {
            focusNext();
        }, 1200);
        scanToggleButton.classList.add("active");
        scanToggleButton.textContent = "Scan Mode: On";
        updateStatus("Scan mode enabled.");
        if (planes.length > 0 && focusedIndex < 0) {
            setFocusedIndex(0, false);
        }
    } else {
        scanToggleButton.classList.remove("active");
        scanToggleButton.textContent = "Scan Mode: Off";
        updateStatus("Scan mode disabled.");
    }
}

function toggleMouthMode(force = null) {
    mouthMode = force === null ? !mouthMode : Boolean(force);
    mouthModeButton.textContent = `Mouth Mode: ${mouthMode ? "On" : "Off"}`;
    mouthModeButton.classList.toggle("active", mouthMode);

    if (mouthMode) {
        updateStatus("Mouth mode enabled: hover dwell focuses points.");
    } else {
        updateStatus("Mouth mode disabled.");
    }
}

async function loadPoints() {
    try {
        const response = await fetch("/static/discoveries.json");
        if (!response.ok) {
            throw new Error("discoveries.json unavailable");
        }
        const pointsData = await response.json();
        if (!Array.isArray(pointsData)) {
            throw new Error("invalid discoveries format");
        }

        if (pointsData.length === 0) {
            updateStatus("No discoveries found yet.");
            return;
        }

        const textureLoader = new THREE.TextureLoader();

        for (const point of pointsData) {
            const visual = point.visual;
            const sourcePath = `/discoveries/${visual}`;
            const previewImage = visualToPreviewImage(visual);
            const previewFallback = fallbackPreviewImage(visual);

            let texture;
            try {
                texture = await textureLoader.loadAsync(`/discoveries/${previewImage}`);
            } catch {
                if (previewFallback === previewImage) {
                    continue;
                }
                try {
                    texture = await textureLoader.loadAsync(`/discoveries/${previewFallback}`);
                } catch {
                    continue;
                }
            }

            const width = texture.image.width || 256;
            const height = texture.image.height || 256;
            const ratio = width / Math.max(1, height);
            const baseHeight = 0.35;
            const baseWidth = baseHeight * ratio;

            const material = new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,
                opacity: DEFAULT_POINT_OPACITY,
            });

            const plane = new THREE.Mesh(new THREE.PlaneGeometry(baseWidth, baseHeight), material);

            plane.position.set(SCALE_FACTOR * point.x, SCALE_FACTOR * point.y, 0);
            plane.userData.sourcePath = sourcePath;
            plane.userData.selected = false;
            plane.userData.focused = false;
            plane.userData.scaleBoost = 1.0;

            updatePlaneStyle(plane);
            planes.push(plane);
            scene.add(plane);
        }

        if (planes.length === 0) {
            updateStatus("No preview images found for discoveries.");
            return;
        }

        setFocusedIndex(0, false);
        updateStatus(`${planes.length} discoveries loaded.`);
    } catch {
        updateStatus("Failed to load discoveries. Refresh after server recompute.");
    }
}

function connectWebsocket() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

    ws.onmessage = async (event) => {
        if (event.data === "refresh") {
            await refreshDiscoveries();
        }
    };

    ws.onclose = () => {
        setTimeout(connectWebsocket, 1000);
    };
}

function panWithKeyboard(key) {
    const panStep = Math.max(0.25, camera.position.z * 0.03);
    if (key === "ArrowUp") {
        camera.position.y += panStep;
        controls.target.y += panStep;
    }
    if (key === "ArrowDown") {
        camera.position.y -= panStep;
        controls.target.y -= panStep;
    }
    if (key === "ArrowLeft") {
        camera.position.x -= panStep;
        controls.target.x -= panStep;
    }
    if (key === "ArrowRight") {
        camera.position.x += panStep;
        controls.target.x += panStep;
    }
}

function zoomWithKeyboard(direction) {
    const factor = direction > 0 ? 0.92 : 1.08;
    camera.position.z = clamp(camera.position.z * factor, cameraDepthBounds.min, cameraDepthBounds.max);
}

window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

window.addEventListener("mousemove", (event) => {
    updateHoverState(event);
});

window.addEventListener("click", (event) => {
    if (isUiEventTarget(event.target)) {
        return;
    }

    if (targetPlaceMode) {
        if (setTargetFromCanvasEvent(event)) {
            toggleTargetPlaceMode(false);
            updateStatus("Target placed.");
        }
        return;
    }

    if (hoveredPlane) {
        const idx = planes.indexOf(hoveredPlane);
        if (idx >= 0) {
            setFocusedIndex(idx, false);
        }
        toggleEntryForPlane(hoveredPlane);
        return;
    }

    if (!mouthMode) {
        hidePreview();
    }
});

window.addEventListener("dblclick", (event) => {
    event.preventDefault();
    // Intentionally disabled to avoid one gesture triggering two actions.
});

window.addEventListener("contextmenu", (event) => {
    event.preventDefault();
});

window.addEventListener("keydown", (event) => {
    const key = event.key;

    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(key)) {
        event.preventDefault();
        panWithKeyboard(key);
        return;
    }

    if (key === "+" || key === "=") {
        event.preventDefault();
        zoomWithKeyboard(1);
        return;
    }

    if (key === "-" || key === "_") {
        event.preventDefault();
        zoomWithKeyboard(-1);
        return;
    }

    if (key === "n" || key === "N") {
        focusNext();
        return;
    }

    if (key === "p" || key === "P") {
        focusPrevious();
        return;
    }

    if (key === " " && focusedIndex >= 0) {
        event.preventDefault();
        toggleEntryForPlane(planes[focusedIndex]);
        return;
    }

    if (key === "Enter" && focusedIndex >= 0) {
        showPreviewForPlane(planes[focusedIndex]);
        return;
    }

    if (key === "t" || key === "T") {
        if (focusedIndex >= 0) {
            setTargetAtWorldPosition(planes[focusedIndex].position);
            updateStatus("Target aligned to focused discovery.");
        }
        return;
    }

    if (key === "a" || key === "A") {
        toggleTargetPlaceMode();
        return;
    }

    if (key === "d" || key === "D") {
        clearTarget();
        updateStatus("Target removed.");
        return;
    }

    if (key === "m" || key === "M") {
        toggleScanMode();
        return;
    }

    if (key === "e" || key === "E") {
        exportEntries();
        return;
    }

    if (key === "c" || key === "C") {
        clearSelection();
        return;
    }
});

prevFocusButton.addEventListener("click", focusPrevious);
nextFocusButton.addEventListener("click", focusNext);
selectFocusedButton.addEventListener("click", () => {
    if (focusedIndex < 0) {
        return;
    }
    toggleEntryForPlane(planes[focusedIndex]);
});
scanToggleButton.addEventListener("click", () => toggleScanMode());
targetPlaceModeButton.addEventListener("click", () => toggleTargetPlaceMode());
removeTargetButton.addEventListener("click", () => {
    clearTarget();
    updateStatus("Target removed.");
});
mouthModeButton.addEventListener("click", () => toggleMouthMode());
clearPreviewButton.addEventListener("click", hidePreview);

targetFocusButton.addEventListener("click", () => {
    if (focusedIndex < 0) {
        return;
    }
    setTargetAtWorldPosition(planes[focusedIndex].position);
    updateStatus("Target aligned to focused discovery.");
});

clearSelectionButton.addEventListener("click", clearSelection);
exportButton.addEventListener("click", exportEntries);
previewSizeSlider.addEventListener("input", (event) => {
    applyPreviewScale(event.target.value);
});

applyPreviewScale(previewSizeSlider.value);
loadTargetSprite();
loadPoints();
connectWebsocket();
updateViewAnimation();
