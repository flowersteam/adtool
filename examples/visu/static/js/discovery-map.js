import * as THREE from "three";

import {
    DISCOVERY_LOAD_GRACE_MS,
    HOVER_OPACITY,
    LIVE_REFRESH_COOLDOWN_MS,
    POINT_OPACITY,
    SCALE_FACTOR,
} from "./config.js";
import { readDiscoveries } from "./api.js";
import { createMapScene } from "./map-scene.js";
import { createSelectionController } from "./selection.js";
import {
    buildDiscoveryMatcher,
    fallbackPreviewImage,
    mediaUrl,
    normalizeVisualPath,
    prettifyEntryLabel,
    visualToPreviewImage,
} from "./utils.js";

export function createDiscoveryMap({ elements, preview, updateStatus }) {
    const mapScene = createMapScene(elements.app);
    const { renderer, scene, textureLoader } = mapScene;
    const planes = [];
    const planesBySource = new Map();

    let hoveredPlane = null;
    let isRefreshing = false;
    let pendingRefresh = false;
    let pointerDown = null;
    let liveRefreshTimerId = null;
    let lastLiveRefreshTimestamp = 0;

    function setTotals() {
        const visibleCount = planes.filter((plane) => plane.visible).length;
        elements.discoveryTotal.textContent = `${visibleCount}`;
        elements.selectionTotal.textContent = `${selection.size()}`;
    }

    function setEmptyState(
        visible,
        message = "Start or refresh an experiment to populate this map.",
        title = "No discoveries available",
    ) {
        elements.emptyState.hidden = !visible;
        const heading = elements.emptyState.querySelector("h2");
        const text = elements.emptyState.querySelector("p");
        if (heading) {
            heading.textContent = title;
        }
        if (text) {
            text.textContent = message;
        }
    }

    function visiblePlanes() {
        return planes.filter((plane) => plane.visible);
    }

    function disposePlane(plane) {
        scene.remove(plane);
        if (plane.material?.map) {
            plane.material.map.dispose();
        }
        plane.material?.dispose();
        plane.geometry?.dispose();
    }

    function updatePlaneStyle(plane) {
        const selected = plane.userData.selected;
        const hovered = plane === hoveredPlane;
        plane.material.color.set(selected ? "#bc6c25" : hovered ? "#255f56" : "#ffffff");
        plane.material.opacity = hovered ? HOVER_OPACITY : POINT_OPACITY;
        plane.userData.scaleBoost = selected ? 1.24 : hovered ? 1.16 : 1.0;
    }

    const selection = createSelectionController({
        entriesList: elements.entriesList,
        getPlanes: () => planes,
        preview,
        updatePlaneStyle,
        updateTotals: setTotals,
    });

    function updateAllPlaneStyles() {
        for (const plane of planes) {
            updatePlaneStyle(plane);
        }
    }

    function fitView() {
        mapScene.fitView(visiblePlanes());
    }

    function pickPlaneAtPointer(event) {
        return mapScene.pickPlaneAtPointer(event, visiblePlanes());
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
            preview.showForPlane(hoveredPlane, event);
        } else {
            renderer.domElement.style.cursor = "grab";
            preview.hide();
        }
    }

    function updatePlaneFromPoint(plane, point) {
        const visual = normalizeVisualPath(point.visual);
        const sourcePath = mediaUrl(visual);
        plane.position.set(
            SCALE_FACTOR * Number(point.x || 0),
            SCALE_FACTOR * Number(point.y || 0),
            0,
        );
        plane.userData.sourcePath = sourcePath;
        plane.userData.label = prettifyEntryLabel(sourcePath).toLowerCase();
        plane.userData.selected = selection.has(sourcePath);
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
            preview.hide();
        }
        planesBySource.delete(sourcePath);
        disposePlane(plane);
    }

    function applyFilter() {
        const search = buildDiscoveryMatcher(elements.searchInput.value);
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
            preview.hide();
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
                selection.unselect(sourcePath);
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
        const pointsData = await readDiscoveries(maxWaitMs, () => {
            updateStatus("Waiting for discovery data...");
        });

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
        elements.refreshButton.disabled = true;
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
            elements.refreshButton.disabled = false;
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

    function resizeRenderer() {
        mapScene.resizeRenderer();
    }

    function startAnimation() {
        mapScene.startAnimation(() => planes);
    }

    function clearSelection() {
        selection.clear();
        updateStatus("Selection cleared.");
    }

    function selectedEntries() {
        return selection.values();
    }

    function markLiveRefreshNow() {
        lastLiveRefreshTimestamp = performance.now();
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
        preview.hide();
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
            selection.togglePlane(plane);
            preview.showForPlane(plane, event);
        }
    });

    return {
        applyFilter,
        clearSelection,
        connectWebsocket,
        fitView,
        markLiveRefreshNow,
        refreshDiscoveries,
        resizeRenderer,
        selectedEntries,
        startAnimation,
    };
}
