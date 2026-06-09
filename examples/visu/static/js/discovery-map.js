import * as THREE from "three";

import {
    DISCOVERY_LOAD_GRACE_MS,
    HOVER_OPACITY,
    HYBRID_GRID_CELL_PX,
    HYBRID_THUMBNAIL_LIMIT,
    HYBRID_THUMBNAIL_WORLD_SIZE,
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

const RENDER_MODES = new Set(["points", "images", "hybrid"]);
const POINT_COLOR = "#2f3a35";
const HOVER_COLOR = "#255f56";
const SELECTED_COLOR = "#bc6c25";

export function createDiscoveryMap({ elements, preview, updateStatus }) {
    const mapScene = createMapScene(elements.app);
    const { renderer, scene, textureLoader } = mapScene;

    const pointGeometry = new THREE.CircleGeometry(0.12, 14);
    const pointMaterials = {
        normal: pointMaterial(POINT_COLOR),
        hover: pointMaterial(HOVER_COLOR),
        selected: pointMaterial(SELECTED_COLOR),
    };
    const atlasTextures = new Map();

    let entries = [];
    let renderMode = "points";
    let hoveredEntry = null;
    let isRefreshing = false;
    let pendingRefresh = false;
    let pointerDown = null;
    let liveRefreshTimerId = null;
    let lastLiveRefreshTimestamp = 0;
    let renderVersion = 0;
    let hybridRebuildId = null;

    function pointMaterial(color) {
        return new THREE.MeshBasicMaterial({
            color,
            opacity: POINT_OPACITY,
            transparent: true,
            depthWrite: false,
        });
    }

    function entrySource(point) {
        return mediaUrl(normalizeVisualPath(point.visual));
    }

    function createEntry(point) {
        const sourcePath = entrySource(point);
        return {
            imageMesh: null,
            pointMesh: null,
            thumbnailMesh: null,
            visible: true,
            position: new THREE.Vector3(
                SCALE_FACTOR * Number(point.x || 0),
                SCALE_FACTOR * Number(point.y || 0),
                0,
            ),
            userData: {
                label: prettifyEntryLabel(sourcePath).toLowerCase(),
                selected: selection.has(sourcePath),
                sourcePath,
                thumbnail: validThumbnail(point.thumbnail),
                visual: normalizeVisualPath(point.visual),
            },
        };
    }

    function validThumbnail(thumbnail) {
        if (
            !thumbnail
            || typeof thumbnail.atlas !== "string"
            || !Array.isArray(thumbnail.uv)
            || thumbnail.uv.length !== 4
        ) {
            return null;
        }
        const uv = thumbnail.uv.map(Number);
        return uv.every(Number.isFinite) ? { atlas: thumbnail.atlas, uv } : null;
    }

    function visibleEntries() {
        return entries.filter((entry) => {
            if (!entry.visible) {
                return false;
            }
            return renderMode !== "images" || Boolean(entry.imageMesh);
        });
    }

    function currentPickMeshes() {
        if (renderMode === "images") {
            return entries.map((entry) => entry.imageMesh).filter(Boolean);
        }
        if (renderMode === "hybrid") {
            return [
                ...entries.map((entry) => entry.thumbnailMesh).filter(Boolean),
                ...entries.map((entry) => entry.pointMesh).filter(Boolean),
            ];
        }
        return entries.map((entry) => entry.pointMesh).filter(Boolean);
    }

    function currentAnimatedMeshes() {
        return entries.flatMap((entry) => [
            entry.pointMesh,
            entry.imageMesh,
            entry.thumbnailMesh,
        ].filter(Boolean));
    }

    function setTotals() {
        elements.discoveryTotal.textContent = `${visibleEntries().length}`;
        elements.selectionTotal.textContent = `${selection.size()}`;
    }

    function setEmptyState(
        visible,
        message = "Start or refresh an experiment to populate this map.",
        title = "No discoveries available",
    ) {
        elements.emptyState.hidden = !visible;
        elements.emptyState.querySelector("h2").textContent = title;
        elements.emptyState.querySelector("p").textContent = message;
    }

    function disposeMesh(mesh, disposeTexture = false) {
        if (!mesh) {
            return;
        }
        scene.remove(mesh);
        if (disposeTexture && mesh.material?.map) {
            mesh.material.map.dispose();
        }
        mesh.material?.dispose();
        mesh.geometry?.dispose();
    }

    function clearPointMeshes() {
        for (const entry of entries) {
            if (entry.pointMesh) {
                scene.remove(entry.pointMesh);
            }
            entry.pointMesh = null;
        }
    }

    function clearImageMeshes() {
        for (const entry of entries) {
            disposeMesh(entry.imageMesh, true);
            entry.imageMesh = null;
        }
    }

    function clearThumbnailMeshes() {
        for (const entry of entries) {
            disposeMesh(entry.thumbnailMesh, false);
            entry.thumbnailMesh = null;
        }
    }

    function clearAtlasTextures() {
        for (const texturePromise of atlasTextures.values()) {
            texturePromise.then((texture) => texture?.dispose());
        }
        atlasTextures.clear();
    }

    function clearRenderedMeshes(clearAtlases = false) {
        clearPointMeshes();
        clearImageMeshes();
        clearThumbnailMeshes();
        if (clearAtlases) {
            clearAtlasTextures();
        }
    }

    function applyEntryStyle(entry) {
        const selected = entry.userData.selected;
        const hovered = entry === hoveredEntry;
        const pointMaterialKey = selected ? "selected" : hovered ? "hover" : "normal";

        if (entry.pointMesh) {
            entry.pointMesh.material = pointMaterials[pointMaterialKey];
            entry.pointMesh.userData.scaleBoost = selected ? 1.24 : hovered ? 1.16 : 1.0;
        }
        if (entry.imageMesh) {
            entry.imageMesh.material.color.set(selected ? SELECTED_COLOR : hovered ? HOVER_COLOR : "#ffffff");
            entry.imageMesh.material.opacity = hovered ? HOVER_OPACITY : POINT_OPACITY;
            entry.imageMesh.userData.scaleBoost = selected ? 1.24 : hovered ? 1.16 : 1.0;
        }
        if (entry.thumbnailMesh) {
            entry.thumbnailMesh.userData.scaleBoost = selected ? 1.2 : hovered ? 1.12 : 1.0;
        }
    }

    function updateEntryStyle(entry) {
        applyEntryStyle(entry);
        scheduleHybridThumbnailRebuild();
    }

    const selection = createSelectionController({
        entriesList: elements.entriesList,
        getPlanes: () => entries,
        preview,
        updatePlaneStyle: updateEntryStyle,
        updateTotals: setTotals,
    });

    function createPointMesh(entry) {
        const mesh = new THREE.Mesh(pointGeometry, pointMaterials.normal);
        mesh.position.copy(entry.position);
        mesh.visible = entry.visible;
        mesh.userData = { ...entry.userData, baseScale: 0.56, entry };
        entry.pointMesh = mesh;
        applyEntryStyle(entry);
        scene.add(mesh);
    }

    async function createImageMesh(entry, version) {
        const previewPath = visualToPreviewImage(entry.userData.visual);
        const fallbackPath = fallbackPreviewImage(entry.userData.visual);
        const texture = await loadPreviewTexture(previewPath, fallbackPath);
        if (!texture || version !== renderVersion) {
            texture?.dispose();
            return;
        }

        const width = texture.image.width || 256;
        const height = texture.image.height || 256;
        const baseHeight = 0.42;
        const mesh = new THREE.Mesh(
            new THREE.PlaneGeometry(baseHeight * width / Math.max(1, height), baseHeight),
            new THREE.MeshBasicMaterial({
                map: texture,
                opacity: POINT_OPACITY,
                transparent: true,
            }),
        );
        mesh.position.copy(entry.position);
        mesh.visible = entry.visible;
        mesh.userData = { ...entry.userData, baseScale: 1.0, entry };
        entry.imageMesh = mesh;
        applyEntryStyle(entry);
        scene.add(mesh);
    }

    async function loadPreviewTexture(previewPath, fallbackPath) {
        try {
            return await textureLoader.loadAsync(mediaUrl(previewPath));
        } catch {
            if (fallbackPath === previewPath) {
                return null;
            }
            try {
                return await textureLoader.loadAsync(mediaUrl(fallbackPath));
            } catch {
                return null;
            }
        }
    }

    async function atlasTexture(atlasName) {
        if (atlasTextures.has(atlasName)) {
            return atlasTextures.get(atlasName);
        }
        const texturePromise = textureLoader.loadAsync(`/static/${atlasName}?v=${renderVersion}`)
            .then((texture) => {
                texture.flipY = false;
                texture.needsUpdate = true;
                return texture;
            })
            .catch(() => null);
        atlasTextures.set(atlasName, texturePromise);
        return texturePromise;
    }

    async function createThumbnailMesh(entry, version) {
        const thumbnail = entry.userData.thumbnail;
        if (!thumbnail) {
            return;
        }

        const texture = await atlasTexture(thumbnail.atlas);
        if (!texture || version !== renderVersion || renderMode !== "hybrid") {
            return;
        }

        const mesh = new THREE.Mesh(
            thumbnailGeometry(thumbnail.uv),
            new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,
                depthWrite: false,
            }),
        );
        mesh.position.copy(entry.position);
        mesh.position.z = 0.1;
        mesh.visible = entry.visible;
        mesh.userData = {
            ...entry.userData,
            baseScale: HYBRID_THUMBNAIL_WORLD_SIZE,
            entry,
        };
        entry.thumbnailMesh = mesh;
        applyEntryStyle(entry);
        scene.add(mesh);
    }

    function thumbnailGeometry([u0, v0, u1, v1]) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute([
            -0.5, -0.5, 0,
            0.5, -0.5, 0,
            0.5, 0.5, 0,
            -0.5, 0.5, 0,
        ], 3));
        geometry.setAttribute("uv", new THREE.Float32BufferAttribute([
            u0, v1,
            u1, v1,
            u1, v0,
            u0, v0,
        ], 2));
        geometry.setIndex([0, 1, 2, 0, 2, 3]);
        return geometry;
    }

    function representativeEntries() {
        const chosen = new Map();
        const cells = new Map();

        for (const entry of entries) {
            if (!entry.visible || !entry.userData.thumbnail) {
                continue;
            }
            const screen = mapScene.screenPoint(entry.position);
            if (!screen.inside) {
                continue;
            }

            const cellX = Math.floor(screen.x / HYBRID_GRID_CELL_PX);
            const cellY = Math.floor(screen.y / HYBRID_GRID_CELL_PX);
            const key = `${cellX}:${cellY}`;
            const score = Math.hypot(
                screen.x - (cellX + 0.5) * HYBRID_GRID_CELL_PX,
                screen.y - (cellY + 0.5) * HYBRID_GRID_CELL_PX,
            );
            const current = cells.get(key);
            if (!current || score < current.score) {
                cells.set(key, { entry, score });
            }
        }

        for (const { entry } of cells.values()) {
            if (chosen.size >= HYBRID_THUMBNAIL_LIMIT) {
                break;
            }
            chosen.set(entry.userData.sourcePath, entry);
        }
        for (const sourcePath of selection.values()) {
            const selected = entries.find((entry) => entry.userData.sourcePath === sourcePath);
            if (selected?.visible && selected.userData.thumbnail) {
                chosen.set(sourcePath, selected);
            }
        }
        if (hoveredEntry?.visible && hoveredEntry.userData.thumbnail) {
            chosen.set(hoveredEntry.userData.sourcePath, hoveredEntry);
        }
        return Array.from(chosen.values());
    }

    function scheduleHybridThumbnailRebuild() {
        if (renderMode !== "hybrid" || hybridRebuildId !== null) {
            return;
        }
        hybridRebuildId = requestAnimationFrame(() => {
            hybridRebuildId = null;
            rebuildHybridThumbnails();
        });
    }

    async function rebuildHybridThumbnails() {
        const version = renderVersion;
        clearThumbnailMeshes();
        if (renderMode !== "hybrid") {
            return;
        }
        await Promise.all(representativeEntries().map((entry) => createThumbnailMesh(entry, version)));
    }

    async function renderCurrentMode(shouldFitInitialView = false) {
        const version = ++renderVersion;
        clearRenderedMeshes(false);

        if (renderMode === "images") {
            await Promise.all(entries.map((entry) => createImageMesh(entry, version)));
        } else {
            for (const entry of entries) {
                createPointMesh(entry);
            }
        }

        applyFilter(false, false);
        if (renderMode === "hybrid") {
            await rebuildHybridThumbnails();
        }
        setEmptyState(
            visibleEntries().length === 0,
            renderMode === "images"
                ? "No usable preview images were found for these discoveries."
                : "No displayable discoveries were found.",
            renderMode === "images" ? "No previews available" : "No discoveries available",
        );
        if (shouldFitInitialView) {
            fitView();
        }
    }

    function applyFilter(emitStatus = true, rebuildHybrid = true) {
        const search = buildDiscoveryMatcher(elements.searchInput.value);
        if (search.error) {
            updateStatus(`Invalid search pattern: ${search.error}`);
            return;
        }

        for (const entry of entries) {
            entry.visible = search.matcher(entry.userData.label);
            if (entry.pointMesh) {
                entry.pointMesh.visible = entry.visible;
            }
            if (entry.imageMesh) {
                entry.imageMesh.visible = entry.visible;
            }
            if (entry.thumbnailMesh) {
                entry.thumbnailMesh.visible = entry.visible;
            }
        }

        if (hoveredEntry && !hoveredEntry.visible) {
            hoveredEntry = null;
            preview.hide();
        }
        setTotals();
        if (rebuildHybrid) {
            scheduleHybridThumbnailRebuild();
        }

        if (!emitStatus) {
            return;
        }
        if (entries.length > 0 && visibleEntries().length === 0) {
            updateStatus("No discoveries match the filter.");
        } else if (entries.length > 0) {
            const modeLabel = search.mode === "empty" ? "" : ` (${search.mode})`;
            updateStatus(`${visibleEntries().length}/${entries.length} discoveries shown${modeLabel}.`);
        }
    }

    function updateRenderModeButtons() {
        for (const button of elements.viewModeControl.querySelectorAll("[data-render-mode]")) {
            button.classList.toggle("active", button.dataset.renderMode === renderMode);
        }
    }

    async function syncPoints(pointsData, shouldFitInitialView = false) {
        clearRenderedMeshes(true);
        hoveredEntry = null;
        preview.hide();
        entries = pointsData.map(createEntry);
        for (const sourcePath of selection.values()) {
            if (!entries.some((entry) => entry.userData.sourcePath === sourcePath)) {
                selection.unselect(sourcePath);
            }
        }
        await renderCurrentMode(shouldFitInitialView);
    }

    async function loadPoints(maxWaitMs = 0, shouldFitInitialView = false) {
        const pointsData = await readDiscoveries(maxWaitMs, () => {
            updateStatus("Waiting for discovery data...");
        });

        if (pointsData.length === 0) {
            clearRenderedMeshes(true);
            entries = [];
            setTotals();
            setEmptyState(
                true,
                "Start or refresh an experiment to populate this map.",
                "No discoveries available",
            );
            updateStatus("No discoveries found yet.");
            return;
        }

        await syncPoints(pointsData, shouldFitInitialView);
        updateStatus(`${visibleEntries().length}/${pointsData.length} discoveries shown as ${renderMode}.`);
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

    function pickEntryAtPointer(event) {
        const picked = mapScene.pickPlaneAtPointer(event, currentPickMeshes());
        return picked?.userData.entry || null;
    }

    function updateHoverState(event) {
        const nextHovered = pickEntryAtPointer(event);
        if (nextHovered !== hoveredEntry) {
            const previous = hoveredEntry;
            hoveredEntry = nextHovered;
            if (previous) {
                updateEntryStyle(previous);
            }
            if (hoveredEntry) {
                updateEntryStyle(hoveredEntry);
            }
        }

        if (hoveredEntry) {
            renderer.domElement.style.cursor = "pointer";
            preview.showForPlane(hoveredEntry, event);
        } else {
            renderer.domElement.style.cursor = "grab";
            preview.hide();
        }
    }

    function fitView() {
        mapScene.fitView(visibleEntries());
    }

    function resizeRenderer() {
        mapScene.resizeRenderer();
        scheduleHybridThumbnailRebuild();
    }

    function startAnimation() {
        mapScene.startAnimation(currentAnimatedMeshes);
    }

    function clearSelection() {
        selection.clear();
        updateStatus("Selection cleared.");
    }

    function selectedEntries() {
        return selection.values();
    }

    async function setRenderMode(mode) {
        if (!RENDER_MODES.has(mode) || mode === renderMode) {
            return;
        }
        renderMode = mode;
        updateRenderModeButtons();
        updateStatus(`Switching display to ${mode}...`);
        await renderCurrentMode(true);
        updateStatus(`${visibleEntries().length}/${entries.length} discoveries shown as ${mode}.`);
    }

    function markLiveRefreshNow() {
        lastLiveRefreshTimestamp = performance.now();
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

    renderer.domElement.addEventListener("pointerdown", (event) => {
        pointerDown = { x: event.clientX, y: event.clientY };
    });
    renderer.domElement.addEventListener("pointermove", updateHoverState);
    renderer.domElement.addEventListener("pointerleave", () => {
        const previous = hoveredEntry;
        hoveredEntry = null;
        if (previous) {
            updateEntryStyle(previous);
        }
        preview.hide();
    });
    renderer.domElement.addEventListener("click", (event) => {
        if (pointerDown && Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y) > 5) {
            return;
        }

        const entry = pickEntryAtPointer(event);
        if (entry) {
            selection.togglePlane(entry);
            preview.showForPlane(entry, event);
        }
    });
    mapScene.onViewChange(scheduleHybridThumbnailRebuild);
    updateRenderModeButtons();

    return {
        applyFilter,
        clearSelection,
        connectWebsocket,
        fitView,
        markLiveRefreshNow,
        refreshDiscoveries,
        resizeRenderer,
        selectedEntries,
        setRenderMode,
        startAnimation,
    };
}
