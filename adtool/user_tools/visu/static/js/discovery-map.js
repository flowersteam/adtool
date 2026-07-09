import * as THREE from "three";

import {
    DISCOVERY_LOAD_GRACE_MS,
    HOVER_OPACITY,
    IMAGE_PREVIEW_WORLD_HEIGHT,
    HYBRID_GRID_CELL_PX,
    HYBRID_PREVIEW_LIMIT,
    LIVE_REFRESH_COOLDOWN_MS,
    POINT_OPACITY,
    SCALE_FACTOR,
} from "./config.js";
import { readDiscoveries } from "./api.js";
import { createHighlightController } from "./highlights/controller.js";
import { createHighlightMaterialCache } from "./highlights/materials.js";
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
const SELECTION_LIST_HOVER_COLOR = "#e63946";
const GOAL_ZONE_FILL_COLOR = "#f4a261";
const GOAL_ZONE_LINE_COLOR = "#e76f51";
const GOAL_ZONE_POINT_RADIUS = 0.06 * SCALE_FACTOR;
const GOAL_ZONE_POINT_OPACITY = 0.5;
const GOAL_ZONE_SEGMENTS = 72;
const MAX_PREVIEW_SCALE_BOOST = 1.24;
const HYBRID_VIEW_REBUILD_DEBOUNCE_MS = 120;

export function createDiscoveryMap({ elements, preview, updateStatus }) {
    const mapScene = createMapScene(elements.app);
    const { renderer, scene, textureLoader } = mapScene;

    const pointGeometry = new THREE.CircleGeometry(0.12, 14);
    const pointRingGeometry = new THREE.RingGeometry(0.145, 0.19, 24);
    const pointMaterials = {
        normal: pointMaterial(POINT_COLOR),
        ringHover: pointMaterial(HOVER_COLOR),
        ringSelected: pointMaterial(SELECTED_COLOR),
        ringSelectionListHover: pointMaterial(SELECTION_LIST_HOVER_COLOR),
    };
    const highlightMaterials = createHighlightMaterialCache({
        THREE,
        opacity: POINT_OPACITY,
    });
    let entries = [];
    let renderMode = "points";
    let hoveredEntry = null;
    let isRefreshing = false;
    let pendingRefresh = false;
    let pointerDown = null;
    let liveRefreshTimerId = null;
    let liveRefreshCooldownMs = LIVE_REFRESH_COOLDOWN_MS;
    let lastLiveRefreshTimestamp = 0;
    let renderVersion = 0;
    let hybridRebuildId = null;
    let hybridRebuildTimeoutId = null;
    let hybridPreviewToken = 0;
    let stickerPreviewWorldHeight = IMAGE_PREVIEW_WORLD_HEIGHT;
    let goalZoneGroups = [];
    let goalZonePlacementActive = false;
    let goalZonePlacementHandler = null;
    let focusedSelectionSource = null;

    function refreshHighlightStyles() {
        for (const entry of entries) {
            applyEntryStyle(entry);
        }
    }

    const highlights = createHighlightController({
        elements,
        onRulesChange: (patch = {}) => {
            refreshHighlightStyles();
            if (Object.keys(patch).length === 1 && patch.color !== undefined) {
                return;
            }
            applyFilter();
        },
    });

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
            filters: point.filters || {},
            pointMesh: null,
            pointRingMesh: null,
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
                visual: normalizeVisualPath(point.visual),
            },
        };
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
                ...entries.map((entry) => entry.imageMesh).filter(Boolean),
                ...entries.map((entry) => entry.pointMesh).filter(Boolean),
            ];
        }
        return entries.map((entry) => entry.pointMesh).filter(Boolean);
    }

    function currentAnimatedMeshes() {
        return entries.flatMap((entry) => [
            entry.pointMesh,
            entry.pointRingMesh,
            entry.imageMesh,
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
            if (entry.pointRingMesh) {
                scene.remove(entry.pointRingMesh);
            }
            entry.pointMesh = null;
            entry.pointRingMesh = null;
        }
    }

    function clearImageMeshes() {
        for (const entry of entries) {
            disposeMesh(entry.imageMesh, true);
            entry.imageMesh = null;
        }
    }

    function clearRenderedMeshes() {
        clearPointMeshes();
        clearImageMeshes();
    }

    function clearGoalZoneMeshes() {
        for (const group of goalZoneGroups) {
            for (const child of group.children) {
                child.geometry?.dispose();
                child.material?.dispose();
            }
            scene.remove(group);
        }
        goalZoneGroups = [];
    }

    function createGoalZoneCoverageMesh(points) {
        const group = new THREE.Group();
        const geometry = new THREE.CircleGeometry(GOAL_ZONE_POINT_RADIUS, 18);
        const material = new THREE.MeshBasicMaterial({
            color: GOAL_ZONE_FILL_COLOR,
            transparent: true,
            opacity: GOAL_ZONE_POINT_OPACITY,
            depthTest: true,
            depthWrite: true,
            depthFunc: THREE.LessDepth,
        });
        const coverage = new THREE.InstancedMesh(geometry, material, points.length);
        const matrix = new THREE.Matrix4();
        for (const [index, point] of points.entries()) {
            matrix.makeTranslation(point.x, point.y, point.z);
            coverage.setMatrixAt(index, matrix);
        }
        coverage.instanceMatrix.needsUpdate = true;
        coverage.renderOrder = 5;
        group.add(coverage);
        return group;
    }

    function createGoalZoneGroup(zone) {
        const points = (zone.points || []).map((point) => new THREE.Vector3(
            SCALE_FACTOR * Number(point[0] || 0),
            SCALE_FACTOR * Number(point[1] || 0),
            0.03,
        ));
        const centroid = new THREE.Vector3(
            SCALE_FACTOR * Number(zone.centroid?.[0] || 0),
            SCALE_FACTOR * Number(zone.centroid?.[1] || 0),
            0.02,
        );
        const group = createGoalZoneCoverageMesh(points);
        const marker = new THREE.Mesh(
            new THREE.CircleGeometry(0.06 * SCALE_FACTOR, 24),
            new THREE.MeshBasicMaterial({
                color: GOAL_ZONE_LINE_COLOR,
                opacity: 0.95,
                transparent: true,
                depthTest: false,
                depthWrite: false,
            }),
        );
        marker.position.copy(centroid);
        marker.scale.set(0.12, 0.12, 1);
        marker.renderOrder = 10;

        group.add(marker);
        scene.add(group);
        return group;
    }

    function renderGoalZones(zones) {
        clearGoalZoneMeshes();
        for (const zone of zones || []) {
            if ((zone.points || []).length === 0) {
                continue;
            }
            goalZoneGroups.push(createGoalZoneGroup(zone));
        }
    }

    function applyEntryStyle(entry) {
        const selected = entry.userData.selected;
        const hovered = entry === hoveredEntry;
        const selectionListHovered = entry.userData.sourcePath === focusedSelectionSource;
        const normalMaterial = pointBaseMaterial(entry);

        if (entry.pointMesh) {
            entry.pointMesh.material = normalMaterial;
            entry.pointMesh.userData.scaleBoost = selected ? 1.24 : hovered ? 1.16 : 1.0;
        }
        if (entry.pointRingMesh) {
            entry.pointRingMesh.visible = entry.visible && (
                selected
                || hovered
                || selectionListHovered
            );
            entry.pointRingMesh.material = selectionListHovered
                ? pointMaterials.ringSelectionListHover
                : selected
                ? pointMaterials.ringSelected
                : pointMaterials.ringHover;
            entry.pointRingMesh.userData.scaleBoost = selected ? 1.32 : 1.22;
        }
        if (entry.imageMesh) {
            entry.imageMesh.material.color.set(selected ? SELECTED_COLOR : hovered ? HOVER_COLOR : "#ffffff");
            entry.imageMesh.material.opacity = hovered ? HOVER_OPACITY : POINT_OPACITY;
            entry.imageMesh.userData.scaleBoost = selected ? 1.24 : hovered ? 1.16 : 1.0;
        }
    }

    function pointBaseMaterial(entry) {
        const colors = highlights.matchedColors(entry.filters);
        if (colors.length === 0) {
            return pointMaterials.normal;
        }
        return highlightMaterials.materialForColors(colors);
    }

    function updateEntryStyle(entry, rebuildHybrid = true) {
        applyEntryStyle(entry);
        if (rebuildHybrid) {
            queueHybridPreviewRebuild();
        }
    }

    const selection = createSelectionController({
        entriesList: elements.entriesList,
        getPlanes: () => entries,
        setFocusedSource: (sourcePath) => {
            if (focusedSelectionSource === sourcePath) {
                return;
            }
            const previousEntry = focusedSelectionSource === null
                ? null
                : entries.find((entry) => entry.userData.sourcePath === focusedSelectionSource) || null;
            focusedSelectionSource = sourcePath;
            if (previousEntry) {
                updateEntryStyle(previousEntry, false);
            }
            if (focusedSelectionSource !== null) {
                const nextEntry = entries.find(
                    (entry) => entry.userData.sourcePath === focusedSelectionSource,
                );
                if (nextEntry) {
                    updateEntryStyle(nextEntry, false);
                }
            }
        },
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

        const ringMesh = new THREE.Mesh(pointRingGeometry, pointMaterials.ringHover);
        ringMesh.position.copy(entry.position);
        ringMesh.visible = false;
        ringMesh.userData = { ...entry.userData, baseScale: 0.56, entry };
        entry.pointRingMesh = ringMesh;

        applyEntryStyle(entry);
        scene.add(mesh);
        scene.add(ringMesh);
    }

    function rectsOverlap(a, b) {
        return a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
    }

    async function createImageMesh(
        entry,
        version,
        baseHeight = IMAGE_PREVIEW_WORLD_HEIGHT,
        { rebuildToken = null, occupiedRects = null } = {},
    ) {
        const previewPath = visualToPreviewImage(entry.userData.visual);
        const fallbackPath = fallbackPreviewImage(entry.userData.visual);
        const texture = await loadPreviewTexture(previewPath, fallbackPath);
        if (
            !texture
            || version !== renderVersion
            || (rebuildToken !== null && rebuildToken !== hybridPreviewToken)
        ) {
            texture?.dispose();
            return;
        }

        const width = texture.image.width || 256;
        const height = texture.image.height || 256;
        const geometryWidth = baseHeight * width / Math.max(1, height);
        if (occupiedRects) {
            const candidateRect = mapScene.planeScreenRect(
                entry.position,
                geometryWidth,
                baseHeight,
                baseHeight,
                MAX_PREVIEW_SCALE_BOOST,
            );
            if (occupiedRects.some((occupiedRect) => rectsOverlap(candidateRect, occupiedRect))) {
                texture.dispose();
                return;
            }
            occupiedRects.push(candidateRect);
        }

        const mesh = new THREE.Mesh(
            new THREE.PlaneGeometry(geometryWidth, baseHeight),
            new THREE.MeshBasicMaterial({
                map: texture,
                opacity: POINT_OPACITY,
                transparent: true,
            }),
        );
        mesh.position.copy(entry.position);
        mesh.visible = entry.visible;
        mesh.userData = { ...entry.userData, baseScale: baseHeight, entry };
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

    function representativeEntries() {
        const candidates = [];
        const cells = new Map();

        for (const entry of entries) {
            if (!entry.visible) {
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

        for (const sourcePath of selection.values()) {
            const selected = entries.find((entry) => entry.userData.sourcePath === sourcePath);
            if (selected?.visible) {
                candidates.push(selected);
            }
        }
        const cellCandidates = Array.from(cells.values())
            .sort((left, right) => left.score - right.score)
            .map(({ entry }) => entry);
        candidates.push(...cellCandidates);

        const chosen = new Map();
        for (const entry of candidates) {
            if (chosen.size >= HYBRID_PREVIEW_LIMIT) {
                break;
            }
            chosen.set(entry.userData.sourcePath, entry);
        }
        return Array.from(chosen.values());
    }

    function queueHybridPreviewRebuild(delayMs = 0) {
        if (renderMode !== "hybrid") {
            return;
        }
        if (hybridRebuildTimeoutId !== null) {
            clearTimeout(hybridRebuildTimeoutId);
            hybridRebuildTimeoutId = null;
        }

        const scheduleFrame = () => {
            if (hybridRebuildId !== null) {
                return;
            }
            hybridRebuildId = requestAnimationFrame(() => {
                hybridRebuildId = null;
                rebuildHybridPreviews();
            });
        };

        if (delayMs > 0) {
            hybridRebuildTimeoutId = window.setTimeout(() => {
                hybridRebuildTimeoutId = null;
                scheduleFrame();
            }, delayMs);
            return;
        }

        scheduleFrame();
    }

    function cancelHybridPreviewRebuild() {
        if (hybridRebuildTimeoutId !== null) {
            clearTimeout(hybridRebuildTimeoutId);
            hybridRebuildTimeoutId = null;
        }
        if (hybridRebuildId !== null) {
            cancelAnimationFrame(hybridRebuildId);
            hybridRebuildId = null;
        }
    }

    async function rebuildHybridPreviews() {
        const version = renderVersion;
        const rebuildToken = ++hybridPreviewToken;
        const occupiedRects = [];
        clearImageMeshes();
        if (renderMode !== "hybrid") {
            return;
        }
        for (const entry of representativeEntries()) {
            await createImageMesh(entry, version, stickerPreviewWorldHeight, {
                rebuildToken,
                occupiedRects,
            });
        }
    }

    async function renderCurrentMode(shouldFitInitialView = false) {
        const version = ++renderVersion;
        cancelHybridPreviewRebuild();
        hybridPreviewToken += 1;
        clearRenderedMeshes();

        if (renderMode === "images") {
            await Promise.all(entries.map((entry) => createImageMesh(entry, version)));
        } else {
            for (const entry of entries) {
                createPointMesh(entry);
            }
        }

        applyFilter(false, false);
        if (renderMode === "hybrid") {
            await rebuildHybridPreviews();
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
            entry.visible = (
                search.matcher(entry.userData.label)
                && highlights.isVisible(entry.filters)
            );
            if (entry.pointMesh) {
                entry.pointMesh.visible = entry.visible;
            }
            if (entry.pointRingMesh) {
                entry.pointRingMesh.visible = entry.visible && (
                    entry.userData.selected || entry === hoveredEntry
                );
            }
            if (entry.imageMesh) {
                entry.imageMesh.visible = entry.visible;
            }
        }

        if (hoveredEntry && !hoveredEntry.visible) {
            hoveredEntry = null;
            preview.hide();
        }
        setTotals();
        if (rebuildHybrid) {
            queueHybridPreviewRebuild();
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
        clearRenderedMeshes();
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
        const [pointsData] = await Promise.all([
            readDiscoveries(maxWaitMs, () => {
                updateStatus("Waiting for discovery data...");
            }),
            highlights.refreshSchema(),
        ]);

        if (pointsData.length === 0) {
            clearRenderedMeshes();
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
        if (goalZonePlacementActive) {
            if (hoveredEntry) {
                const previous = hoveredEntry;
                hoveredEntry = null;
                updateEntryStyle(previous, false);
            }
            renderer.domElement.style.cursor = "crosshair";
            preview.hide();
            return;
        }

        const nextHovered = pickEntryAtPointer(event);
        if (nextHovered !== hoveredEntry) {
            const previous = hoveredEntry;
            hoveredEntry = nextHovered;
            if (previous) {
                updateEntryStyle(previous, false);
            }
            if (hoveredEntry) {
                updateEntryStyle(hoveredEntry, false);
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
        queueHybridPreviewRebuild(HYBRID_VIEW_REBUILD_DEBOUNCE_MS);
    }

    function startAnimation() {
        mapScene.startAnimation(currentAnimatedMeshes);
    }

    function clearSelection() {
        selection.clear();
        updateStatus("Selection cleared.");
    }

    function setGoalZonePlacement(active, onPlace) {
        goalZonePlacementActive = Boolean(active);
        goalZonePlacementHandler = onPlace || null;
        if (!goalZonePlacementActive) {
            renderer.domElement.style.cursor = hoveredEntry ? "pointer" : "grab";
        } else {
            preview.hide();
            renderer.domElement.style.cursor = "crosshair";
        }
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

    async function setRenderSettings(settings) {
        const nextPreviewHeight = Number(settings.sticker_preview_world_height);
        stickerPreviewWorldHeight = Number.isFinite(nextPreviewHeight)
            ? nextPreviewHeight
            : IMAGE_PREVIEW_WORLD_HEIGHT;
        if (renderMode === "hybrid") {
            await renderCurrentMode(false);
        }
    }

    function markLiveRefreshNow() {
        lastLiveRefreshTimestamp = performance.now();
    }

    function scheduleLiveRefresh() {
        if (liveRefreshTimerId !== null) {
            return;
        }

        const elapsed = performance.now() - lastLiveRefreshTimestamp;
        const delay = Math.max(0, liveRefreshCooldownMs - elapsed);

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

    function setLiveRefreshCooldown(ms) {
        liveRefreshCooldownMs = Math.max(0, Number(ms) || LIVE_REFRESH_COOLDOWN_MS);
    }

    renderer.domElement.addEventListener("pointerdown", (event) => {
        pointerDown = { x: event.clientX, y: event.clientY };
    });
    renderer.domElement.addEventListener("pointermove", updateHoverState);
    renderer.domElement.addEventListener("pointerleave", () => {
        const previous = hoveredEntry;
        hoveredEntry = null;
        if (previous) {
            updateEntryStyle(previous, false);
        }
        preview.hide();
    });
    renderer.domElement.addEventListener("click", (event) => {
        if (pointerDown && Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y) > 5) {
            return;
        }

        if (goalZonePlacementActive && goalZonePlacementHandler) {
            const point = mapScene.worldPointAtPointer(event);
            if (point) {
                goalZonePlacementHandler({
                    x: point.x / SCALE_FACTOR,
                    y: point.y / SCALE_FACTOR,
                });
            }
            return;
        }

        const entry = pickEntryAtPointer(event);
        if (entry) {
            selection.togglePlane(entry);
            preview.showForPlane(entry, event);
        }
    });
    mapScene.onViewChange(() => queueHybridPreviewRebuild(HYBRID_VIEW_REBUILD_DEBOUNCE_MS));
    updateRenderModeButtons();

    return {
        applyFilter,
        clearSelection,
        connectWebsocket,
        fitView,
        markLiveRefreshNow,
        refreshDiscoveries,
        renderGoalZones,
        resizeRenderer,
        selectedEntries,
        setGoalZonePlacement,
        setLiveRefreshCooldown,
        setRenderSettings,
        setRenderMode,
        startAnimation,
    };
}
