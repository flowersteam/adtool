import { getGoalTargeting, setGoalTargeting } from "../api.js";

function zoneLabel(index) {
    return `Zone ${index + 1}`;
}

function formatCoord(value) {
    return Number(value || 0).toFixed(3);
}

export function createGoalTargetingController({
    discoveryMap,
    elements,
    updateStatus,
}) {
    let state = null;
    let placementActive = false;

    function currentGoalTargeting() {
        return state?.goal_targeting || {
            radius: Number(elements.goalZoneRadiusInput.value || 0.18),
            zones: [],
            placement_supported: false,
            message: "",
        };
    }

    function updateRadiusLabel(value) {
        elements.goalZoneRadiusValue.textContent = Number(value || 0).toFixed(2);
    }

    function setPlacementActive(active) {
        placementActive = Boolean(active);
        elements.goalZonePlacementButton.classList.toggle("active", placementActive);
        elements.goalZonePlacementButton.textContent = placementActive ? "Placing..." : "Place zone";
        discoveryMap.setGoalZonePlacement(placementActive, placementActive ? placeZone : null);
    }

    function renderList(goalTargeting) {
        elements.goalZoneList.replaceChildren();

        if ((goalTargeting.zones || []).length === 0) {
            const empty = document.createElement("div");
            empty.className = "goalZoneEmpty";
            empty.textContent = goalTargeting.placement_supported
                ? "No active goal zones."
                : "Goal zones are disabled for this projection.";
            elements.goalZoneList.appendChild(empty);
            return;
        }

        goalTargeting.zones.forEach((zone, index) => {
            const item = document.createElement("div");
            item.className = "goalZoneItem";

            const text = document.createElement("div");
            const title = document.createElement("strong");
            title.textContent = zoneLabel(index);
            const meta = document.createElement("span");
            meta.textContent = `x ${formatCoord(zone.center[0])} · y ${formatCoord(zone.center[1])}`;
            text.appendChild(title);
            text.appendChild(meta);

            const removeButton = document.createElement("button");
            removeButton.type = "button";
            removeButton.textContent = "Delete";
            removeButton.dataset.zoneId = zone.id;

            item.appendChild(text);
            item.appendChild(removeButton);
            elements.goalZoneList.appendChild(item);
        });
    }

    function render(payload) {
        state = payload;
        const goalTargeting = currentGoalTargeting();
        elements.goalTargetingSection.hidden = !payload.feature_enabled;
        if (elements.goalTargetingSection.hidden) {
            setPlacementActive(false);
            discoveryMap.renderGoalZones([], 0);
            return;
        }

        const supported = Boolean(goalTargeting.placement_supported);
        elements.goalZoneControls.hidden = !supported;
        elements.goalZoneUnsupportedMessage.hidden = supported;
        elements.goalZoneSupportText.hidden = !supported;
        elements.goalZoneRadiusControl.hidden = !supported;
        elements.goalZoneList.hidden = !supported;
        elements.goalZoneSupportText.textContent = goalTargeting.message || "Click on the map to place circular goal zones.";
        elements.goalZoneRadiusInput.value = `${goalTargeting.radius}`;
        updateRadiusLabel(goalTargeting.radius);
        elements.goalZonePlacementButton.disabled = !supported;
        elements.goalZoneClearButton.disabled = (goalTargeting.zones || []).length === 0;
        if (!supported) {
            setPlacementActive(false);
            discoveryMap.renderGoalZones([], 0);
            renderList({ ...goalTargeting, zones: [] });
            return;
        }

        discoveryMap.renderGoalZones(goalTargeting.zones || [], Number(elements.goalZoneRadiusInput.value));
        renderList(goalTargeting);
    }

    async function refresh() {
        try {
            render(await getGoalTargeting());
        } catch (error) {
            updateStatus(error.message || "Goal targeting unavailable.");
        }
    }

    async function save(nextGoalTargeting) {
        try {
            render(await setGoalTargeting(nextGoalTargeting));
        } catch (error) {
            updateStatus(error.message || "Goal targeting update failed.");
        }
    }

    async function placeZone(point) {
        const goalTargeting = currentGoalTargeting();
        if (!goalTargeting.placement_supported) {
            return;
        }

        const zones = [
            ...(goalTargeting.zones || []),
            {
                id: `zone_${Date.now()}_${Math.floor(Math.random() * 1_000_000)}`,
                center: [point.x, point.y],
            },
        ];
        await save({
            radius: Number(elements.goalZoneRadiusInput.value),
            zones,
        });
        updateStatus(`Goal zone added at (${formatCoord(point.x)}, ${formatCoord(point.y)}).`);
    }

    async function clearZones() {
        await save({
            radius: Number(elements.goalZoneRadiusInput.value),
            zones: [],
        });
        updateStatus("Goal zones cleared.");
    }

    async function deleteZone(zoneId) {
        const goalTargeting = currentGoalTargeting();
        const zones = (goalTargeting.zones || []).filter((zone) => zone.id !== zoneId);
        await save({
            radius: Number(elements.goalZoneRadiusInput.value),
            zones,
        });
        updateStatus("Goal zone removed.");
    }

    function bindEvents() {
        elements.goalZonePlacementButton.addEventListener("click", () => {
            if (!currentGoalTargeting().placement_supported) {
                return;
            }
            setPlacementActive(!placementActive);
        });
        elements.goalZoneClearButton.addEventListener("click", clearZones);
        elements.goalZoneRadiusInput.addEventListener("input", () => {
            updateRadiusLabel(elements.goalZoneRadiusInput.value);
            discoveryMap.renderGoalZones(
                currentGoalTargeting().zones || [],
                Number(elements.goalZoneRadiusInput.value),
            );
        });
        elements.goalZoneRadiusInput.addEventListener("change", async () => {
            await save({
                radius: Number(elements.goalZoneRadiusInput.value),
                zones: currentGoalTargeting().zones || [],
            });
            updateStatus("Goal zone radius updated.");
        });
        elements.goalZoneList.addEventListener("click", async (event) => {
            const button = event.target.closest("[data-zone-id]");
            if (!button) {
                return;
            }
            await deleteZone(button.dataset.zoneId);
        });
    }

    async function initialize() {
        bindEvents();
        await refresh();
    }

    return {
        initialize,
        refresh,
    };
}
