from __future__ import annotations

import json
from typing import Any

import numpy as np

from adtool.examples.visu.runtime import RuntimeState
from adtool.utils.interaction.experiment_control import (
    default_goal_targeting,
    read_experiment_control,
    write_experiment_control,
)


SUPPORTED_GOAL_TARGETING_PROJECTIONS = {"axis", "pca"}


def goal_targeting_enabled(config_path: str | None) -> bool:
    if not config_path:
        return False

    with open(config_path) as handle:
        config = json.load(handle)
    return bool(config.get("goal_oriented_sampler", False))


def _sanitize_zones(raw_zones: Any) -> list[dict[str, Any]]:
    zones = []
    for index, raw_zone in enumerate(raw_zones or []):
        center = raw_zone.get("center", [0.0, 0.0])
        zone_id = str(raw_zone.get("id", f"zone_{index}"))
        zones.append(
            {
                "id": zone_id,
                "center": [float(center[0]), float(center[1])],
            }
        )
    return zones


def _sanitize_radius(raw_radius: Any, fallback: float) -> float:
    radius = float(raw_radius if raw_radius is not None else fallback)
    return max(0.01, min(1.0, radius))


def resolve_goal_targeting(
    raw_goal_targeting: dict[str, Any] | None,
    runtime_state: RuntimeState,
    enabled: bool = True,
) -> dict[str, Any]:
    payload = {
        **default_goal_targeting(),
        **(raw_goal_targeting or {}),
    }
    payload["zones"] = _sanitize_zones(payload.get("zones"))
    payload["radius"] = _sanitize_radius(payload.get("radius"), default_goal_targeting()["radius"])

    online_state = runtime_state.online_update_state
    projection_method = runtime_state.projection_method
    projection_axes = list(runtime_state.projection_axes)

    if not enabled:
        return {
            **payload,
            "placement_supported": False,
            "projection_method": projection_method,
            "projection_axes": projection_axes,
            "message": "",
            "resolved": None,
        }

    if online_state is None:
        return {
            **payload,
            "placement_supported": False,
            "projection_method": projection_method,
            "projection_axes": projection_axes,
            "message": "Goal zones are unavailable until discoveries are loaded.",
            "resolved": None,
        }

    if projection_method not in SUPPORTED_GOAL_TARGETING_PROJECTIONS:
        return {
            **payload,
            "placement_supported": False,
            "projection_method": projection_method,
            "projection_axes": projection_axes,
            "message": "Goal zones are available only for axis and PCA projections.",
            "resolved": None,
        }

    projection_model = online_state.projection_model

    if projection_method == "axis":
        if projection_model.axis_center is None:
            return {
                **payload,
                "placement_supported": False,
                "projection_method": projection_method,
                "projection_axes": projection_axes,
                "message": "Axis goal zones are unavailable until the axis layout is ready.",
                "resolved": None,
            }

        resolved = None
        if payload["zones"]:
            resolved = {
                "kind": "axis",
                "projection_axes": projection_axes,
                "axis_center": np.asarray(projection_model.axis_center, dtype=float).tolist(),
                "axis_scale": float(projection_model.axis_scale),
                "zones": [
                    {
                        "id": zone["id"],
                        "center": zone["center"],
                        "radius": payload["radius"],
                    }
                    for zone in payload["zones"]
                ],
            }

        return {
            **payload,
            "placement_supported": True,
            "projection_method": projection_method,
            "projection_axes": projection_axes,
            "message": "",
            "resolved": resolved,
        }

    if projection_model.reducer is None or projection_model.input_mean is None or projection_model.input_std is None:
        return {
            **payload,
            "placement_supported": False,
            "projection_method": projection_method,
            "projection_axes": projection_axes,
            "message": "PCA goal zones need a fitted PCA layout.",
            "resolved": None,
        }

    resolved = None
    if payload["zones"]:
        resolved = {
            "kind": "pca",
            "input_mean": np.asarray(projection_model.input_mean, dtype=float).tolist(),
            "input_std": np.asarray(projection_model.input_std, dtype=float).tolist(),
            "pca_components": np.asarray(projection_model.reducer.components_, dtype=float).tolist(),
            "pca_mean": np.asarray(projection_model.reducer.mean_, dtype=float).tolist(),
            "output_center": np.asarray(projection_model.output_center, dtype=float).tolist(),
            "output_scale": float(projection_model.output_scale),
            "zones": [
                {
                    "id": zone["id"],
                    "center": zone["center"],
                    "radius": payload["radius"],
                }
                for zone in payload["zones"]
            ],
        }

    return {
        **payload,
        "placement_supported": True,
        "projection_method": projection_method,
        "projection_axes": projection_axes,
        "message": "",
        "resolved": resolved,
    }


def sync_goal_targeting(
    control_dir: str,
    runtime_state: RuntimeState,
    enabled: bool = True,
) -> dict[str, Any]:
    control = read_experiment_control(control_dir)
    goal_targeting = resolve_goal_targeting(
        control.get("goal_targeting"),
        runtime_state,
        enabled=enabled,
    )
    if goal_targeting != control.get("goal_targeting"):
        return write_experiment_control(control_dir, goal_targeting=goal_targeting)
    return {
        **control,
        "goal_targeting": goal_targeting,
    }
