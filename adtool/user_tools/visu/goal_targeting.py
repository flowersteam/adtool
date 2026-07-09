from __future__ import annotations

import json
from datetime import datetime, timezone
from math import pi, sqrt
from typing import Any

import numpy as np

from .coordinates import _ensure_2d_projection
from .runtime import RuntimeState
from adtool.utils.interaction.experiment_control import (
    DEFAULT_GOAL_ZONE_RADIUS,
    default_goal_targeting,
    persisted_goal_targeting,
    read_experiment_control,
    write_experiment_control,
)


SUPPORTED_GOAL_TARGETING_PROJECTIONS = {"axis", "pca"}
TARGET_ZONE_ANCHOR_COUNT = 64
AXIS_COMPLETION_NEIGHBORS = 5


def goal_targeting_enabled(config_path: str | None) -> bool:
    if not config_path:
        return False

    with open(config_path) as handle:
        config = json.load(handle)
    return bool(config.get("goal_oriented_sampler", False))


def _sanitize_radius(raw_radius: Any, fallback: float) -> float:
    radius = float(raw_radius if raw_radius is not None else fallback)
    return max(0.01, min(1.0, radius))


def _coverage_radius(radius: float, point_count: int) -> float:
    return max(radius / sqrt(max(1, point_count)), 1e-6)


def _synthetic_anchor_count(radius: float) -> int:
    return max(8, int(round(TARGET_ZONE_ANCHOR_COUNT * (radius / DEFAULT_GOAL_ZONE_RADIUS) ** 2)))


def _sanitize_zones(raw_zones: Any, radius: float) -> list[dict[str, Any]]:
    zones = []
    for index, raw_zone in enumerate(raw_zones or []):
        if not isinstance(raw_zone, dict):
            continue
        try:
            vectors = np.asarray(raw_zone.get("behavior_vectors", []), dtype=float)
        except (TypeError, ValueError):
            continue
        if vectors.ndim == 1 and vectors.size > 0:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or len(vectors) == 0:
            continue
        finite_mask = np.all(np.isfinite(vectors), axis=1)
        vectors = vectors[finite_mask]
        if len(vectors) == 0:
            continue
        zones.append(
            {
                "id": str(raw_zone.get("id", f"zone_{index}")),
                "behavior_vectors": vectors.tolist(),
                "coverage_radius": max(
                    1e-6,
                    float(raw_zone.get("coverage_radius", _coverage_radius(radius, _synthetic_anchor_count(radius)))),
                ),
            }
        )
    return zones


def _project_behavior_vectors_with_model(vectors: np.ndarray, projection_model: Any) -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or len(vectors) == 0:
        return np.empty((0, 2), dtype=float)

    if projection_model.method == "axis":
        if projection_model.axis_center is None:
            raise ValueError("Axis projection model is not ready.")
        points = vectors[:, [projection_model.axes[0], projection_model.axes[1]]]
        return (points - projection_model.axis_center) / projection_model.axis_scale

    if projection_model.method == "pca":
        reducer = projection_model.reducer
        if (
            reducer is None
            or projection_model.input_mean is None
            or projection_model.input_std is None
            or projection_model.output_center is None
            or getattr(reducer, "n_components_", 0) < 2
        ):
            raise ValueError("PCA projection model is not ready.")
        x_norm = (vectors - projection_model.input_mean) / projection_model.input_std
        projected = _ensure_2d_projection(reducer.transform(x_norm))
        return (projected - projection_model.output_center) / projection_model.output_scale

    raise ValueError(f"Unsupported projection method for goal targeting: {projection_model.method}")


def project_behavior_vectors(vectors: np.ndarray, runtime_state: RuntimeState) -> np.ndarray:
    online_state = runtime_state.online_update_state
    if online_state is None:
        raise ValueError("Goal targeting requires loaded discoveries.")
    return _project_behavior_vectors_with_model(np.asarray(vectors, dtype=float), online_state.projection_model)


def _goal_targeting_status(runtime_state: RuntimeState, enabled: bool) -> tuple[bool, str]:
    if not enabled:
        return False, ""

    online_state = runtime_state.online_update_state
    if online_state is None:
        return False, "Goal zones are unavailable until discoveries are loaded."

    projection_method = runtime_state.projection_method
    if projection_method not in SUPPORTED_GOAL_TARGETING_PROJECTIONS:
        return False, "Goal zones are available only for axis and PCA projections."

    projection_model = online_state.projection_model
    if projection_method == "axis":
        if projection_model.axis_center is None:
            return False, "Axis goal zones are unavailable until the axis layout is ready."
        return True, ""

    reducer = projection_model.reducer
    if (
        reducer is None
        or projection_model.input_mean is None
        or projection_model.input_std is None
        or projection_model.output_center is None
        or getattr(reducer, "n_components_", 0) < 2
    ):
        return False, "PCA goal zones need a fitted 2D PCA layout."
    return True, ""


def _projection_snapshot(runtime_state: RuntimeState) -> dict[str, Any]:
    projection_model = runtime_state.online_update_state.projection_model
    projection = {
        "projection_method": runtime_state.projection_method,
        "projection_axes": list(runtime_state.projection_axes),
    }
    if runtime_state.projection_method == "axis":
        projection["axis_center"] = np.asarray(projection_model.axis_center, dtype=float).tolist()
        projection["axis_scale"] = float(projection_model.axis_scale)
        return projection

    projection["input_mean"] = np.asarray(projection_model.input_mean, dtype=float).tolist()
    projection["input_std"] = np.asarray(projection_model.input_std, dtype=float).tolist()
    projection["pca_components"] = np.asarray(projection_model.reducer.components_, dtype=float).tolist()
    projection["pca_mean"] = np.asarray(projection_model.reducer.mean_, dtype=float).tolist()
    projection["output_center"] = np.asarray(projection_model.output_center, dtype=float).tolist()
    projection["output_scale"] = float(projection_model.output_scale)
    return projection


def _normalize_goal_targeting(raw_goal_targeting: dict[str, Any] | None) -> dict[str, Any]:
    payload = {
        **default_goal_targeting(),
        **(raw_goal_targeting or {}),
    }
    payload["radius"] = _sanitize_radius(payload.get("radius"), default_goal_targeting()["radius"])
    payload["zones"] = _sanitize_zones(payload.get("zones"), payload["radius"])
    return payload


def _sunflower_disk_points(center: np.ndarray, radius: float, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=float)
    golden_angle = pi * (3.0 - sqrt(5.0))
    points = np.empty((count, 2), dtype=float)
    for index in range(count):
        distance = radius * sqrt((index + 0.5) / count)
        angle = index * golden_angle
        points[index] = np.array(
            [
                center[0] + distance * np.cos(angle),
                center[1] + distance * np.sin(angle),
            ],
            dtype=float,
        )
    return points


def _inverse_project_pca_points(points: np.ndarray, projection_model: Any) -> np.ndarray:
    target_pca = projection_model.output_center + points * projection_model.output_scale
    x_norm = projection_model.reducer.inverse_transform(target_pca)
    return x_norm * projection_model.input_std + projection_model.input_mean


def _inverse_project_axis_points(
    points: np.ndarray,
    projected_matrix: np.ndarray,
    behavior_matrix: np.ndarray,
    projection_model: Any,
) -> np.ndarray:
    axes = list(projection_model.axes)
    global_mean = behavior_matrix.mean(axis=0)
    k = min(AXIS_COMPLETION_NEIGHBORS, len(behavior_matrix))
    completed = np.empty((len(points), behavior_matrix.shape[1]), dtype=float)
    for index, point in enumerate(points):
        if k > 0:
            distances = np.linalg.norm(projected_matrix - point.reshape(1, 2), axis=1)
            neighbor_indices = np.argsort(distances)[:k]
            base = behavior_matrix[neighbor_indices].mean(axis=0)
        else:
            base = global_mean
        if not np.all(np.isfinite(base)):
            base = global_mean
        candidate = np.array(base, copy=True)
        candidate[axes[0]] = projection_model.axis_center[0] + point[0] * projection_model.axis_scale
        candidate[axes[1]] = projection_model.axis_center[1] + point[1] * projection_model.axis_scale
        completed[index] = candidate
    return completed


def _build_zone_vectors(center: np.ndarray, radius: float, runtime_state: RuntimeState) -> np.ndarray:
    online_state = runtime_state.online_update_state
    if online_state is None:
        raise ValueError("Goal targeting requires loaded discoveries.")

    behavior_matrix = np.asarray(online_state.all_embedding_matrix, dtype=float)
    if behavior_matrix.ndim != 2 or len(behavior_matrix) == 0:
        raise ValueError("Goal targeting requires loaded discoveries.")

    projected = project_behavior_vectors(behavior_matrix, runtime_state)
    inside_mask = np.linalg.norm(projected - center.reshape(1, 2), axis=1) <= radius
    real_vectors = behavior_matrix[inside_mask]
    synthetic_anchor_count = _synthetic_anchor_count(radius)
    coverage_radius = _coverage_radius(radius, synthetic_anchor_count)
    synthetic_points = _sunflower_disk_points(center, radius, synthetic_anchor_count)
    if len(real_vectors) > 0:
        real_points = projected[inside_mask]
        synthetic_points = synthetic_points[
            np.array(
                [
                    np.all(np.linalg.norm(real_points - point.reshape(1, 2), axis=1) > coverage_radius)
                    for point in synthetic_points
                ],
                dtype=bool,
            )
        ]
    if len(synthetic_points) == 0:
        return real_vectors

    projection_model = online_state.projection_model
    if projection_model.method == "pca":
        synthetic_vectors = _inverse_project_pca_points(synthetic_points, projection_model)
    elif projection_model.method == "axis":
        synthetic_vectors = _inverse_project_axis_points(
            synthetic_points,
            projected,
            behavior_matrix,
            projection_model,
        )
    else:
        raise ValueError("Goal zones are available only for axis and PCA projections.")

    if len(real_vectors) == 0:
        return synthetic_vectors
    return np.vstack([real_vectors, synthetic_vectors])


def apply_goal_targeting_action(
    raw_goal_targeting: dict[str, Any] | None,
    action_payload: dict[str, Any],
    runtime_state: RuntimeState,
    enabled: bool = True,
) -> dict[str, Any]:
    payload = _normalize_goal_targeting(raw_goal_targeting)
    zones = list(payload["zones"])

    action = str(action_payload.get("action", "")).strip().lower()
    if action not in {"add", "delete", "clear", "set_radius"}:
        raise ValueError("Goal targeting action must be one of add, delete, clear, or set_radius.")

    next_radius = _sanitize_radius(action_payload.get("radius"), payload["radius"])
    if action == "set_radius":
        return {
            **persisted_goal_targeting(payload),
            "radius": next_radius,
            "zones": [
                {
                    **zone,
                    "coverage_radius": _coverage_radius(next_radius, _synthetic_anchor_count(next_radius)),
                }
                for zone in zones
            ],
            "resolved": None,
        }
    if action == "clear":
        return {
            **persisted_goal_targeting(payload),
            "radius": next_radius,
            "zones": [],
            "resolved": None,
        }
    if action == "delete":
        zone_id = str(action_payload.get("zone_id", "")).strip()
        return {
            **persisted_goal_targeting(payload),
            "radius": next_radius,
            "zones": [zone for zone in zones if zone["id"] != zone_id],
            "resolved": None,
        }

    supported, message = _goal_targeting_status(runtime_state, enabled)
    if not supported:
        raise ValueError(message or "Goal targeting is unavailable.")

    center = action_payload.get("center")
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise ValueError("Goal targeting add action requires a 2D center.")

    try:
        center_point = np.asarray([float(center[0]), float(center[1])], dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Goal targeting add action requires numeric coordinates.") from None

    zone_vectors = _build_zone_vectors(center_point, next_radius, runtime_state)
    zones.append(
        {
            "id": f"zone_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}",
            "behavior_vectors": zone_vectors.tolist(),
            "coverage_radius": _coverage_radius(next_radius, _synthetic_anchor_count(next_radius)),
        }
    )
    return {
        **persisted_goal_targeting(payload),
        "radius": next_radius,
        "zones": zones,
        "resolved": None,
    }


def resolve_goal_targeting(
    raw_goal_targeting: dict[str, Any] | None,
    runtime_state: RuntimeState,
    enabled: bool = True,
) -> dict[str, Any]:
    payload = _normalize_goal_targeting(raw_goal_targeting)
    supported, message = _goal_targeting_status(runtime_state, enabled)
    response = {
        **payload,
        "placement_supported": supported,
        "message": message,
        "display_zones": [],
        "resolved": None,
    }
    if not supported:
        return response

    response["resolved"] = {
        "radius": payload["radius"],
        **_projection_snapshot(runtime_state),
        "zones": [
            {
                "id": zone["id"],
                "behavior_vectors": zone["behavior_vectors"],
                "coverage_radius": float(zone["coverage_radius"]),
            }
            for zone in payload["zones"]
        ],
    }
    response["display_zones"] = []
    for zone in payload["zones"]:
        points = project_behavior_vectors(np.asarray(zone["behavior_vectors"], dtype=float), runtime_state)
        response["display_zones"].append(
            {
                "id": zone["id"],
                "points": points.tolist(),
                "centroid": (points.mean(axis=0) if len(points) > 0 else np.zeros(2, dtype=float)).tolist(),
                "point_count": int(len(points)),
            }
        )
    return response


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
    current_persisted = persisted_goal_targeting(control.get("goal_targeting"))
    next_persisted = persisted_goal_targeting(goal_targeting)
    if next_persisted != current_persisted:
        control = write_experiment_control(control_dir, goal_targeting=goal_targeting)
    return {
        **control,
        "goal_targeting": goal_targeting,
    }
