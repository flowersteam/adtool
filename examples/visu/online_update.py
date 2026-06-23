from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from adtool.examples.visu.coordinates import (
    _cache_mtime,
    _cached_layout,
    _dataset_layout_input,
    _downsample_for_display,
    _ensure_2d_projection,
    _iter_discovery_paths,
    _normalize_embedding_matrix,
    _normalize_projection_bounds,
    _purge_stale_caches,
    _saved_coordinates,
    _scan_discoveries,
    _write_empty_layout,
    _write_highlight_schema,
    _write_layout_result,
    export_last_frame,
    process_discovery,
)
from adtool.examples.visu.highlights import load_highlight_export_context
from adtool.examples.visu.runtime import (
    ONLINE_FULL_RECOMPUTE_INTERVAL_SECONDS,
    ONLINE_POINT_UPDATE_INTERVAL_SECONDS,
    RuntimeState,
    ServerConfig,
)


@dataclass
class OnlineProjectionModel:
    method: str
    axes: tuple[int, int]
    input_mean: np.ndarray | None = None
    input_std: np.ndarray | None = None
    reducer: PCA | None = None
    output_center: np.ndarray | None = None
    output_scale: float = 1.0
    axis_center: np.ndarray | None = None
    axis_scale: float = 1.0


@dataclass
class OnlineUpdateState:
    root_path: Path
    config_path: Path | None
    static_dir: Path
    dataset_signature: tuple[tuple[str, float], ...]
    known_discovery_files: set[str]
    all_discoveries: list[dict]
    all_embedding_matrix: np.ndarray
    all_layout_embedding: np.ndarray
    saved_coordinates: list[dict]
    displayed_discovery_files: set[str]
    projection_model: OnlineProjectionModel
    layout_mode: str
    fit_count: int
    max_displayed: int
    projection_method: str
    projection_axes: tuple[int, int]
    filters_detected: bool
    highlight_schema: dict
    last_full_recompute_time: float


def _highlight_schema(config_path: Path | None, root_path: Path, discoveries: list[dict]) -> dict:
    highlight_context = load_highlight_export_context(config_path)
    return {
        **highlight_context.schema,
        "filters_detected": any(discovery.get("filters") for discovery in discoveries),
        "storage_key": f"{root_path}|{config_path.resolve() if config_path else ''}",
    }


def _projection_model(matrix: np.ndarray, method: str, axes: tuple[int, int]) -> OnlineProjectionModel:
    model = OnlineProjectionModel(method=method, axes=axes)

    if len(matrix) == 0:
        return model

    if method == "axis":
        axis_embedding = matrix[:, [axes[0], axes[1]]]
        axis_center, axis_scale = _normalize_projection_bounds(axis_embedding)
        model.axis_center = axis_center
        model.axis_scale = axis_scale
        return model

    if method == "pca" and len(matrix) >= 3:
        x_norm, input_mean, input_std = _normalize_embedding_matrix(matrix)
        n_components = min(2, x_norm.shape[0], x_norm.shape[1])
        reducer = PCA(n_components=n_components, random_state=0)
        projected = _ensure_2d_projection(reducer.fit_transform(x_norm))
        output_center, output_scale = _normalize_projection_bounds(projected)
        model.input_mean = input_mean
        model.input_std = input_std
        model.reducer = reducer
        model.output_center = output_center
        model.output_scale = output_scale
        return model

    return model


def _relative_discovery_file(root_path: Path, discovery: dict) -> str:
    return os.fspath(Path(discovery["discovery_file"]).resolve().relative_to(root_path)).replace(os.sep, "/")


def _project_incremental(discovery: dict, state: OnlineUpdateState) -> np.ndarray:
    embedding = np.asarray(discovery["embedding"], dtype=float)
    model = state.projection_model

    if model.method == "axis":
        point = embedding[[model.axes[0], model.axes[1]]]
        return (point - model.axis_center) / model.axis_scale

    if model.method == "pca" and model.reducer is not None:
        x_norm = (embedding - model.input_mean) / model.input_std
        projected = model.reducer.transform(x_norm.reshape(1, -1))
        projected = _ensure_2d_projection(projected)[0]
        return (projected - model.output_center) / model.output_scale

    if len(state.all_embedding_matrix) == 0:
        return np.zeros(2)

    distances = np.linalg.norm(state.all_embedding_matrix - embedding, axis=1)
    neighbors = min(5, len(distances))
    neighbor_indices = np.argsort(distances)[:neighbors]
    weights = 1.0 / (distances[neighbor_indices] + 1e-6)
    return np.average(state.all_layout_embedding[neighbor_indices], axis=0, weights=weights)


def _write_state(state: OnlineUpdateState) -> None:
    discoveries_path = state.static_dir / "discoveries.json"
    _write_highlight_schema(state.static_dir, state.highlight_schema)
    _write_layout_result(
        state.static_dir,
        discoveries_path,
        state.saved_coordinates,
        state.layout_mode,
        len(state.all_discoveries),
        state.fit_count,
        state.max_displayed,
        state.projection_method,
        state.projection_axes,
    )


def recompute_online_discoveries(config: ServerConfig, runtime_state: RuntimeState) -> OnlineUpdateState:
    root_path = config.discoveries.resolve()
    static_dir = config.static_dir
    static_dir.mkdir(parents=True, exist_ok=True)
    discoveries_path = static_dir / "discoveries.json"

    discoveries, dataset_signature = _scan_discoveries(root_path)
    highlight_schema = _highlight_schema(config.config_file, root_path, discoveries)
    _write_highlight_schema(static_dir, highlight_schema)

    if len(discoveries) == 0:
        _write_empty_layout(static_dir, discoveries_path)
        state = OnlineUpdateState(
            root_path=root_path,
            config_path=config.config_file,
            static_dir=static_dir,
            dataset_signature=dataset_signature,
            known_discovery_files=set(),
            all_discoveries=[],
            all_embedding_matrix=np.empty((0, 0)),
            all_layout_embedding=np.empty((0, 2)),
            saved_coordinates=[],
            displayed_discovery_files=set(),
            projection_model=OnlineProjectionModel(
                method=runtime_state.projection_method,
                axes=runtime_state.projection_axes,
            ),
            layout_mode="empty",
            fit_count=0,
            max_displayed=runtime_state.display_limit,
            projection_method=runtime_state.projection_method,
            projection_axes=runtime_state.projection_axes,
            filters_detected=highlight_schema["filters_detected"],
            highlight_schema=highlight_schema,
            last_full_recompute_time=time.monotonic(),
        )
        runtime_state.online_update_state = state
        runtime_state.last_recompute_time = state.last_full_recompute_time
        return state

    root_key = os.fspath(root_path)
    _purge_stale_caches(root_key, dataset_signature)
    layout_discoveries, layout_embedding, layout_mode, fit_count = _cached_layout(
        root_key,
        dataset_signature,
        discoveries,
        projection_method=runtime_state.projection_method,
        projection_axes=runtime_state.projection_axes,
    )
    export_last_frame(layout_discoveries)

    display_discoveries, display_embedding = _downsample_for_display(
        layout_discoveries,
        layout_embedding,
        runtime_state.display_limit,
    )
    saved_coordinates = _saved_coordinates(display_discoveries, display_embedding, root_path)
    _write_layout_result(
        static_dir,
        discoveries_path,
        saved_coordinates,
        layout_mode,
        len(layout_discoveries),
        fit_count,
        runtime_state.display_limit,
        runtime_state.projection_method,
        runtime_state.projection_axes,
    )

    valid_discoveries, matrix = _dataset_layout_input(root_key, dataset_signature, discoveries)
    state = OnlineUpdateState(
        root_path=root_path,
        config_path=config.config_file,
        static_dir=static_dir,
        dataset_signature=dataset_signature,
        known_discovery_files={_relative_discovery_file(root_path, discovery) for discovery in valid_discoveries},
        all_discoveries=list(valid_discoveries),
        all_embedding_matrix=np.array(matrix, copy=True),
        all_layout_embedding=np.array(layout_embedding, copy=True),
        saved_coordinates=saved_coordinates,
        displayed_discovery_files={_relative_discovery_file(root_path, discovery) for discovery in display_discoveries},
        projection_model=_projection_model(matrix, runtime_state.projection_method, runtime_state.projection_axes),
        layout_mode=layout_mode,
        fit_count=fit_count,
        max_displayed=runtime_state.display_limit,
        projection_method=runtime_state.projection_method,
        projection_axes=runtime_state.projection_axes,
        filters_detected=highlight_schema["filters_detected"],
        highlight_schema=highlight_schema,
        last_full_recompute_time=time.monotonic(),
    )
    runtime_state.online_update_state = state
    runtime_state.last_recompute_time = state.last_full_recompute_time
    return state


def append_online_discoveries(config: ServerConfig, runtime_state: RuntimeState) -> int:
    state = runtime_state.online_update_state
    if state is None:
        recompute_online_discoveries(config, runtime_state)
        return 0

    appended = 0
    discovered_filters = state.filters_detected

    for discovery_path in _iter_discovery_paths(state.root_path):
        cache_mtime = _cache_mtime(discovery_path.parent, discovery_path)
        if cache_mtime is None:
            continue

        relative_file = os.fspath(discovery_path.resolve().relative_to(state.root_path)).replace(os.sep, "/")
        if relative_file in state.known_discovery_files:
            continue

        discovery = process_discovery(discovery_path, cache_mtime=cache_mtime)
        if discovery is None:
            continue

        export_last_frame([discovery])
        point = _project_incremental(discovery, state)
        state.known_discovery_files.add(relative_file)
        state.all_discoveries.append(discovery)
        if state.all_embedding_matrix.size == 0:
            state.all_embedding_matrix = np.asarray(discovery["embedding"], dtype=float).reshape(1, -1)
        else:
            state.all_embedding_matrix = np.vstack([
                state.all_embedding_matrix,
                np.asarray(discovery["embedding"], dtype=float),
            ])
        if state.all_layout_embedding.size == 0:
            state.all_layout_embedding = np.asarray(point, dtype=float).reshape(1, 2)
        else:
            state.all_layout_embedding = np.vstack([
                state.all_layout_embedding,
                np.asarray(point, dtype=float),
            ])

        saved_point = _saved_coordinates([discovery], np.asarray(point, dtype=float).reshape(1, 2), state.root_path)[0]
        state.saved_coordinates.append(saved_point)
        state.displayed_discovery_files.add(relative_file)
        discovered_filters = discovered_filters or bool(discovery.get("filters"))
        appended += 1

    if appended == 0:
        return 0

    if discovered_filters != state.filters_detected:
        state.filters_detected = discovered_filters
        state.highlight_schema = {
            **state.highlight_schema,
            "filters_detected": discovered_filters,
        }

    _write_state(state)
    return appended


def online_update_loop(config: ServerConfig, runtime_state: RuntimeState) -> None:
    while True:
        try:
            with runtime_state.recompute_lock:
                state = runtime_state.online_update_state
                now = time.monotonic()
                if (
                    state is None
                    or now - state.last_full_recompute_time >= ONLINE_FULL_RECOMPUTE_INTERVAL_SECONDS
                ):
                    recompute_online_discoveries(config, runtime_state)
                else:
                    append_online_discoveries(config, runtime_state)
        except Exception as error:
            print(f"Online discovery update failed: {error}")

        time.sleep(ONLINE_POINT_UPDATE_INTERVAL_SECONDS)


def start_online_updates(config: ServerConfig, runtime_state: RuntimeState) -> threading.Thread:
    recompute_online_discoveries(config, runtime_state)
    thread = threading.Thread(
        target=online_update_loop,
        args=(config, runtime_state),
        daemon=True,
    )
    thread.start()
    return thread
