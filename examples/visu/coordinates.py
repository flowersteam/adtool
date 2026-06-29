from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from adtool.examples.visu.highlights import (
    empty_highlight_schema,
    load_highlight_export_context,
)


DEFAULT_MAX_RENDERED_DISCOVERIES = 500
VALID_VISUAL_SUFFIXES = (".mp4", ".png")
VALID_PROJECTION_METHODS = ("umap", "pca", "tsne", "axis")
DEFAULT_PROJECTION_METHOD = "umap"
DEFAULT_PROJECTION_AXES = (0, 1)
DISCOVERY_HIGHLIGHTS_FILENAME = "discovery_highlights.json"

Discovery = dict[str, Any]
DatasetSignature = tuple[tuple[str, float], ...]
_CACHE_MISS = object()
_discovery_cache: dict[str, tuple[float, Discovery | None]] = {}
_dataset_cache: dict[tuple[str, DatasetSignature], tuple[list[Discovery], np.ndarray]] = {}
_layout_cache: dict[
    tuple[str, DatasetSignature, str, tuple[int, int]],
    tuple[list[Discovery], np.ndarray, str, int],
] = {}
_cache_lock = threading.Lock()


def _cache_mtime(discovery_dir: Path, discovery_path: Path) -> float | None:
    try:
        return max(
            discovery_path.stat().st_mtime,
            discovery_dir.stat().st_mtime,
        )
    except OSError:
        return None


def _cached_discovery(discovery_path: Path, cache_mtime: float) -> object:
    cache_key = os.fspath(discovery_path)
    with _cache_lock:
        cached = _discovery_cache.get(cache_key)
    if cached is not None and cached[0] == cache_mtime:
        return cached[1]
    return _CACHE_MISS


def _store_cached_discovery(
    discovery_path: Path,
    cache_mtime: float,
    discovery: Discovery | None,
) -> None:
    cache_key = os.fspath(discovery_path)
    with _cache_lock:
        _discovery_cache[cache_key] = (cache_mtime, discovery)


def _valid_embedding(payload: dict[str, Any]) -> list[float] | None:
    if "output" not in payload:
        return None

    try:
        embedding = np.asarray(payload["output"], dtype=float)
    except (TypeError, ValueError):
        return None

    if embedding.ndim != 1 or embedding.size == 0:
        return None
    if np.isnan(embedding).any():
        print("nan found")
        return None
    if np.isinf(embedding).any():
        print("infinities found")
        return None

    return embedding.tolist()


def _visual_from_rendered_outputs(payload: dict[str, Any], discovery_dir: Path) -> Path | None:
    rendered_outputs = payload.get("rendered_outputs")
    if not isinstance(rendered_outputs, list):
        return None

    for rendered_output in rendered_outputs:
        if not isinstance(rendered_output, str):
            continue

        visual_path = discovery_dir / rendered_output
        if (
            visual_path.suffix.lower() in VALID_VISUAL_SUFFIXES
            and visual_path.is_file()
        ):
            return visual_path

    return None


def _first_visual_file(discovery_dir: Path) -> Path | None:
    try:
        files = list(discovery_dir.iterdir())
    except OSError:
        return None

    for suffix in VALID_VISUAL_SUFFIXES:
        for path in files:
            if path.is_file() and path.suffix.lower() == suffix:
                return path
    return None


def _resolve_visual_path(payload: dict[str, Any], discovery_dir: Path) -> Path | None:
    visual_path = _visual_from_rendered_outputs(payload, discovery_dir)
    if visual_path is not None:
        return visual_path
    return _first_visual_file(discovery_dir)


def process_discovery(
    discovery_path: str | os.PathLike[str],
    cache_mtime: float | None = None,
) -> Discovery | None:
    discovery_path = Path(discovery_path)
    discovery_dir = discovery_path.parent

    if cache_mtime is None:
        cache_mtime = _cache_mtime(discovery_dir, discovery_path)
    if cache_mtime is None:
        return None

    cached = _cached_discovery(discovery_path, cache_mtime)
    if cached is not _CACHE_MISS:
        return cached

    try:
        with discovery_path.open() as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        _store_cached_discovery(discovery_path, cache_mtime, None)
        return None
    if not isinstance(payload, dict):
        _store_cached_discovery(discovery_path, cache_mtime, None)
        return None

    embedding = _valid_embedding(payload)
    if embedding is None:
        _store_cached_discovery(discovery_path, cache_mtime, None)
        return None

    visual_path = _resolve_visual_path(payload, discovery_dir)
    if visual_path is None:
        _store_cached_discovery(discovery_path, cache_mtime, None)
        return None

    discovery = {
        "visual": os.fspath(visual_path),
        "embedding": embedding,
        "filters": payload.get("filters", {}),
        "discovery_file": os.fspath(discovery_path),
    }
    _store_cached_discovery(discovery_path, cache_mtime, discovery)
    return discovery


def _iter_discovery_paths(path: str | os.PathLike[str]) -> list[Path]:
    return sorted(Path(path).rglob("discovery.json"))


def _scan_discoveries(
    path: str | os.PathLike[str],
) -> tuple[list[Discovery], DatasetSignature]:
    discoveries: list[Discovery] = []
    root_path = Path(path)
    dataset_signature: list[tuple[str, float]] = []

    for discovery_path in _iter_discovery_paths(root_path):
        cache_mtime = _cache_mtime(discovery_path.parent, discovery_path)
        if cache_mtime is None:
            continue

        relative_path = os.fspath(discovery_path.relative_to(root_path)).replace(os.sep, "/")
        dataset_signature.append((relative_path, cache_mtime))

        result = process_discovery(discovery_path, cache_mtime=cache_mtime)
        if result:
            discoveries.append(result)

    discoveries.sort(key=lambda discovery: discovery["visual"])
    return discoveries, tuple(dataset_signature)


def list_discoveries(path: str | os.PathLike[str]) -> list[Discovery]:
    discoveries, _ = _scan_discoveries(path)

    print("Number of discoveries: ", len(discoveries))
    return discoveries


def _last_frame_needs_export(visual_path: Path, image_path: Path) -> bool:
    try:
        return not image_path.exists() or image_path.stat().st_mtime < visual_path.stat().st_mtime
    except OSError:
        return False


def _export_video_last_frame(visual_path: Path) -> None:
    image_path = visual_path.with_suffix(".jpg")
    if not _last_frame_needs_export(visual_path, image_path):
        return

    video = cv2.VideoCapture(os.fspath(visual_path))
    ret = False
    frame = None
    try:
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = video.read()
    finally:
        video.release()

    if ret:
        cv2.imwrite(os.fspath(image_path), frame)


def export_last_frame(discoveries: list[Discovery]) -> None:
    for discovery in discoveries:
        visual_path = Path(os.fspath(discovery["visual"]))
        if visual_path.exists() and visual_path.suffix.lower() == ".mp4":
            _export_video_last_frame(visual_path)


def _write_json_atomic(path: str | os.PathLike[str], payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle)
    tmp_path.replace(target)


def _write_layout_status(static_dir: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    status = {
        **payload,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json_atomic(Path(static_dir) / "layout_status.json", status)


def _write_highlight_schema(
    static_dir: str | os.PathLike[str],
    schema: dict[str, Any] | None,
) -> None:
    _write_json_atomic(
        Path(static_dir) / DISCOVERY_HIGHLIGHTS_FILENAME,
        schema or empty_highlight_schema(),
    )


def _write_empty_layout(static_dir: str | os.PathLike[str], discoveries_path: Path) -> None:
    _write_json_atomic(discoveries_path, [])
    _write_layout_status(
        static_dir,
        {
            "mode": "empty",
            "stable": False,
            "count": 0,
            "displayed_count": 0,
            "fit_count": 0,
        },
    )


def _valid_discoveries(discoveries: list[Discovery]) -> tuple[list[Discovery], np.ndarray]:
    filtered: list[Discovery] = []
    embeddings = []
    expected_dim = None

    for discovery in discoveries:
        if "embedding" not in discovery:
            continue

        embedding = np.asarray(discovery["embedding"], dtype=float)
        if embedding.ndim != 1 or embedding.size == 0:
            continue
        if expected_dim is None:
            expected_dim = embedding.size
        if embedding.size != expected_dim:
            continue
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            continue

        filtered.append(discovery)
        embeddings.append(embedding)

    if not embeddings:
        return [], np.empty((0, 0))

    return filtered, np.vstack(embeddings)


def _purge_stale_caches(root_key: str, dataset_signature: DatasetSignature) -> None:
    with _cache_lock:
        stale_dataset_keys = [
            cache_key
            for cache_key in _dataset_cache
            if cache_key[0] == root_key and cache_key[1] != dataset_signature
        ]
        stale_layout_keys = [
            cache_key
            for cache_key in _layout_cache
            if cache_key[0] == root_key and cache_key[1] != dataset_signature
        ]

        for cache_key in stale_dataset_keys:
            del _dataset_cache[cache_key]
        for cache_key in stale_layout_keys:
            del _layout_cache[cache_key]


def _dataset_layout_input(
    root_path: str | os.PathLike[str],
    dataset_signature: DatasetSignature,
    discoveries: list[Discovery],
) -> tuple[list[Discovery], np.ndarray]:
    cache_key = (os.fspath(root_path), dataset_signature)
    with _cache_lock:
        cached = _dataset_cache.get(cache_key)
    if cached is not None:
        return cached

    cached = _valid_discoveries(discoveries)
    with _cache_lock:
        _dataset_cache[cache_key] = cached
    return cached


def _cached_layout(
    root_path: str | os.PathLike[str],
    dataset_signature: DatasetSignature,
    discoveries: list[Discovery],
    projection_method: str,
    projection_axes: tuple[int, int],
) -> tuple[list[Discovery], np.ndarray, str, int]:
    root_key = os.fspath(root_path)
    cache_key = (
        root_key,
        dataset_signature,
        projection_method.lower(),
        projection_axes,
    )
    with _cache_lock:
        cached = _layout_cache.get(cache_key)
    if cached is not None:
        return cached

    dataset_discoveries, matrix = _dataset_layout_input(root_key, dataset_signature, discoveries)
    if len(dataset_discoveries) == 0:
        cached = (dataset_discoveries, np.empty((0, 2)), "empty", 0)
    else:
        embedding, layout_mode, fit_count = _project_layout(
            matrix,
            projection_method=projection_method,
            projection_axes=projection_axes,
        )
        cached = (dataset_discoveries, embedding, layout_mode, fit_count)

    with _cache_lock:
        _layout_cache[cache_key] = cached
    return cached


def _normalize_embedding_matrix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    return (x - mean) / std, mean, std


def _normalize_projection_bounds(embedding: np.ndarray) -> tuple[np.ndarray, float]:
    min_xy = embedding.min(axis=0)
    max_xy = embedding.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    scale = float(np.max(max_xy - min_xy))
    if scale <= 1e-9:
        scale = 1.0
    return center, scale


def _normalized_projection(embedding: np.ndarray) -> np.ndarray:
    embedding = _ensure_2d_projection(embedding)
    center, scale = _normalize_projection_bounds(embedding)
    return (embedding - center) / scale


def _ensure_2d_projection(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim == 0:
        embedding = embedding.reshape(1, 1)
    if embedding.ndim == 1:
        embedding = embedding.reshape(-1, 1)

    if embedding.shape[1] == 0:
        return np.zeros((embedding.shape[0], 2))
    if embedding.shape[1] == 1:
        return np.hstack([embedding, np.zeros((embedding.shape[0], 1))])
    return embedding[:, :2]


def _bootstrap_layout(x: np.ndarray) -> np.ndarray:
    if len(x) == 1:
        return np.array([[0.0, 0.0]])
    if len(x) == 2:
        return np.array([[-0.5, 0.0], [0.5, 0.0]])

    return _project_with_pca(x)


def _project_with_pca(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return _bootstrap_layout(x)

    x_norm, _, _ = _normalize_embedding_matrix(x)
    n_components = min(2, x_norm.shape[0], x_norm.shape[1])
    reducer = PCA(n_components=n_components, random_state=0)
    embedding = reducer.fit_transform(x_norm)
    return _normalized_projection(embedding)


def _project_with_umap(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return _bootstrap_layout(x)

    x_norm, _, _ = _normalize_embedding_matrix(x)
    reducer = umap.UMAP(
        n_components=2,
        random_state=0,
        n_neighbors=min(10, len(x_norm) - 1),
    )
    embedding = reducer.fit_transform(x_norm)
    return _normalized_projection(embedding)


def _project_with_tsne(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return _bootstrap_layout(x)

    x_norm, _, _ = _normalize_embedding_matrix(x)
    perplexity = min(30.0, max(1.0, (len(x_norm) - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        random_state=0,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
    )
    embedding = reducer.fit_transform(x_norm)
    return _normalized_projection(embedding)


def _project_with_axes(x: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    x_axis, y_axis = axes
    if x_axis == y_axis:
        raise ValueError("Axis ids must be different.")

    output_dim = x.shape[1] if x.ndim == 2 else 0
    if x_axis < 0 or y_axis < 0 or x_axis >= output_dim or y_axis >= output_dim:
        raise ValueError(
            f"Axis ids must be between 0 and {max(0, output_dim - 1)} for this output.",
        )

    embedding = x[:, [x_axis, y_axis]]
    return _normalized_projection(embedding)


def _project_layout(
    x: np.ndarray,
    projection_method: str = DEFAULT_PROJECTION_METHOD,
    projection_axes: tuple[int, int] = DEFAULT_PROJECTION_AXES,
) -> tuple[np.ndarray, str, int]:
    method = projection_method.lower()
    if method not in VALID_PROJECTION_METHODS:
        raise ValueError(f"Unknown projection method: {projection_method}")

    if method == "axis":
        return _project_with_axes(x, projection_axes), (
            f"axis_{projection_axes[0]}_{projection_axes[1]}"
        ), len(x)
    if method == "pca":
        return _project_with_pca(x), "pca", len(x)
    if method == "tsne":
        return _project_with_tsne(x), "tsne", len(x)
    if len(x) < 3:
        return _bootstrap_layout(x), "umap_bootstrap_pca", 0
    return _project_with_umap(x), "umap", len(x)


def _downsample_for_display(
    discoveries: list[Discovery],
    embedding: np.ndarray,
    max_displayed: int,
    root_path: str | os.PathLike[str] | None = None,
    selected_sources: set[str] | None = None,
) -> tuple[list[Discovery], np.ndarray]:
    if len(discoveries) <= max_displayed:
        return discoveries, embedding

    selected_sources = selected_sources or set()
    selected_indices = []
    if root_path is not None and selected_sources:
        for index, discovery in enumerate(discoveries):
            if _discovery_source_path(discovery, root_path) in selected_sources:
                selected_indices.append(index)

    if len(selected_indices) >= max_displayed:
        selected_indices.sort()
        return [discoveries[i] for i in selected_indices], embedding[selected_indices]

    selected_index_set = set(selected_indices)
    candidate_indices = [index for index in range(len(discoveries)) if index not in selected_index_set]
    if not candidate_indices:
        selected_indices.sort()
        return [discoveries[i] for i in selected_indices], embedding[selected_indices]

    remaining_slots = max_displayed - len(selected_indices)
    candidate_embedding = embedding[candidate_indices]

    kmeans = KMeans(n_clusters=remaining_slots, random_state=0)
    labels = kmeans.fit_predict(candidate_embedding)
    centers = kmeans.cluster_centers_

    sampled_indices = []
    for cluster_idx, center in enumerate(centers):
        members = np.where(labels == cluster_idx)[0]
        if len(members) == 0:
            continue

        cluster_points = candidate_embedding[members]
        nearest_member = members[np.argmin(np.linalg.norm(cluster_points - center, axis=1))]
        sampled_indices.append(candidate_indices[nearest_member])

    selected_indices.extend(sampled_indices)
    selected_indices.sort()
    return [discoveries[i] for i in selected_indices], embedding[selected_indices]


def _saved_coordinates(
    discoveries: list[Discovery],
    embedding: np.ndarray,
    root_path: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    root_path = os.fspath(root_path)
    saved_coordinates = []

    for discovery, point in zip(discoveries, embedding):
        if np.isnan(point).any() or np.isinf(point).any():
            continue

        visual_path = os.path.relpath(os.fspath(discovery["visual"]), root_path)
        saved_point = {
            "x": float(point[0]),
            "y": float(point[1]),
            "visual": visual_path.replace(os.sep, "/"),
            "filters": discovery.get("filters", {}),
        }

        saved_coordinates.append(saved_point)

    return saved_coordinates


def _discovery_source_path(
    discovery: Discovery,
    root_path: str | os.PathLike[str],
) -> str:
    relative_visual_path = os.path.relpath(os.fspath(discovery["visual"]), os.fspath(root_path))
    return f"/discoveries/{relative_visual_path.replace(os.sep, '/')}"


def _write_layout_result(
    static_dir: str | os.PathLike[str],
    discoveries_path: Path,
    saved_coordinates: list[dict[str, Any]],
    layout_mode: str,
    count: int,
    fit_count: int,
    max_displayed: int,
    projection_method: str,
    projection_axes: tuple[int, int],
) -> None:
    _write_json_atomic(discoveries_path, saved_coordinates)
    _write_layout_status(
        static_dir,
        {
            "mode": layout_mode,
            "stable": False,
            "count": count,
            "displayed_count": len(saved_coordinates),
            "fit_count": fit_count,
            "max_displayed": max_displayed,
            "projection_method": projection_method,
            "projection_axes": list(projection_axes),
        },
    )


def compute_coordinates(
    path: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    static_dir: str | os.PathLike[str] = "static",
    max_displayed: int = DEFAULT_MAX_RENDERED_DISCOVERIES,
    projection_method: str = DEFAULT_PROJECTION_METHOD,
    projection_axes: tuple[int, int] = DEFAULT_PROJECTION_AXES,
    selected_sources: set[str] | None = None,
) -> None:
    print("computing coordinates", path)
    root_path = Path(path).resolve()
    static_dir = Path(static_dir)
    static_dir.mkdir(parents=True, exist_ok=True)
    discoveries_path = static_dir / "discoveries.json"
    concatenated_path = static_dir / "concatenated.webm"
    highlight_context = load_highlight_export_context(config_path)

    discoveries, dataset_signature = _scan_discoveries(root_path)
    highlight_schema = {
        **highlight_context.schema,
        "filters_detected": any(discovery.get("filters") for discovery in discoveries),
        "storage_key": f"{root_path}|{Path(config_path).resolve() if config_path else ''}",
    }
    _write_highlight_schema(static_dir, highlight_schema)
    print("Number of discoveries: ", len(discoveries))
    if len(discoveries) == 0:
        _write_empty_layout(static_dir, discoveries_path)
        if concatenated_path.exists():
            concatenated_path.unlink()
        return

    root_key = os.fspath(root_path)
    _purge_stale_caches(root_key, dataset_signature)
    layout_discoveries, layout_embedding, layout_mode, fit_count = _cached_layout(
        root_key,
        dataset_signature,
        discoveries,
        projection_method=projection_method,
        projection_axes=projection_axes,
    )
    if len(layout_discoveries) == 0:
        _write_empty_layout(static_dir, discoveries_path)
        return

    export_last_frame(layout_discoveries)

    display_discoveries, display_embedding = _downsample_for_display(
        layout_discoveries,
        layout_embedding,
        max_displayed,
        root_path=root_path,
        selected_sources=selected_sources,
    )
    saved_coordinates = _saved_coordinates(
        display_discoveries,
        display_embedding,
        root_path,
    )

    _write_layout_result(
        static_dir,
        discoveries_path,
        saved_coordinates,
        layout_mode,
        len(layout_discoveries),
        fit_count,
        max_displayed,
        projection_method,
        projection_axes,
    )
