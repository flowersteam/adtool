from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


MIN_STABLE_UMAP_DISCOVERIES = 10
DEFAULT_MAX_RENDERED_DISCOVERIES = 500
VALID_VISUAL_SUFFIXES = (".mp4", ".png")

Discovery = dict[str, Any]

_discovery_cache: dict[str, dict[str, Any]] = {}
_cache_lock = threading.Lock()


def _cache_mtime(discovery_dir: Path, discovery_path: Path) -> float | None:
    try:
        return max(discovery_path.stat().st_mtime, discovery_dir.stat().st_mtime)
    except OSError:
        return None


def _cached_discovery(discovery_path: Path, cache_mtime: float) -> Discovery | None:
    cache_key = os.fspath(discovery_path)
    with _cache_lock:
        cached = _discovery_cache.get(cache_key)
        if isinstance(cached, dict) and cached.get("mtime") == cache_mtime:
            return cached["discovery"]
    return None


def _store_cached_discovery(
    discovery_path: Path,
    cache_mtime: float,
    discovery: Discovery,
) -> None:
    cache_key = os.fspath(discovery_path)
    with _cache_lock:
        _discovery_cache[cache_key] = {
            "mtime": cache_mtime,
            "discovery": discovery,
        }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None

    return payload if isinstance(payload, dict) else None


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


def process_discovery(root: str | os.PathLike[str], name: str) -> Discovery | None:
    discovery_dir = Path(root) / name
    discovery_path = discovery_dir / "discovery.json"
    if not discovery_path.exists():
        return None

    cache_mtime = _cache_mtime(discovery_dir, discovery_path)
    if cache_mtime is None:
        return None

    cached = _cached_discovery(discovery_path, cache_mtime)
    if cached is not None:
        return cached

    payload = _load_json(discovery_path)
    if payload is None:
        return None

    embedding = _valid_embedding(payload)
    if embedding is None:
        return None

    visual_path = _first_visual_file(discovery_dir)
    if visual_path is None:
        return None

    discovery = {
        "visual": os.fspath(visual_path),
        "embedding": embedding,
    }
    _store_cached_discovery(discovery_path, cache_mtime, discovery)
    return discovery


def list_discoveries(path: str | os.PathLike[str]) -> list[Discovery]:
    tasks = []
    discoveries: list[Discovery] = []

    with ThreadPoolExecutor() as executor:
        for root, dirs, _ in os.walk(path):
            for name in dirs:
                tasks.append(executor.submit(process_discovery, root, name))

        for future in as_completed(tasks):
            result = future.result()
            if result:
                discoveries.append(result)

    print("Number of discoveries: ", len(discoveries))
    return sorted(discoveries, key=lambda discovery: discovery["visual"])


def export_last_frame(discoveries: list[Discovery]) -> None:
    for discovery in discoveries:
        visual_path = Path(os.fspath(discovery["visual"]))
        if not visual_path.exists() or visual_path.suffix.lower() != ".mp4":
            continue

        image_path = visual_path.with_suffix(".jpg")
        try:
            if image_path.exists() and image_path.stat().st_mtime >= visual_path.stat().st_mtime:
                continue
        except OSError:
            continue

        video = cv2.VideoCapture(os.fspath(visual_path))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            video.release()
            continue

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = video.read()
        video.release()
        if ret:
            cv2.imwrite(os.fspath(image_path), frame)


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


def _write_empty_layout(static_dir: str | os.PathLike[str], discoveries_path: Path) -> None:
    _write_json_atomic(discoveries_path, [])
    _write_layout_status(static_dir, {
        "mode": "empty",
        "stable": False,
        "count": 0,
        "displayed_count": 0,
        "fit_count": 0,
    })


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


def _project_with_temporary_layout(x: np.ndarray) -> np.ndarray:
    if len(x) == 1:
        return np.array([[0.0, 0.0]])
    if len(x) == 2:
        return np.array([[-0.5, 0.0], [0.5, 0.0]])

    x_norm, _, _ = _normalize_embedding_matrix(x)
    reducer = PCA(n_components=2, random_state=0)
    embedding = reducer.fit_transform(x_norm)
    center, scale = _normalize_projection_bounds(embedding)
    return (embedding - center) / scale


def _project_with_umap(x: np.ndarray) -> np.ndarray:
    x_norm, _, _ = _normalize_embedding_matrix(x)
    reducer = umap.UMAP(
        n_components=2,
        random_state=0,
        n_neighbors=min(10, len(x_norm) - 1),
    )
    embedding = reducer.fit_transform(x_norm)
    center, scale = _normalize_projection_bounds(embedding)
    return (embedding - center) / scale


def _downsample_for_display(
    discoveries: list[Discovery],
    embedding: np.ndarray,
    max_displayed: int,
) -> tuple[list[Discovery], np.ndarray]:
    if len(discoveries) <= max_displayed:
        return discoveries, embedding

    kmeans = KMeans(n_clusters=max_displayed, random_state=0)
    labels = kmeans.fit_predict(embedding)
    centers = kmeans.cluster_centers_

    selected_indices = []
    for cluster_idx, center in enumerate(centers):
        members = np.where(labels == cluster_idx)[0]
        if len(members) == 0:
            continue

        cluster_points = embedding[members]
        nearest_member = members[np.argmin(np.linalg.norm(cluster_points - center, axis=1))]
        selected_indices.append(nearest_member)

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
        saved_coordinates.append({
            "x": float(point[0]),
            "y": float(point[1]),
            "visual": visual_path.replace(os.sep, "/"),
        })

    return saved_coordinates


def compute_coordinates(
    path: str | os.PathLike[str],
    static_dir: str | os.PathLike[str] = "static",
    max_displayed: int = DEFAULT_MAX_RENDERED_DISCOVERIES,
) -> None:
    print("computing coordinates", path)
    static_dir = Path(static_dir)
    discoveries_path = static_dir / "discoveries.json"
    concatenated_path = static_dir / "concatenated.webm"

    discoveries = list_discoveries(path)
    if len(discoveries) == 0:
        _write_empty_layout(static_dir, discoveries_path)
        if concatenated_path.exists():
            concatenated_path.unlink()
        return

    discoveries, x = _valid_discoveries(discoveries)
    if len(discoveries) == 0:
        _write_empty_layout(static_dir, discoveries_path)
        return

    if len(discoveries) >= MIN_STABLE_UMAP_DISCOVERIES:
        embedding = _project_with_umap(x)
        layout_mode = "refit_umap"
        fit_count = len(discoveries)
    else:
        embedding = _project_with_temporary_layout(x)
        layout_mode = "bootstrap_pca"
        fit_count = 0

    export_last_frame(discoveries)

    display_discoveries, display_embedding = _downsample_for_display(
        discoveries,
        embedding,
        max_displayed,
    )
    saved_coordinates = _saved_coordinates(display_discoveries, display_embedding, path)

    _write_json_atomic(discoveries_path, saved_coordinates)
    _write_layout_status(static_dir, {
        "mode": layout_mode,
        "stable": False,
        "count": len(discoveries),
        "displayed_count": len(saved_coordinates),
        "fit_count": fit_count,
        "max_displayed": max_displayed,
    })
