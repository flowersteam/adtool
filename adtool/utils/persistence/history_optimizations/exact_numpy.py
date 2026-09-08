"""Exact disk-backed NumPy index for unlimited-lookback history queries."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from .base import HistoryOptimization


@dataclass(frozen=True)
class _FeatureChunk:
    history_path: Path
    feature_path: Path
    indices_path: Path
    dimension: int
    lower: np.ndarray
    upper: np.ndarray
    position_offset: int
    record_count: int


@dataclass(frozen=True)
class _Candidate:
    distance: float
    position: int
    feature: np.ndarray
    history_path: Path | None
    record_index: int


class ExactNumpyHistoryOptimization(HistoryOptimization):
    """Exact nearest-neighbour search over checkpoint-sidecar feature arrays.

    Raw discoveries stay in the normal history pickle files.  Only flattened,
    finite feature vectors are written to ``.npy`` files and scanned during a
    query.  The complete pickle is loaded only for final winners.
    """

    VERSION = 1

    def __init__(self) -> None:
        self._history = None
        self._chunks_by_dimension: dict[int, list[_FeatureChunk]] = {}
        self._first_dimension: int | None = None
        self._persisted_count = 0
        self._pending: list[tuple[int, np.ndarray]] = []

    @property
    def implementation_path(self) -> str:
        """Return the import path of this concrete optimization class."""
        implementation = type(self)
        return f"{implementation.__module__}.{implementation.__qualname__}"

    def attach(self, history) -> None:
        self._history = history
        self._pending = []
        for index, record in enumerate(history._pending):
            feature = self._feature_for(record)
            if feature is not None:
                self._pending.append((index, feature))

    def record(self, discovery: dict[str, Any]) -> None:
        if self._history is None:
            raise RuntimeError("History optimization is not attached")
        feature = self._feature_for(discovery)
        if feature is not None:
            self._pending.append((len(self._history._pending) - 1, feature))

    def set_head(self, checkpoint_dir: Path) -> None:
        if self._history is None:
            raise RuntimeError("History optimization is not attached")

        self._chunks_by_dimension = {}
        self._first_dimension = None
        self._persisted_count = 0
        self._pending = []

        chain = self._checkpoint_chain(Path(checkpoint_dir).resolve())
        for checkpoint in chain:
            manifest = self._read_manifest(checkpoint)
            metadata = manifest.get("history_optimization")
            if (
                not isinstance(metadata, dict)
                or metadata.get("implementation_path") != self.implementation_path
            ):
                raise ValueError(
                    "Checkpoint {} has no history index for {}. "
                    "This optimization supports newly indexed checkpoints only; "
                    "start a fresh run or remove explorer.config.history_optimization."
                    .format(checkpoint.name, self.implementation_path)
                )
            if metadata.get("version") != self.VERSION:
                raise ValueError(
                    f"Checkpoint {checkpoint.name} has unsupported history index version"
                )
            if metadata.get("feature_key") != self._history.feature_key:
                raise ValueError(
                    f"Checkpoint {checkpoint.name} indexes feature key "
                    f"{metadata.get('feature_key')!r}, expected {self._history.feature_key!r}"
                )

            entries = metadata.get("history_files")
            if not isinstance(entries, dict):
                raise ValueError(
                    f"Checkpoint {checkpoint.name} has invalid history index metadata"
                )
            counts = manifest.get("history_file_counts", {})
            for history_filename in manifest.get("history_files", []):
                count = counts.get(history_filename)
                if type(count) is not int or count < 0:
                    raise ValueError(
                        f"Checkpoint {checkpoint.name} has invalid count for {history_filename}"
                    )
                entry = entries.get(history_filename)
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Checkpoint {checkpoint.name} has no index for {history_filename}"
                    )
                first_dimension = entry.get("first_dimension")
                if self._first_dimension is None and isinstance(first_dimension, int):
                    self._first_dimension = first_dimension
                dimensions = entry.get("dimensions", {})
                if not isinstance(dimensions, dict):
                    raise ValueError(
                        f"Checkpoint {checkpoint.name} has invalid feature dimensions"
                    )
                for raw_dimension, group in dimensions.items():
                    if not isinstance(group, dict):
                        raise ValueError(
                            f"Checkpoint {checkpoint.name} has invalid feature group"
                        )
                    try:
                        dimension = int(raw_dimension)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"Checkpoint {checkpoint.name} has invalid feature dimension"
                        ) from error
                    feature_path = checkpoint / str(group.get("features_file", ""))
                    indices_path = checkpoint / str(group.get("indices_file", ""))
                    if not feature_path.is_file() or not indices_path.is_file():
                        raise ValueError(
                            f"Checkpoint {checkpoint.name} has missing feature sidecar files"
                        )
                    lower = np.asarray(group.get("lower"), dtype=float).reshape(-1)
                    upper = np.asarray(group.get("upper"), dtype=float).reshape(-1)
                    if lower.size != dimension or upper.size != dimension:
                        raise ValueError(
                            f"Checkpoint {checkpoint.name} has invalid feature bounds"
                        )
                    self._chunks_by_dimension.setdefault(dimension, []).append(
                        _FeatureChunk(
                            history_path=checkpoint / history_filename,
                            feature_path=feature_path,
                            indices_path=indices_path,
                            dimension=dimension,
                            lower=lower,
                            upper=upper,
                            position_offset=self._persisted_count,
                            record_count=count,
                        )
                    )
                self._persisted_count += count

    def write_checkpoint(
        self,
        directory: str | Path,
        *,
        checkpoint_name: str,
        history_files: list[str],
    ) -> dict[str, Any]:
        directory = Path(directory)
        entries: dict[str, Any] = {}
        if len(history_files) > 1:
            raise ValueError("ExactNumpyHistoryOptimization expects one history file per checkpoint")
        if history_files:
            grouped: dict[int, list[tuple[int, np.ndarray]]] = {}
            for record_index, feature in self._pending:
                grouped.setdefault(feature.size, []).append((record_index, feature))

            dimensions: dict[str, Any] = {}
            for dimension, rows in grouped.items():
                matrix = np.vstack([feature for _, feature in rows])
                indices = np.asarray([index for index, _ in rows], dtype=np.int64)
                feature_filename = f"history-features-{checkpoint_name}-d{dimension}.npy"
                indices_filename = f"history-feature-indices-{checkpoint_name}-d{dimension}.npy"
                np.save(directory / feature_filename, matrix)
                np.save(directory / indices_filename, indices)
                dimensions[str(dimension)] = {
                    "features_file": feature_filename,
                    "indices_file": indices_filename,
                    "count": int(matrix.shape[0]),
                    "lower": matrix.min(axis=0).tolist(),
                    "upper": matrix.max(axis=0).tolist(),
                }
            entries[history_files[0]] = {
                "first_dimension": self._pending[0][1].size if self._pending else None,
                "dimensions": dimensions,
            }

        return {
            "implementation_path": self.implementation_path,
            "version": self.VERSION,
            "feature_key": self._history.feature_key if self._history is not None else "",
            "history_files": entries,
        }

    def feature_bounds(
        self, history_lookback_length: int = -1
    ) -> tuple[np.ndarray, np.ndarray] | None:
        start = self._lookback_start(history_lookback_length)
        if start is None:
            return None

        dimension = self._first_dimension_for(start)
        if dimension is None:
            return None

        lower: np.ndarray | None = None
        upper: np.ndarray | None = None
        for chunk in self._chunks_by_dimension.get(dimension, []):
            if start >= chunk.position_offset + chunk.record_count:
                continue
            if start <= chunk.position_offset:
                chunk_lower, chunk_upper = chunk.lower, chunk.upper
            else:
                features = np.load(chunk.feature_path, mmap_mode="r")
                indices = np.load(chunk.indices_path, mmap_mode="r")
                self._validate_sidecar(features, indices, chunk)
                selected = features[indices + chunk.position_offset >= start]
                if selected.size == 0:
                    continue
                chunk_lower = selected.min(axis=0)
                chunk_upper = selected.max(axis=0)
            lower = (
                np.asarray(chunk_lower, dtype=float).copy()
                if lower is None
                else np.minimum(lower, chunk_lower)
            )
            upper = (
                np.asarray(chunk_upper, dtype=float).copy()
                if upper is None
                else np.maximum(upper, chunk_upper)
            )
        for record_index, feature in self._pending:
            if (
                feature.size == dimension
                and self._persisted_count + record_index >= start
            ):
                lower = feature.copy() if lower is None else np.minimum(lower, feature)
                upper = feature.copy() if upper is None else np.maximum(upper, feature)
        if lower is None or upper is None:
            return None
        return lower, upper

    def nearest(
        self,
        goal: np.ndarray,
        *,
        k: int,
        history_lookback_length: int,
        normalized: bool,
        normalization_bounds: tuple[np.ndarray, np.ndarray] | None,
    ) -> list[Any]:
        if self._history is None or k <= 0:
            return []
        start = self._lookback_start(history_lookback_length)
        if start is None:
            return []
        goal = np.asarray(goal, dtype=float).reshape(-1)
        scale = self._normalization_scale(
            goal, normalized, normalization_bounds, history_lookback_length
        )
        if normalized and scale is None:
            return []

        candidates: list[_Candidate] = []
        for chunk in self._chunks_by_dimension.get(goal.size, []):
            if start >= chunk.position_offset + chunk.record_count:
                continue
            features = np.load(chunk.feature_path, mmap_mode="r")
            indices = np.load(chunk.indices_path, mmap_mode="r")
            self._validate_sidecar(features, indices, chunk)
            positions = indices + chunk.position_offset
            selected = np.flatnonzero(positions >= start)
            if selected.size == 0:
                continue
            distances = self._distances(features[selected], goal, scale)
            for offset in self._top_k_indices(distances, positions[selected], k):
                index = int(selected[offset])
                candidates.append(
                    _Candidate(
                        distance=float(distances[offset]),
                        position=int(indices[index] + chunk.position_offset),
                        feature=np.asarray(features[index], dtype=float).copy(),
                        history_path=chunk.history_path,
                        record_index=int(indices[index]),
                    )
                )

        for index, feature in self._pending:
            if feature.size != goal.size or self._persisted_count + index < start:
                continue
            distance = float(self._distances(feature.reshape(1, -1), goal, scale)[0])
            candidates.append(
                _Candidate(
                    distance=distance,
                    position=self._persisted_count + index,
                    feature=feature.copy(),
                    history_path=None,
                    record_index=index,
                )
            )

        candidates.sort(key=lambda candidate: (candidate.distance, candidate.position))
        candidates = candidates[:k]
        records = self._winner_records(candidates)
        from adtool.utils.persistence.history import HistoryMatch

        return [
            HistoryMatch(
                record=deepcopy(record),
                feature=candidate.feature.copy(),
                payload=deepcopy(record[self._history.payload_key]),
                distance=candidate.distance,
                position=candidate.position - start,
            )
            for candidate, record in zip(candidates, records)
        ]

    def _winner_records(self, candidates: list[_Candidate]) -> list[dict[str, Any]]:
        loaded: dict[Path, list[dict[str, Any]]] = {}
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate.history_path is None:
                records.append(self._history._pending[candidate.record_index])
                continue
            if candidate.history_path not in loaded:
                with candidate.history_path.open("rb") as file:
                    chunk = pickle.load(file)
                if not isinstance(chunk, list):
                    raise ValueError(f"Invalid history chunk: {candidate.history_path}")
                loaded[candidate.history_path] = chunk
            records.append(loaded[candidate.history_path][candidate.record_index])
        return records

    def _normalization_scale(
        self,
        goal: np.ndarray,
        normalized: bool,
        normalization_bounds: tuple[np.ndarray, np.ndarray] | None,
        history_lookback_length: int,
    ) -> np.ndarray | None:
        if not normalized:
            return None
        if normalization_bounds is None:
            bounds = self.feature_bounds(history_lookback_length)
            if bounds is None:
                return None
            lower, upper = bounds
        else:
            lower = np.asarray(normalization_bounds[0], dtype=float).reshape(-1)
            upper = np.asarray(normalization_bounds[1], dtype=float).reshape(-1)
        if (
            lower.shape != upper.shape
            or lower.shape != goal.shape
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
        ):
            raise ValueError(
                "normalization_bounds must be finite lower/upper vectors "
                "with the same shape as goal"
            )
        scale = upper - lower
        scale[scale == 0] = 1.0
        return scale

    def _lookback_start(self, history_lookback_length: int) -> int | None:
        """Return the first global record position included by a lookback."""
        if self._history is None or history_lookback_length == 0:
            return None
        total = self._persisted_count + len(self._history._pending)
        if total == 0:
            return None
        if history_lookback_length < 0:
            return 0
        return max(0, total - int(history_lookback_length))

    def _first_dimension_for(self, start: int) -> int | None:
        """Find the first valid feature dimension in the selected history tail."""
        if start == 0 and self._first_dimension is not None:
            return self._first_dimension

        first_position: int | None = None
        first_dimension: int | None = None
        for dimension, chunks in self._chunks_by_dimension.items():
            for chunk in chunks:
                if start >= chunk.position_offset + chunk.record_count:
                    continue
                indices = np.load(chunk.indices_path, mmap_mode="r")
                if indices.ndim != 1:
                    raise ValueError(f"Invalid feature sidecar for {chunk.history_path}")
                selected = indices[indices + chunk.position_offset >= start]
                if selected.size == 0:
                    continue
                position = int(selected[0] + chunk.position_offset)
                if first_position is None or position < first_position:
                    first_position = position
                    first_dimension = dimension
        for index, feature in self._pending:
            position = self._persisted_count + index
            if position >= start and (first_position is None or position < first_position):
                first_position = position
                first_dimension = feature.size
        return first_dimension

    @staticmethod
    def _distances(features: np.ndarray, goal: np.ndarray, scale: np.ndarray | None) -> np.ndarray:
        difference = goal - features
        if scale is not None:
            difference = difference / scale
        return np.einsum("ij,ij->i", difference, difference)

    @staticmethod
    def _top_k_indices(distances: np.ndarray, positions: np.ndarray, k: int) -> np.ndarray:
        if distances.size <= k:
            return np.lexsort((positions, distances))
        selected = np.argpartition(distances, k - 1)[:k]
        threshold = distances[selected].max()
        strictly_better = np.flatnonzero(distances < threshold)
        remaining = k - strictly_better.size
        tied = np.flatnonzero(distances == threshold)
        tied = tied[np.argsort(positions[tied], kind="stable")[:remaining]]
        selected = np.concatenate((strictly_better, tied))
        return selected[np.lexsort((positions[selected], distances[selected]))]

    @staticmethod
    def _validate_sidecar(features: np.ndarray, indices: np.ndarray, chunk: _FeatureChunk) -> None:
        if (
            features.ndim != 2
            or features.shape[1] != chunk.dimension
            or indices.ndim != 1
            or indices.size != features.shape[0]
        ):
            raise ValueError(f"Invalid feature sidecar for {chunk.history_path}")

    def _feature_for(self, record: dict[str, Any]) -> np.ndarray | None:
        if self._history is None:
            return None
        if (
            self._history.feature_key not in record
            or self._history.payload_key not in record
        ):
            return None
        try:
            feature = np.asarray(record[self._history.feature_key], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if feature.size == 0 or not np.all(np.isfinite(feature)):
            return None
        return feature

    @staticmethod
    def _read_manifest(checkpoint: Path) -> dict[str, Any]:
        with (checkpoint / "manifest.json").open() as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid checkpoint manifest: {checkpoint}")
        return manifest

    def _checkpoint_chain(self, checkpoint: Path) -> list[Path]:
        chain: list[Path] = []
        current: Path | None = checkpoint
        seen: set[Path] = set()
        while current is not None:
            if current in seen:
                raise ValueError(f"Checkpoint parent cycle detected at {current}")
            seen.add(current)
            manifest = self._read_manifest(current)
            chain.append(current)
            parent = manifest.get("parent_checkpoint")
            current = Path(parent) if parent else None
        return list(reversed(chain))
