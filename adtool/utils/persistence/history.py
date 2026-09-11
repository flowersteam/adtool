"""Chunked, file-backed exploration history.

``HistoryStore`` is deliberately independent from :class:`~adtool.utils.leaf.Leaf`.
It owns explorer discoveries, while checkpoints own the files containing completed
history chunks.  This keeps explorer state small and makes history traversal
explicit when an experiment branches from an older checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import dataclass
import heapq
import json
import pickle
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class HistoryMatch:
    """A result selected from history."""

    record: dict[str, Any]
    feature: np.ndarray
    payload: Any
    distance: float
    position: int


@dataclass(frozen=True)
class _ChunkRef:
    """A persisted chunk and its manifest-recorded discovery count."""

    path: Path
    count: int


class HistoryStore:
    """Central history API used by explorers.

    Newly discovered records are buffered in memory until the next checkpoint,
    which writes that checkpoint's batch onto disk. A
    separate rolling cache keeps the newest ``cache_size`` discoveries
    decoded in memory. A size of ``0`` disables the cache, while ``-1`` keeps
    complete history.
    """

    def __init__(
        self,
        feature_key: str = "output",
        payload_key: str = "params",
        cache_size: int = 100,
    ) -> None:
        if cache_size < -1:
            raise ValueError("history cache_size must be >= -1")
        self.feature_key = feature_key
        self.payload_key = payload_key
        self._cache_size = int(cache_size)
        self._head: Path | None = None
        self._pending: list[dict[str, Any]] = []
        self._checkpoint_refs: list[_ChunkRef] | None = []
        self._recent_cache: deque[dict[str, Any]] = deque(
            maxlen=None if self._cache_size == -1 else self._cache_size
        )
        self._last: dict[str, Any] | None = None
        self._logger = None

    def set_logger(self, logger) -> None:
        """Attach the process-local logger used for history diagnostics."""
        self._logger = logger

    def _debug(self, event: str, **fields: Any) -> None:
        """Emit one compact history diagnostic when DEBUG logging is enabled."""
        if self._logger is None:
            return
        details = ", ".join(f"{name}={value}" for name, value in fields.items())
        message = f"[HISTORY] {event}"
        self._logger.debug(f"{message}: {details}" if details else message)

    @property
    def cache_size(self) -> int:
        """RAM cache capacity: ``0`` disables it and ``-1`` retains all history."""
        return self._cache_size

    @cache_size.setter
    def cache_size(self, value: int) -> None:
        previous_size = self._cache_size
        value = int(value)
        if value < -1:
            raise ValueError("history cache_size must be >= -1")
        previous = list(getattr(self, "_recent_cache", ()))
        self._cache_size = value
        maxlen = None if value == -1 else value
        self._recent_cache = deque(
            previous if value == -1 else previous[-value:], maxlen=maxlen
        )
        if self._head is not None and (
            value == -1 or len(self._recent_cache) < value
        ):
            self._warm_recent_cache()
        self._debug(
            "cache configured",
            previous=previous_size,
            capacity=value,
            cached=len(self._recent_cache),
        )

    @property
    def head(self) -> Path | None:
        return self._head

    @property
    def has_pending_records(self) -> bool:
        return bool(self._pending)

    def set_head(self, checkpoint_dir: str | Path | None) -> None:
        """Attach this store to a completed checkpoint chain."""
        self._head = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        self._pending = []
        self._checkpoint_refs = None if self._head is not None else []
        self._last = None
        self._recent_cache.clear()
        if self._head is not None:
            self._checkpoint_chunk_refs()
            self._warm_recent_cache()
        self._debug(
            "checkpoint head selected",
            checkpoint=self._head.name if self._head is not None else "none",
            chunks=len(self._checkpoint_refs or []),
            cached=len(self._recent_cache),
        )

    def record(self, discovery: dict[str, Any]) -> dict[str, Any]:
        """Record one complete discovery and return an isolated working copy."""
        stored = deepcopy(discovery)
        pending_before = len(self._pending)
        cache_before = len(self._recent_cache)
        evicted = self._cache_size > 0 and cache_before >= self._cache_size
        self._pending.append(stored)
        # The cache and persistence buffer are both private, read-only owners
        # of this history entry.  Sharing this isolated copy avoids duplicating
        # every discovery in RAM while external callers still receive copies.
        self._recent_cache.append(stored)
        self._last = stored
        self._debug(
            "discovery recorded",
            pending=f"{pending_before}->{len(self._pending)}",
            cache=f"{len(self._recent_cache)}/{self._cache_size}",
            evicted_oldest=evicted,
        )
        return deepcopy(stored)

    def last(self) -> dict[str, Any]:
        """Return the latest complete discovery without retaining all history."""
        if self._last is not None:
            return deepcopy(self._last)

        if self._recent_cache:
            self._last = deepcopy(self._recent_cache[-1])
            return deepcopy(self._last)
        for source in reversed(self._sources()):
            records = self._source_records(source)
            if records:
                self._last = records[-1]
                return deepcopy(self._last)
        raise IndexError("HistoryStore is empty")

    def iter_chunks(
        self, history_lookback_length: int = -1
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield safe copied chunks for external consumers.

        Built-in explorers do not use this API; it is kept for custom export
        and inspection code. A positive lookback applies to discoveries.
        """
        for chunk in self._iter_selected_chunks(history_lookback_length):
            if chunk:
                yield deepcopy(chunk)

    def iter_history(
        self, history_lookback_length: int = -1
    ) -> Iterator[dict[str, Any]]:
        """Yield safe copied discoveries for external consumers.

        Built-in explorers use the private no-copy path; this API is retained
        for custom export and inspection code.
        """
        for chunk in self.iter_chunks(history_lookback_length):
            for record in chunk:
                yield record

    def features(self, history_lookback_length: int = -1) -> np.ndarray:
        """Materialize numeric features for callers that explicitly require it.

        Explorers should generally prefer :meth:`feature_bounds`,
        :meth:`nearest`, or :meth:`random`, which keep memory bounded by a
        history chunk.  This compatibility helper intentionally remains
        available for external algorithms whose API requires a matrix.
        """
        features: list[np.ndarray] = []
        for _, feature, _ in self._iter_retrieval_records(history_lookback_length):
            features.append(feature)
        if not features:
            return np.zeros((0, 0), dtype=float)
        return np.vstack(features)

    def feature_bounds(
        self, history_lookback_length: int = -1
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return numeric feature bounds using one streaming pass."""
        lower, upper = self._feature_bounds(history_lookback_length)
        if lower is None or upper is None:
            self._debug("feature bounds", features=0)
            return None
        self._debug("feature bounds", dimensions=lower.size)
        return lower, upper

    def nearest(
        self,
        goal: np.ndarray,
        *,
        k: int = 1,
        history_lookback_length: int = -1,
        normalized: bool = False,
        normalization_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[HistoryMatch]:
        """Return exact nearest matches by scanning history chunks.

        When normalized distances are required, callers that already sampled a
        goal from :meth:`feature_bounds` can provide those same bounds.  This
        avoids a redundant full-history pass before the nearest-neighbour scan.
        """
        if k <= 0:
            return []
        goal = np.asarray(goal, dtype=float).reshape(-1)
        self._debug(
            "nearest search started",
            neighbors=k,
            dimensions=goal.size,
            normalized=normalized,
        )

        lower: np.ndarray | None = None
        scale: np.ndarray | None = None
        if normalized:
            if normalization_bounds is None:
                self._debug("normalization bounds", source="history scan")
                lower, upper = self._feature_bounds(history_lookback_length)
            else:
                self._debug("normalization bounds", source="provided by caller")
                lower, upper = (
                    np.asarray(normalization_bounds[0], dtype=float).reshape(-1),
                    np.asarray(normalization_bounds[1], dtype=float).reshape(-1),
                )
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
            if lower is None or upper is None:
                self._debug("nearest search finished", matches=0, reason="no valid features")
                return []
            scale = upper - lower
            scale[scale == 0] = 1.0

        winners: list[tuple[float, int, dict[str, Any], np.ndarray]] = []
        for position, feature, record in self._iter_retrieval_records(
            history_lookback_length
        ):
            if feature.shape != goal.shape:
                continue
            difference = goal - feature
            if scale is not None:
                difference = difference / scale
            distance = float(np.dot(difference, difference))
            item = (-distance, -position, record, feature)
            if len(winners) < k:
                heapq.heappush(winners, item)
            elif item[:2] > winners[0][:2]:
                heapq.heapreplace(winners, item)

        matches: list[HistoryMatch] = []
        for negative_distance, negative_position, record, feature in sorted(
            winners, key=lambda item: (-item[0], -item[1])
        ):
            matches.append(
                HistoryMatch(
                    record=deepcopy(record),
                    feature=feature.copy(),
                    payload=deepcopy(record[self.payload_key]),
                    distance=-negative_distance,
                    position=-negative_position,
                )
            )
        self._debug(
            "nearest search finished",
            matches=len(matches),
            positions=",".join(str(match.position) for match in matches) or "none",
            distances=(
                ",".join(f"{match.distance:.4g}" for match in matches) or "none"
            ),
        )
        return matches

    def random(self, history_lookback_length: int = -1) -> HistoryMatch | None:
        """Choose one valid history item with reservoir sampling."""
        choice: tuple[int, np.ndarray, dict[str, Any]] | None = None
        count = 0
        for position, feature, record in self._iter_retrieval_records(
            history_lookback_length
        ):
            count += 1
            if np.random.randint(count) == 0:
                choice = (position, feature, record)
        if choice is None:
            self._debug("random selection", candidates=0, position="none")
            return None
        position, feature, record = choice
        match = HistoryMatch(
            record=deepcopy(record),
            feature=feature.copy(),
            payload=deepcopy(record[self.payload_key]),
            distance=0.0,
            position=position,
        )
        self._debug("random selection", candidates=count, position=position)
        return match

    def write_checkpoint_history(
        self, directory: str | Path, *, checkpoint_name: str
    ) -> tuple[list[str], dict[str, int]]:
        """Write the pending batch directly into a temporary checkpoint directory."""
        if not self._pending:
            self._debug("checkpoint batch skipped", pending=0)
            return [], {}
        filename = f"history-{checkpoint_name}.pickle"
        self._debug(
            "checkpoint batch write",
            discoveries=len(self._pending),
            file=filename,
        )
        with (Path(directory) / filename).open("wb") as file:
            pickle.dump(self._pending, file, protocol=pickle.HIGHEST_PROTOCOL)
        self._debug(
            "checkpoint batch written",
            discoveries=len(self._pending),
            file=filename,
        )
        return [filename], {filename: len(self._pending)}

    def commit_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        history_files: list[str],
        history_file_counts: dict[str, int],
    ) -> None:
        """Mark pending records as owned by a successfully written checkpoint."""
        checkpoint_path = Path(checkpoint_dir).resolve()
        new_refs: list[_ChunkRef] = []
        for filename in history_files:
            count = history_file_counts.get(filename)
            if type(count) is not int or count < 0:
                raise ValueError(
                    f"Checkpoint history file {filename!r} has no valid discovery count"
                )
            new_refs.append(_ChunkRef(checkpoint_path / filename, count))
        cached_refs = self._checkpoint_chunk_refs()
        self._head = checkpoint_path
        self._pending = []
        cached_refs.extend(new_refs)
        self._debug(
            "checkpoint committed",
            checkpoint=checkpoint_path.name,
            discoveries=sum(history_file_counts.values()),
            chunks=len(cached_refs),
            cached=len(self._recent_cache),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        feature_key: str,
        payload_key: str,
        cache_size: int,
        logger=None,
    ) -> "HistoryStore":
        store = cls(
            feature_key=feature_key,
            payload_key=payload_key,
            cache_size=cache_size,
        )
        store.set_logger(logger)
        store.set_head(checkpoint_dir)
        return store

    def _checkpoint_chunk_refs(self) -> list[_ChunkRef]:
        """Return cached chronological refs for the selected checkpoint chain."""
        if self._checkpoint_refs is None:
            self._checkpoint_refs = self._read_checkpoint_chunk_refs()
        else:
            self._debug("chunk index reused", chunks=len(self._checkpoint_refs))
        return self._checkpoint_refs

    def _read_checkpoint_chunk_refs(self) -> list[_ChunkRef]:
        """Read checkpoint manifests once to build chronological chunk refs."""
        checkpoints: list[tuple[Path, dict[str, Any]]] = []
        current = self._head
        while current is not None:
            manifest_path = current / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Checkpoint manifest does not exist: {manifest_path}")
            with manifest_path.open() as file:
                manifest = json.load(file)
            checkpoints.append((current, manifest))
            parent = manifest.get("parent_checkpoint")
            current = Path(parent) if parent else None

        # The parent chain was collected newest-to-oldest.  Reverse only the
        # checkpoints, preserving each checkpoint's own chunk order.
        refs: list[_ChunkRef] = []
        for checkpoint, manifest in reversed(checkpoints):
            counts = manifest.get("history_file_counts")
            for name in manifest.get("history_files", []):
                count = counts.get(name)
                refs.append(
                    _ChunkRef(
                        checkpoint / name,
                        count,
                    )
                )
        self._debug(
            "chunk index built",
            checkpoints=len(checkpoints),
            chunks=len(refs),
            discoveries=sum(ref.count for ref in refs),
        )
        return refs

    def _sources(self) -> list[tuple[_ChunkRef | None, list[dict[str, Any]] | None]]:
        """Return chronological persisted, then pending history sources."""
        sources: list[tuple[_ChunkRef | None, list[dict[str, Any]] | None]] = [
            (ref, None) for ref in self._checkpoint_chunk_refs()
        ]
        if self._pending:
            sources.append((None, self._pending))
        return sources

    def _source_count(
        self, source: tuple[_ChunkRef | None, list[dict[str, Any]] | None]
    ) -> int:
        ref, records = source
        if records is not None:
            return len(records)
        if ref is None:
            return 0
        return ref.count

    def _source_records(
        self, source: tuple[_ChunkRef | None, list[dict[str, Any]] | None]
    ) -> list[dict[str, Any]]:
        ref, records = source
        if records is not None:
            self._debug("pending discoveries used", discoveries=len(records))
            return records
        if ref is None:
            return []
        self._debug(
            "checkpoint chunk loaded",
            chunk=f"{ref.path.parent.name}/{ref.path.name}",
            discoveries=ref.count,
        )
        return self._load_chunk(ref.path, reverse=False)

    def _iter_selected_chunks(
        self, history_lookback_length: int
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield a chronological history tail, reading only records outside cache."""
        if history_lookback_length == 0:
            self._debug("retrieval skipped", reason="lookback is zero")
            return
        if self._cache_size == -1:
            # Complete-buffer mode is intentionally independent from the
            # checkpoint chain after initialization: do not even read its
            # manifests during normal retrieval.
            cached = list(self._recent_cache)
            if history_lookback_length > 0:
                cached = cached[-int(history_lookback_length):]
            if cached:
                self._debug(
                    "retrieval from complete cache",
                    discoveries=len(cached),
                )
                yield cached
            return
        if 0 < history_lookback_length <= len(self._recent_cache):
            cached = list(self._recent_cache)[-int(history_lookback_length):]
            self._debug("retrieval from cache", discoveries=len(cached))
            yield cached
            return

        sources = self._sources()
        counts = [self._source_count(source) for source in sources]
        total = sum(counts)
        if total == 0:
            self._debug("retrieval skipped", reason="history is empty")
            return

        limit = (
            total
            if history_lookback_length < 0
            else min(int(history_lookback_length), total)
        )
        cache_count = min(len(self._recent_cache), total)
        requested_start = total - limit
        disk_end = total - cache_count
        self._debug(
            "retrieval planned",
            available=total,
            requested=limit,
            from_cache=cache_count,
            from_chunks=max(0, limit - cache_count),
            chunks=len(self._checkpoint_refs or []),
            pending=len(self._pending),
        )
        offset = 0
        for source, count in zip(sources, counts):
            source_end = offset + count
            if source_end > requested_start and offset < disk_end:
                records = self._source_records(source)
                start = max(0, requested_start - offset)
                end = min(len(records), disk_end - offset)
                if start < end:
                    yield records[start:end]
            if source_end >= disk_end:
                break
            offset = source_end

        cache_start = max(0, requested_start - disk_end)
        cached = list(self._recent_cache)[cache_start:]
        if cached:
            self._debug("retrieval cache tail", discoveries=len(cached))
            yield cached

    def _warm_recent_cache(self) -> None:
        """Fill the recent window from newest persisted history sources."""
        if self._cache_size == 0:
            self._recent_cache.clear()
            self._debug("cache warm skipped", capacity=0)
            return
        self._debug(
            "cache warm started",
            capacity=self._cache_size,
            chunks=len(self._checkpoint_refs or []),
        )
        newest_first: list[dict[str, Any]] = []
        remaining: int | None = None if self._cache_size == -1 else self._cache_size
        for source in reversed(self._sources()):
            if remaining is not None and remaining <= 0:
                break
            records = self._source_records(source)
            take = records if remaining is None else records[-remaining:]
            newest_first.extend(reversed(take))
            if remaining is not None:
                remaining -= len(take)
        self._recent_cache = deque(
            reversed(newest_first),
            maxlen=None if self._cache_size == -1 else self._cache_size,
        )
        if self._recent_cache:
            self._last = deepcopy(self._recent_cache[-1])
        self._debug(
            "cache warm finished",
            cached=len(self._recent_cache),
            capacity=self._cache_size,
        )

    @staticmethod
    def _load_chunk(file_path: Path, reverse: bool) -> list[dict[str, Any]]:
        with file_path.open("rb") as file:
            chunk = pickle.load(file)
        if not isinstance(chunk, list):
            raise ValueError(f"History chunk is not a list: {file_path}")
        return list(reversed(chunk)) if reverse else chunk

    def _iter_retrieval_records(
        self, history_lookback_length: int
    ) -> Iterator[tuple[int, np.ndarray, dict[str, Any]]]:
        """Iterate private history records without defensive copying.

        This path is used only by HistoryStore's own read-only algorithms.
        Public ``iter_chunks`` and ``iter_history`` continue to copy their
        output, so callers cannot mutate cached or persisted history.
        """
        position = 0
        for chunk in self._iter_selected_chunks(history_lookback_length):
            for record in chunk:
                feature = self._feature_for(record)
                if feature is not None:
                    yield position, feature, record
                position += 1

    def _feature_bounds(
        self, history_lookback_length: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        lower: np.ndarray | None = None
        upper: np.ndarray | None = None
        for _, feature, _ in self._iter_retrieval_records(history_lookback_length):
            if lower is None:
                lower = feature.copy()
                upper = feature.copy()
            elif feature.shape == lower.shape:
                lower = np.minimum(lower, feature)
                upper = np.maximum(upper, feature)
        return lower, upper

    def _feature_for(self, record: dict[str, Any]) -> np.ndarray | None:
        if self.feature_key not in record or self.payload_key not in record:
            return None
        try:
            feature = np.asarray(record[self.feature_key], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if feature.size == 0 or not np.all(np.isfinite(feature)):
            return None
        return feature