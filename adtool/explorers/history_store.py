from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np

"""History storage helpers shared by explorers.

This module provides a small in-memory store with two complementary roles:

1. keep the latest full discovery payload for persistence and replay flows
2. keep a compact numeric retrieval cache for fast nearest-neighbor queries

The retrieval cache is intentionally generic: callers provide an extractor that
defines a *retrieval view* of each stored discovery. That retrieval view is a
pair made of:

- a flat numeric feature vector used for nearest-neighbor lookup
- an aligned payload returned when a neighbor is selected

This is distinct from the algorithm's behavior-space projection. In most
explorers, the discovery item already contains the projected behavior under
`output`; the retrieval extractor simply selects which stored fields should be
indexed and returned by the cache.
"""


class HistoryStore:
    """
    Replay-first in-memory history store for explorers.

    The persisted source of truth is `discoveries/*/discovery.json`. This
    store keeps the last full discovery and, optionally, a compact retrieval
    cache made of feature vectors aligned with payloads.
    """

    def __init__(self, retain_full_history: bool = True) -> None:
        """Initialize the store.

        Args:
            retain_full_history:
                Whether to keep every full discovery item in memory. When
                disabled, the store still keeps the latest full discovery and
                the compact retrieval cache.
        """
        self.retain_full_history = retain_full_history
        self.buffer: List[Dict] = []
        self._last_item: Dict | None = None

        self._retrieval_extractor: Callable[[Dict], tuple[np.ndarray, Any] | None] | None = None
        self._retrieval_dimension: int | None = None
        self._retrieval_count = 0
        self._retrieval_capacity = 0
        self._retrieval_matrix = np.empty((0, 0), dtype=float)
        self._retrieval_payloads: List[Any] = []

    def record(self, item: Dict) -> Dict:
        """Store one discovery item.

        The returned value is a fresh deep copy so downstream code can mutate
        it without corrupting the stored history.
        """
        stored_item = deepcopy(item)
        self._last_item = stored_item
        if self.retain_full_history:
            self.buffer.append(stored_item)
        self._append_retrieval(stored_item)
        return deepcopy(stored_item)

    def get_history(self, lookback_length: int = -1) -> List[Dict]:
        """Return full discovery items from memory.

        Args:
            lookback_length:
            Number of most recent items to expose. A negative value means
                "all retained history".

        Raises:
            RuntimeError:
                If full-history retention was disabled for this store.
        """
        if not self.retain_full_history:
            raise RuntimeError(
                "HistoryStore full-history retention is disabled for this instance. "
                "Use the retrieval cache instead.",
            )
        if lookback_length == 0:
            return []
        if lookback_length < 0 or lookback_length >= len(self.buffer):
            return self.buffer
        return self.buffer[-lookback_length:]

    def last(self) -> Dict:
        """Return the most recent full discovery item."""
        if self._last_item is None:
            raise IndexError("HistoryStore is empty")
        return self._last_item

    def configure_retrieval_view(
        self,
        extractor: Callable[[Dict], tuple[np.ndarray, Any] | None],
    ) -> None:
        """Configure the compact retrieval cache.

        Args:
            extractor:
                Function turning one stored discovery item into the retrieval
                view `(feature, payload)`.

                This does not necessarily perform the algorithmic projection
                from raw system output to behavior space. In the current
                explorers that projection has usually already happened before
                the item reaches history. The extractor instead defines which
                fields of the stored discovery should populate the retrieval
                cache. Example: A, B, C => (A, C)

                Returning `None` skips the item from retrieval while still
                allowing it to exist as a full discovery.

        Notes:
            Reconfiguring the extractor resets the retrieval cache and, when
            full history is retained, rebuilds it from the stored items.
        """
        self._retrieval_extractor = extractor
        self._retrieval_dimension = None
        self._retrieval_count = 0
        self._retrieval_capacity = 0
        self._retrieval_matrix = np.empty((0, 0), dtype=float)
        self._retrieval_payloads = []

        if self.retain_full_history:
            for item in self.buffer:
                self._append_retrieval(item)

    def get_retrieval_view(
        self,
        lookback_length: int = -1,
    ) -> tuple[np.ndarray, List[Any]]:
        """Return retrieval features and payloads for the requested window."""
        return (
            self.get_retrieval_features(lookback_length),
            self.get_retrieval_payloads(lookback_length),
        )

    def get_retrieval_features(self, lookback_length: int = -1) -> np.ndarray:
        """Return the cached retrieval feature matrix for the lookback window."""
        if self._retrieval_count == 0:
            return np.zeros((0, 0), dtype=float)
        if lookback_length == 0:
            return np.zeros((0, self._retrieval_dimension or 0), dtype=float)

        start = self._retrieval_start(lookback_length)
        return self._retrieval_matrix[start : self._retrieval_count]

    def get_retrieval_payloads(self, lookback_length: int = -1) -> List[Any]:
        """Return payloads aligned with the cached retrieval feature rows."""
        if self._retrieval_count == 0 or lookback_length == 0:
            return []

        start = self._retrieval_start(lookback_length)
        return self._retrieval_payloads[start : self._retrieval_count]

    def find_nearest_payloads(
        self,
        goal: np.ndarray,
        *,
        k: int,
        lookback_length: int = -1,
        distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> List[Any]:
        """Return the payloads of the `k` nearest cached retrieval vectors."""
        indices = self.find_nearest_indices(
            goal,
            k=k,
            lookback_length=lookback_length,
            distance_fn=distance_fn,
        )
        if indices.size == 0:
            return []

        start = self._retrieval_start(lookback_length)
        return [self._retrieval_payloads[start + int(index)] for index in indices]

    def find_nearest_indices(
        self,
        goal: np.ndarray,
        *,
        k: int,
        lookback_length: int = -1,
        distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return nearest retrieval row indices for a goal vector.

        The storage/query path is shared, but the caller can supply a custom
        distance function to preserve explorer-specific selection semantics.
        """
        features = self.get_retrieval_features(lookback_length)
        if features.shape[0] == 0 or k <= 0:
            return np.zeros(0, dtype=int)

        goal = np.asarray(goal, dtype=float).reshape(-1)
        if goal.shape[0] != features.shape[1]:
            raise ValueError(
                "Goal dimension does not match retrieval feature dimension: "
                f"{goal.shape[0]} != {features.shape[1]}",
            )

        distances = self._compute_distances(goal, features, distance_fn)
        k_eff = min(int(k), len(distances))
        if k_eff == 1:
            return np.array([int(np.argmin(distances))], dtype=int)

        candidate_indices = np.argpartition(distances, k_eff - 1)[:k_eff]
        order = np.argsort(distances[candidate_indices], kind="stable")
        return candidate_indices[order]

    def _append_retrieval(self, item: Dict) -> None:
        """Try to append one item to the compact retrieval cache.

        Invalid items are skipped silently:
        - no retrieval extractor configured
        - extractor returns `None`
        - empty, non-finite, or dimension-mismatched feature vectors
        """
        if self._retrieval_extractor is None:
            return

        retrieval_entry = self._retrieval_extractor(item)
        if retrieval_entry is None:
            return

        vector_raw, payload = retrieval_entry
        vector = np.asarray(vector_raw, dtype=float).reshape(-1)
        if vector.size == 0:
            return
        if np.isnan(vector).any() or np.isinf(vector).any():
            return

        if self._retrieval_dimension is None:
            self._retrieval_dimension = int(vector.size)
            self._retrieval_capacity = 16
            self._retrieval_matrix = np.empty(
                (self._retrieval_capacity, self._retrieval_dimension),
                dtype=float,
            )
        elif vector.size != self._retrieval_dimension:
            return

        if self._retrieval_count >= self._retrieval_capacity:
            new_capacity = max(1, self._retrieval_capacity * 2)
            grown = np.empty((new_capacity, self._retrieval_dimension), dtype=float)
            if self._retrieval_count > 0:
                grown[: self._retrieval_count] = self._retrieval_matrix[: self._retrieval_count]
            self._retrieval_matrix = grown
            self._retrieval_capacity = new_capacity

        self._retrieval_matrix[self._retrieval_count] = vector
        self._retrieval_payloads.append(deepcopy(payload))
        self._retrieval_count += 1

    def _retrieval_start(self, lookback_length: int) -> int:
        """Compute the starting row for a retrieval lookback window."""
        if lookback_length < 0 or lookback_length >= self._retrieval_count:
            return 0
        return self._retrieval_count - lookback_length

    @staticmethod
    def _compute_distances(
        goal: np.ndarray,
        features: np.ndarray,
        distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    ) -> np.ndarray:
        if distance_fn is None:
            return np.sum((goal.reshape(1, -1) - features) ** 2, axis=1)

        distances = np.asarray(distance_fn(goal, features), dtype=float).reshape(-1)
        if distances.shape[0] != features.shape[0]:
            raise ValueError(
                "Custom distance function must return one value per feature row: "
                f"{distances.shape[0]} != {features.shape[0]}",
            )
        if np.isnan(distances).any() or np.isinf(distances).any():
            raise ValueError("Custom distance function returned non-finite values")
        return distances
