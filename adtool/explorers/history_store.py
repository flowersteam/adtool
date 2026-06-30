from __future__ import annotations

from copy import deepcopy
from typing import Dict, List


class HistoryStore:
    """
    Replay-first in-memory history store for explorers.

    The persisted source of truth is `discoveries/*/discovery.json`. This
    store only caches replayed and newly observed entries needed by the live
    explorer policy.
    """

    def __init__(self) -> None:
        self.buffer: List[Dict] = []

    def append(self, item: Dict) -> Dict:
        stored_item = deepcopy(item)
        self.buffer.append(stored_item)
        return stored_item

    def map(self, item: Dict) -> Dict:
        """
        Compatibility shim for legacy `SaveWrapper.map(...)` call sites.
        """
        self.append(item)
        return deepcopy(item)

    def get_history(self, lookback_length: int = -1) -> List[Dict]:
        if lookback_length == 0:
            return []
        if lookback_length < 0:
            return self.buffer
        if lookback_length >= len(self.buffer):
            return self.buffer
        return self.buffer[-lookback_length:]

    def last(self) -> Dict:
        return self.buffer[-1]
