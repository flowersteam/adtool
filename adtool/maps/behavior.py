"""Behavior-space map contract."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from adtool.maps.base import BaseMap


class BehaviorMap(BaseMap):
    """Transform a system result and provide goals in the behavior space."""

    @abstractmethod
    def sample(self, **kwargs: Any) -> Any:
        """Return one behavior-space goal."""
        raise NotImplementedError
