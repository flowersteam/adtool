"""Parameter-space map contract."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from adtool.maps.base import BaseMap


class ParameterMap(BaseMap):
    """Create parameter payloads for an explorer.

    Mutation is deliberately not part of this base contract: a configured
    mutator determines whether and how an individual parameter map mutates a
    parent policy.
    """

    @abstractmethod
    def sample(self) -> Any:
        """Return one parameter payload."""
        raise NotImplementedError
