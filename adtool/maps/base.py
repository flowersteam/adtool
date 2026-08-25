"""Common payload transformation contract for exploration maps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator


Payload = dict[str, Any]


class BaseMap(Leaf, ABC):
    """A stateful transformation of an experiment payload.

    Unlike the legacy ``Map`` type, this contract does not imply that the
    transformation can sample a value.  Sampling belongs to the concrete
    behavior and parameter-map roles.
    """

    def __init__(self, premap_key: str = "input", postmap_key: str = "output") -> None:
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.postmap_key = postmap_key

    @abstractmethod
    def map(self, payload: Payload, override_existing: bool = True) -> Payload:
        """Return a transformed payload without mutating ``payload``."""
        raise NotImplementedError
