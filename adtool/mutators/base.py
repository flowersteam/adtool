"""Base contract for IMGEP mutation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMutator(ABC):
    """An importable, checkpoint-safe strategy for mutating a policy.

    Mutators deliberately receive the parameter map when they are called
    instead of retaining it.  Parameter maps are leaves in the experiment
    tree, so retaining one would make a mutator serialize runtime parents.
    """

    @abstractmethod
    def __call__(self, parameters: Any, *, parameter_map: Any = None) -> Any:
        """Return a mutated version of ``parameters``."""
