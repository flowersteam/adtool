from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict

from adtool.utils.leaf.Leaf import Leaf


class BaseParameterMap(Leaf, ABC):
    """Base parameter map with sample/mutate hooks."""

    def __init__(self, premap_key: str = "params") -> None:
        super().__init__()
        self.premap_key = premap_key

    @abstractmethod
    def sample(self) -> Any:
        ...

    @abstractmethod
    def mutate(self, parameter_dict: Any) -> Any:
        ...

    def map(self, input: Dict[str, Any], override_existing: bool = True) -> Dict[str, Any]:
        intermed = deepcopy(input)
        if (override_existing and self.premap_key in intermed) or (
            self.premap_key not in intermed
        ):
            intermed[self.premap_key] = self.sample()
        return intermed
