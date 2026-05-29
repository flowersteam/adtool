from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict

from adtool.maps.Map import Map


class BaseParameterMap(Map, ABC):
    """Base parameter map with sample/mutate hooks."""

    def __init__(self, premap_key: str = "params", postmap_key: str = "params") -> None:
        super().__init__(premap_key=premap_key, postmap_key=postmap_key)
        self.premap_key = premap_key
        self.postmap_key = postmap_key

    @abstractmethod
    def sample(self) -> Any:
        ...

    @abstractmethod
    def mutate(self, parameter_dict: Any) -> Any:
        """Mutate one parent payload or a nearest-first list of parent payloads."""
        ...

    def map(self, input: Dict[str, Any], override_existing: bool = True) -> Dict[str, Any]:
        intermed = deepcopy(input)
        if (override_existing and self.postmap_key in intermed) or (
            self.postmap_key not in intermed
        ):
            intermed[self.postmap_key] = self.sample()
        return intermed
