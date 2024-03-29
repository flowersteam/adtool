
from copy import deepcopy
from typing import Dict

import torch
from adtool.maps.Map import Map
from adtool.wrappers.BoxProjector import BoxProjector
from adtool.utils.leaf.locators.locators import BlobLocator

torch.set_default_dtype(torch.float32)


class IdentityBehaviorMap(Map):

    def __init__(
        self,
        premap_key: str = "output",
        postmap_key: str = "output",
    ) -> None:
        super().__init__()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.locator = BlobLocator()

        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        return self.projector.map(input)

    def sample(self) -> torch.Tensor:
        return self.projector.sample()