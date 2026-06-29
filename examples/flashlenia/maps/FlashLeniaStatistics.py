from typing import Dict
import numpy as np
from copy import deepcopy
from adtool.utils.leaf.Leaf import Leaf
from adtool.wrappers.BoxProjector import BoxProjector
from examples.flashlenia.systems.FlashLenia import FlashLenia

class FlashLeniaStatistics(Leaf):
    def __init__(
        self,
        system: FlashLenia,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()
        self.premap_key = premap_key
        self.postmap_key = postmap_key

        # Projector for behavior space
        self.projector = BoxProjector(
            premap_key=self.postmap_key,
            bound_lower=np.array([0, 0]),  # Assuming non-negative entropy values
            bound_upper=np.array([10, 10]),  # Adjust these bounds as needed
            init_low=np.array([0, 0]),
            init_high=np.array([10, 10])
        )

    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)
        print(intermed_dict,self.premap_key)
        output = intermed_dict[self.premap_key]
        
        stats = np.array([
            output["mean_entropy"],
            output["variance_entropy"]
        ])
        
        intermed_dict[self.postmap_key] = stats
        intermed_dict = self.projector.map(intermed_dict)

        return intermed_dict

    def sample(self):
        return self.projector.sample()