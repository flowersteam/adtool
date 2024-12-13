
from typing import Dict
import numpy as np
from copy import deepcopy
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from adtool.wrappers.BoxProjector import BoxProjector
from examples.synth.systems.Synth import Synth

class SynthStatistics(Leaf):
    """
    Compute statistics on Synth's output.
    """

    def __init__(
        self,
        system: Synth,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.postmap_key = postmap_key

        # projector for behavior space
        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: Dict) -> Dict:
        """
        Compute statistics on Synth's output.
        Args:
            input: Synth's output
        Returns:
            A dictionary with the computed statistics.
        """

        intermed_dict = deepcopy(input)

        # store raw output
        array = np.array(intermed_dict[self.premap_key])
        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = array
        del intermed_dict[self.premap_key]

        embedding = self._calc_static_statistics(array)

        intermed_dict[self.postmap_key] = embedding
        intermed_dict = self.projector.map(intermed_dict)

        return intermed_dict

    def sample(self):
        return self.projector.sample()
    

    def _calc_static_statistics(self, array: np.ndarray) -> np.ndarray:


        sample = array[::len(array)//200]


        fft = np.fft.fft(sample)
        fft = fft[:len(fft)//2]
        fft = np.abs(fft)
        fft = fft / np.max(fft)

        return fft
