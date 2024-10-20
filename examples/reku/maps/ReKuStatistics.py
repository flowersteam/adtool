from typing import Dict
import numpy as np
from copy import deepcopy
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from adtool.wrappers.BoxProjector import BoxProjector
from examples.reku.systems.ReKu import ReKu

class ReKuStatistics(Leaf):
    """
    Compute statistics on ReKu's output.
    """

    def __init__(
        self,
        system: ReKu,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()
        self.sync_pop=system.sync_pop
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.postmap_key = postmap_key

        # projector for behavior space
        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: Dict) -> Dict:
        """
        Compute statistics on ReKu's output.
        Args:
            input: ReKu's output
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
        """
        Calculate static statistics from the array.
        Args:
            array: A numpy array containing the phases of the oscillators.
        Returns:
            A numpy array with calculated statistics.
        """

        # order_parameter = np.abs(np.mean(np.exp(1j * array[-50:, :]   ), axis=1))
        # fft = np.fft.fft(order_parameter)
        # fft = np.abs(fft)
        # fft = fft / np.sum(fft)


        # sample 50 points in array uniformly from beginning to end
        # sample = array[]
        # sample=array

        # keep only the last 10 timeseries
        sample = array[len(array)//2::len(array)//200, -self.sync_pop:]
        print(sample.shape)


        order_parameter = np.abs(np.mean(np.exp(1j * sample), axis=1))
        fft = np.fft.fft(order_parameter)
        fft = np.abs(fft)
        fft = fft / np.max(fft)
        # keep only the first half of the fft
        fft = fft[:len(fft)//2]


        # find the max difference between the phases
        diff = np.diff(array[len(array)//2:, -self.sync_pop:], axis=0)
        print("diff.shape", diff.shape)
        diff = np.abs(diff)
        diff = np.sum(diff, axis=1)
        print("diff.shape", diff.shape)
        max_diff = np.max(diff)
        print("max_diff", max_diff)










       # print("fft", fft.shape)



        return np.concatenate([fft, [
         1/(1+np.sqrt(   max_diff))
            
            ]])
