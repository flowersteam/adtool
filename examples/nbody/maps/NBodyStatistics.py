import typing
from copy import deepcopy

import numpy as np
from addict import Dict
from adtool.wrappers.BoxProjector import BoxProjector

from adtool.utils.leaf.Leaf import Leaf
from examples.nbody.systems.NBody import NBodySimulation

DT = 0.001  # Time step for the simulation (global constant)
SAVE_INTERVAL = 10  # Save interval for the simulation (global constant)

STABILITY_WEIGHT = 10

class NBodyStatistics(Leaf):
    def __init__(
        self,
        system: NBodySimulation,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()

        self.max_steps=system.max_steps

        self.max_distance_limit= system.max_distance_limit

        self.premap_key = premap_key
        self.postmap_key = postmap_key


        self._n_latents = 1 + system.pattern_size // 2  # timestep + FFT magnitudes up to 1 Hz

        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: typing.Dict) -> typing.Dict:
        intermed_dict = deepcopy(input)

        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = intermed_dict[self.premap_key]
        del intermed_dict[self.premap_key]

        embedding = self._calc_static_statistics(intermed_dict[raw_output_key])

        intermed_dict[self.postmap_key] = embedding
        intermed_dict = self.projector.map(intermed_dict)


        return intermed_dict

    def sample(self):
        sampled = self.projector.sample()
        # put first element to 1
        sampled[0] = STABILITY_WEIGHT
        return sampled
    
    def _calc_static_statistics(self, output: Dict) -> np.ndarray:
        positions_over_time = output["positions"]
        distances_over_time = output["distances"]
        timestep = output["timestep"]

        norm_over_time = np.linalg.norm(distances_over_time, axis=1)/self.max_distance_limit
        
        freqs, magnitudes = self._plot_fft_up_to_1hz(norm_over_time)
        
        embedding = np.concatenate(([STABILITY_WEIGHT* timestep / self.max_steps
                                     ], magnitudes))
        
        return embedding

    def _plot_fft_up_to_1hz(self, distance_series):
        N = len(distance_series)
        pattern_size = 4000  # As specified in the config

        if N < pattern_size:
            distance_series = np.interp(np.linspace(0, N, pattern_size), np.arange(N), distance_series)
            d = SAVE_INTERVAL * DT  * N / pattern_size  # 10 is the save_interval
        #    print("d", d)
        else:
            d = SAVE_INTERVAL * DT  # 10 is the save_interval

     #   print("distance_series: ", distance_series.shape)

        distance_fft = np.fft.fft(distance_series)
        freqs = np.fft.fftfreq(pattern_size, d=d)

        magnitude = np.abs(distance_fft) / pattern_size

        mask = (freqs >= 0) & (freqs <= 100)
        freqs_filtered = freqs[mask]
        magnitude_filtered = magnitude[mask]

        magnitude_filtered = np.interp(np.linspace(0, 100, 20), freqs_filtered, magnitude_filtered)


     #   print("magnitude_filtered: ", magnitude_filtered.shape)

        return freqs_filtered, magnitude_filtered