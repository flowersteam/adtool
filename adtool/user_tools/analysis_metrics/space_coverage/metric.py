from abc import ABC, abstractmethod

import numpy as np

from adtool.utils.factory import instantiate_object


class SpaceCoverageMetric(ABC):
    title = "Space coverage"
    y_label = "covered cells"

    @abstractmethod
    def compute_progression(self, points: np.ndarray):
        pass


def load_space_coverage_metric(config):
    return instantiate_object(config, object_name="space coverage metric")
