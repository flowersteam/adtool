from abc import ABC, abstractmethod

import numpy as np

from adtool.examples.analysis_metrics.shared import load_dotted_object


class SpaceCoverageMetric(ABC):
    title = "Space coverage"
    y_label = "covered cells"

    @abstractmethod
    def compute_progression(self, points: np.ndarray):
        pass


def load_space_coverage_metric(config):
    metric_cls = load_dotted_object(config.path)
    return metric_cls(**dict(config.config or {}))
