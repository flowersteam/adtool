from dataclasses import dataclass, field

import numpy as np

from adtool.examples.analysis_metrics.shared.imports import load_dotted_object


@dataclass(frozen=True)
class ProjectionConfig:
    path: str
    config: dict = field(default_factory=dict)


def load_projection_config(section):
    projection = section["projection"]
    return ProjectionConfig(
        path=projection["path"],
        config=dict(projection.get("config") or {}),
    )


def apply_projection(config, dataset_a, dataset_b):
    projection = load_dotted_object(config.path)
    result = projection(dataset_a, dataset_b, config.config)
    values_a = np.asarray(result[0], dtype=float)
    values_b = np.asarray(result[1], dtype=float)
    if values_a.ndim == 1:
        values_a = values_a.reshape(-1, 1)
    if values_b.ndim == 1:
        values_b = values_b.reshape(-1, 1)
    labels = None if len(result) < 3 else list(result[2])
    return values_a, values_b, labels
