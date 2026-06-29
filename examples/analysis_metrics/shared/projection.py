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


def apply_projection(config, datasets):
    projection = load_dotted_object(config.path)
    result = projection(datasets, config.config)
    values = [np.asarray(value, dtype=float) for value in result[:-1]]
    normalized = []
    for value in values:
        if value.ndim == 1:
            value = value.reshape(-1, 1)
        normalized.append(value)
    labels = None if len(result) < 2 else list(result[-1])
    return normalized, labels
