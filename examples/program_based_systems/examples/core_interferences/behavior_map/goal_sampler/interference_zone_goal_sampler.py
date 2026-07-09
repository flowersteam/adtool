from __future__ import annotations

from typing import Any, Optional

import numpy as np

from adtool.utils.factory import instantiate_object, object_spec
from examples.program_based_systems.types import GoalSampler


class InterferenceZoneGoalSampler(GoalSampler):
    def __init__(
        self,
        base_sampler_config: Optional[dict[str, Any]] = object_spec(
            "examples.program_based_systems.behavior_map.goal_sampler.RandomMinMaxGoalSampler"
        ),
        inside_probability: float = 0.8,
        max_attempts: int = 256,
    ) -> None:
        self.base_sampler = instantiate_object(
            base_sampler_config,
            object_name="base goal sampler",
        )
        self.inside_probability = float(inside_probability)
        self.max_attempts = int(max_attempts)

    def sample(
        self,
        history: list[np.ndarray],
        feature_size: Optional[int],
        min_: Optional[np.ndarray] = None,
        max_: Optional[np.ndarray] = None,
        goal_targeting: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> np.ndarray:
        sample_base = lambda: np.asarray(
            self.base_sampler.sample(
                history,
                feature_size,
                min_=min_,
                max_=max_,
                **kwargs,
            ),
            dtype=float,
        ).reshape(-1)
        base_goal = sample_base()
        if goal_targeting is None or not goal_targeting.get("zones"):
            return base_goal

        if np.random.random() < self.inside_probability:
            zone = goal_targeting["zones"][np.random.randint(0, len(goal_targeting["zones"]))]
            vectors = np.asarray(zone.get("behavior_vectors", []), dtype=float)
            if vectors.ndim == 1 and vectors.size > 0:
                vectors = vectors.reshape(1, -1)
            if vectors.ndim != 2 or len(vectors) == 0:
                raise ValueError("Goal-targeted zone does not contain any behavior vectors.")
            return np.array(vectors[np.random.randint(0, len(vectors))], copy=True)

        projected_zones = [
            (
                self._project_behavior_vectors(np.asarray(zone.get("behavior_vectors", []), dtype=float), goal_targeting),
                float(zone.get("coverage_radius", 0.0)),
            )
            for zone in goal_targeting["zones"]
        ]
        candidate = sample_base()
        for _ in range(self.max_attempts):
            point = self._project_behavior_vectors(candidate.reshape(1, -1), goal_targeting)[0]
            if not any(
                len(zone_points) > 0
                and np.any(np.linalg.norm(zone_points - point, axis=1) <= radius)
                for zone_points, radius in projected_zones
            ):
                return candidate
            candidate = sample_base()
        return candidate

    def _project_behavior_vectors(
        self,
        behavior_vectors: np.ndarray,
        goal_targeting: dict[str, Any],
    ) -> np.ndarray:
        if behavior_vectors.ndim == 1 and behavior_vectors.size > 0:
            behavior_vectors = behavior_vectors.reshape(1, -1)
        if behavior_vectors.ndim != 2 or len(behavior_vectors) == 0:
            return np.empty((0, 2), dtype=float)

        if goal_targeting.get("projection_method") == "axis":
            axes = list(goal_targeting["projection_axes"])
            axis_center = np.asarray(goal_targeting["axis_center"], dtype=float)
            axis_scale = float(goal_targeting["axis_scale"])
            return (behavior_vectors[:, axes] - axis_center) / axis_scale

        if goal_targeting.get("projection_method") == "pca":
            components = np.asarray(goal_targeting["pca_components"], dtype=float)
            pca_mean = np.asarray(goal_targeting["pca_mean"], dtype=float)
            output_center = np.asarray(goal_targeting["output_center"], dtype=float)
            output_scale = float(goal_targeting["output_scale"])
            input_mean = np.asarray(goal_targeting["input_mean"], dtype=float)
            input_std = np.asarray(goal_targeting["input_std"], dtype=float)
            x_norm = (behavior_vectors - input_mean) / input_std
            pca_points = (x_norm - pca_mean) @ components.T
            return (pca_points - output_center) / output_scale

        raise ValueError(f"Unsupported goal-targeting projection method: {goal_targeting.get('projection_method')}")
