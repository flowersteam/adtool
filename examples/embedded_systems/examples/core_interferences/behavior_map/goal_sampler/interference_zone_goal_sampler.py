from __future__ import annotations

from typing import Any, Optional

import numpy as np

from adtool.examples.embedded_systems.helpers.module_factory import make_module
from adtool.examples.embedded_systems.types import GoalSampler


class InterferenceZoneGoalSampler(GoalSampler):
    def __init__(
        self,
        base_sampler_config: Optional[dict[str, Any]] = None,
        inside_probability: float = 0.8,
        max_attempts: int = 256,
    ) -> None:
        self.base_sampler = make_module(
            "base_goal_sampler",
            **(
                base_sampler_config
            ),
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
        base_goal = self._sample_base(history, feature_size, min_=min_, max_=max_)
        if goal_targeting is None or not goal_targeting.get("zones"):
            return base_goal

        if goal_targeting.get("kind") == "axis":
            if np.random.random() < self.inside_probability:
                return self._sample_inside_axis(base_goal, goal_targeting)
            return self._sample_outside_axis(
                history,
                feature_size,
                goal_targeting,
                min_=min_,
                max_=max_,
            )

        if goal_targeting.get("kind") == "pca":
            if np.random.random() < self.inside_probability:
                return self._sample_inside_pca(base_goal, goal_targeting)
            return self._sample_outside_pca(
                history,
                feature_size,
                goal_targeting,
                min_=min_,
                max_=max_,
            )

        return base_goal

    def _sample_base(
        self,
        history: list[np.ndarray],
        feature_size: Optional[int],
        **kwargs,
    ) -> np.ndarray:
        return np.asarray(
            self.base_sampler.sample(
                history,
                feature_size,
                **kwargs,
            ),
            dtype=float,
        ).reshape(-1)

    def _sample_inside_axis(self, base_goal: np.ndarray, goal_targeting: dict[str, Any]) -> np.ndarray:
        zone = self._random_zone(goal_targeting["zones"])
        point = np.asarray(zone["center"], dtype=float) + self._sample_disk(float(zone["radius"]))
        candidate = np.array(base_goal, copy=True)
        axes = list(goal_targeting["projection_axes"])
        axis_center = np.asarray(goal_targeting["axis_center"], dtype=float)
        axis_scale = float(goal_targeting["axis_scale"])
        candidate[axes[0]] = axis_center[0] + point[0] * axis_scale
        candidate[axes[1]] = axis_center[1] + point[1] * axis_scale
        return candidate

    def _sample_outside_axis(
        self,
        history: list[np.ndarray],
        feature_size: Optional[int],
        goal_targeting: dict[str, Any],
        **kwargs,
    ) -> np.ndarray:
        axes = list(goal_targeting["projection_axes"])
        axis_center = np.asarray(goal_targeting["axis_center"], dtype=float)
        axis_scale = float(goal_targeting["axis_scale"])

        candidate = self._sample_base(history, feature_size, **kwargs)
        for _ in range(self.max_attempts):
            point = (candidate[axes] - axis_center) / axis_scale
            if not self._point_in_any_zone(point, goal_targeting["zones"]):
                return candidate
            candidate = self._sample_base(history, feature_size, **kwargs)
        return candidate

    def _sample_inside_pca(self, base_goal: np.ndarray, goal_targeting: dict[str, Any]) -> np.ndarray:
        zone = self._random_zone(goal_targeting["zones"])
        target_norm = np.asarray(zone["center"], dtype=float) + self._sample_disk(float(zone["radius"]))
        components = np.asarray(goal_targeting["pca_components"], dtype=float)
        pca_mean = np.asarray(goal_targeting["pca_mean"], dtype=float)
        output_center = np.asarray(goal_targeting["output_center"], dtype=float)
        output_scale = float(goal_targeting["output_scale"])
        input_mean = np.asarray(goal_targeting["input_mean"], dtype=float)
        input_std = np.asarray(goal_targeting["input_std"], dtype=float)

        x_norm = (np.asarray(base_goal, dtype=float) - input_mean) / input_std
        current_pca = (x_norm - pca_mean) @ components.T
        target_pca = output_center + target_norm * output_scale
        delta = (target_pca - current_pca) @ components
        shifted = x_norm + delta
        return shifted * input_std + input_mean

    def _sample_outside_pca(
        self,
        history: list[np.ndarray],
        feature_size: Optional[int],
        goal_targeting: dict[str, Any],
        **kwargs,
    ) -> np.ndarray:
        components = np.asarray(goal_targeting["pca_components"], dtype=float)
        pca_mean = np.asarray(goal_targeting["pca_mean"], dtype=float)
        output_center = np.asarray(goal_targeting["output_center"], dtype=float)
        output_scale = float(goal_targeting["output_scale"])
        input_mean = np.asarray(goal_targeting["input_mean"], dtype=float)
        input_std = np.asarray(goal_targeting["input_std"], dtype=float)

        candidate = self._sample_base(history, feature_size, **kwargs)
        for _ in range(self.max_attempts):
            x_norm = (candidate - input_mean) / input_std
            pca_point = (x_norm - pca_mean) @ components.T
            point = (pca_point - output_center) / output_scale
            if not self._point_in_any_zone(point, goal_targeting["zones"]):
                return candidate
            candidate = self._sample_base(history, feature_size, **kwargs)
        return candidate

    def _point_in_any_zone(self, point: np.ndarray, zones: list[dict[str, Any]]) -> bool:
        return any(
            np.linalg.norm(point - np.asarray(zone["center"], dtype=float)) <= float(zone["radius"])
            for zone in zones
        )

    def _random_zone(self, zones: list[dict[str, Any]]) -> dict[str, Any]:
        return zones[np.random.randint(0, len(zones))]

    def _sample_disk(self, radius: float) -> np.ndarray:
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        magnitude = radius * np.sqrt(np.random.uniform(0.0, 1.0))
        return np.array(
            [
                magnitude * np.cos(angle),
                magnitude * np.sin(angle),
            ],
            dtype=float,
        )
