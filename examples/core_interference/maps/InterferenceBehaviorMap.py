from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydoc import locate

import numpy as np

from adtool.utils.leaf.Leaf import Leaf
from examples.core_interference.systems.InterferenceSystem import InterferenceSystem
from examples.core_interference.types import GoalSampler


class InterferenceBehaviorMap(Leaf):
	"""Behavior map."""

	_SELECTION = [
		"miss_core0",
		"miss_core1",
		"hits_core0",
		"hits_core1",
		"diff_time_core0",
		"diff_time_core1",
		"diff_time",
	]
	_SELECTION += [
		f"L2_{c}_{type_}_{core}"
		for c in ["miss", "hit"]
		for type_ in ["write", "read"]
		for core in ["core0", "core1"]
	]
	# The ordering of `_SELECTION` defines the geometry of the behavior space.
	# Changing this order changes nearest-neighbor behavior in the explorer.

	def __init__(
		self,
		system: InterferenceSystem,
		premap_key: str = "output",
		postmap_key: str = "output",
		goal_sampler: str = (
			"examples.core_interference.goal_samplers.RandomMinMaxGoalSampler"
		),
		goal_sampler_config: Optional[Dict[str, Any]] = None,
	) -> None:
		super().__init__()
		_ = system
		self.premap_key = premap_key
		self.postmap_key = postmap_key
		self._history: List[np.ndarray] = []
		self._feature_size = None
		self.goal_sampler = self.make_goal_sampler(
			goal_sampler,
			goal_sampler_config or {},
		)

	def _extract_metrics(self, sim_output: Dict) -> np.ndarray:
		mutual = sim_output.get("mutual", {})
		observation_vec = []
		# Keep feature ordering deterministic because the explorer compares
		# vectors across time using index-wise distances.
		for key in self._SELECTION:
			value = np.array(mutual.get(key, 0.0), dtype=float).reshape((-1,))
			observation_vec.append(value)

		if not observation_vec:
			return np.array([], dtype=float)

		metrics = np.concatenate(observation_vec)
		# Sanitize non-finite values early so downstream kNN policy never sees
		# NaN/Inf and can keep a stable history.
		metrics = np.nan_to_num(metrics, nan=0.0, posinf=0.0, neginf=0.0)
		return metrics.astype(float)

	def map(self, input: Dict) -> Dict:
		intermed = deepcopy(input)

		raw_output = intermed[self.premap_key]
		embedding = self._extract_metrics(raw_output)

		intermed["raw_" + self.premap_key] = raw_output
		del intermed[self.premap_key]
		# `postmap_key` is what explorers treat as the behavior embedding used
		# for kNN retrieval and goal matching.
		intermed[self.postmap_key] = embedding

		self._history.append(embedding)
		# `_feature_size` allows safe cold-start goal sampling even before enough
		# history has accumulated.
		if self._feature_size is None:
			self._feature_size = embedding.size
		return intermed

	def sample(self) -> np.ndarray:
		"""Sample goals from behavior history."""
		return self.goal_sampler.sample(self._history, self._feature_size)

	def make_goal_sampler(
		self,
		goal_sampler_path: str,
		goal_sampler_config: Dict[str, Any],
	) -> GoalSampler:
		goal_sampler_cls = locate(goal_sampler_path)
		if goal_sampler_cls is None:
			raise ValueError(
				f"Could not retrieve goal sampler class from path: {goal_sampler_path}."
			)
		return goal_sampler_cls(**goal_sampler_config)
