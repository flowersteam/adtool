from copy import deepcopy
from typing import Dict, List

import numpy as np

from adtool.utils.leaf.Leaf import Leaf
from examples.core_interference.helpers.goal_sampling import sample_goal_from_history
from examples.core_interference.systems.InterferenceSystem import InterferenceSystem


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
	) -> None:
		super().__init__()
		_ = system
		self.premap_key = premap_key
		self.postmap_key = postmap_key
		self._history: List[np.ndarray] = []
		self._feature_size = None

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
		return sample_goal_from_history(self._history, self._feature_size)
