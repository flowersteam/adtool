from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydoc import locate

import numpy as np

from adtool.utils.leaf.Leaf import Leaf
from examples.core_interference.systems.InterferenceSystem import InterferenceSystem
from examples.core_interference.types import BehaviorEncoder, GoalSampler


class InterferenceBehaviorMap(Leaf):
	"""Behavior map."""

	def __init__(
		self,
		system: InterferenceSystem,
		premap_key: str = "output",
		postmap_key: str = "output",
		goal_sampler: str = (
			"examples.core_interference.goal_samplers.RandomMinMaxGoalSampler"
		),
		goal_sampler_config: Optional[Dict[str, Any]] = None,
		behavior_encoder: str = (
			"examples.core_interference.behavior_encoders.InterferenceMetricEncoder"
		),
		behavior_encoder_config: Optional[Dict[str, Any]] = None,
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
		self.behavior_encoder = self.make_behavior_encoder(
			behavior_encoder,
			behavior_encoder_config or {},
		)

	def map(self, input: Dict) -> Dict:
		intermed = deepcopy(input)

		raw_output = intermed[self.premap_key]
		embedding = self.behavior_encoder.encode(raw_output)

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

	def make_behavior_encoder(
		self,
		behavior_encoder_path: str,
		behavior_encoder_config: Dict[str, Any],
	) -> BehaviorEncoder:
		behavior_encoder_cls = locate(behavior_encoder_path)
		if behavior_encoder_cls is None:
			raise ValueError(
				f"Could not retrieve behavior encoder class from path: {behavior_encoder_path}."
			)
		return behavior_encoder_cls(**behavior_encoder_config)
