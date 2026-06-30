from typing import Any, Dict, Optional

from examples.program_based_systems.behavior_map.program_based_systems_behavior_map import (
    BaseBehaviorMap,
)
from examples.program_based_systems.examples.core_interferences.systems.InterferenceSystem import (
    InterferenceSystem,
)
from examples.program_based_systems.helpers.module_factory import make_module


class InterferenceBehaviorMap(BaseBehaviorMap):
    """Behavior map."""

    def __init__(
            self,
            system: InterferenceSystem,
            premap_key: str = "output",
            postmap_key: str = "output",
            goal_sampler_config: Optional[Dict[str, Any]] = {
                "path": "examples.program_based_systems.examples.core_interferences.behavior_map.goal_sampler.InterferenceZoneGoalSampler",
                "base_sampler_config": {
                    "path": "examples.program_based_systems.behavior_map.goal_sampler.RandomMinMaxGoalSampler"
                },
            },
            behavior_encoder_config: Optional[Dict[str, Any]] = {
                "path": "examples.program_based_systems.examples.core_interferences.behavior_map.encoder.InterferenceMetricEncoder"
            },
    ) -> None:
        _ = system
        goal_sampler = make_module("goal_sampler", **goal_sampler_config)
        behavior_encoder = make_module("behavior_encoder", **behavior_encoder_config)
        super().__init__(
            premap_key=premap_key,
            postmap_key=postmap_key,
            goal_sampler=goal_sampler,
            behavior_encoder=behavior_encoder,
        )
