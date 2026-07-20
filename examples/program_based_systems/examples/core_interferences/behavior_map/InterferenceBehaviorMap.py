from typing import Any, Dict, Optional

from adtool.utils.factory import instantiate_object, object_spec
from examples.program_based_systems.behavior_map.program_based_systems_behavior_map import (
    BaseBehaviorMap,
)
from examples.program_based_systems.examples.core_interferences.systems.InterferenceSystem import (
    InterferenceSystem,
)


class InterferenceBehaviorMap(BaseBehaviorMap):
    """Behavior map."""

    def __init__(
            self,
            system: InterferenceSystem,
            premap_key: str = "output",
            postmap_key: str = "output",
            goal_sampler: Optional[Dict[str, Any]] = object_spec(
                "examples.program_based_systems.examples.core_interferences.behavior_map.goal_sampler.InterferenceZoneGoalSampler",
                {
                    "base_sampler": {
                        "path": "examples.program_based_systems.behavior_map.goal_sampler.RandomMinMaxGoalSampler",
                        "config": {},
                    },
                },
            ),
            behavior_encoder: Optional[Dict[str, Any]] = object_spec(
                "examples.program_based_systems.examples.core_interferences.behavior_map.encoder.InterferenceMetricEncoder"
            ),
    ) -> None:
        _ = system
        goal_sampler = instantiate_object(
            goal_sampler,
            object_name="goal sampler",
        )
        behavior_encoder = instantiate_object(
            behavior_encoder,
            object_name="behavior encoder",
        )
        super().__init__(
            premap_key=premap_key,
            postmap_key=postmap_key,
            goal_sampler=goal_sampler,
            behavior_encoder=behavior_encoder,
        )
