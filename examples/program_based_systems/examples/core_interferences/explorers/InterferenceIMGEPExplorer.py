from pydantic import Field

from adtool.utils.factory import ObjectSpec, object_spec
from adtool.utils.expose_config.expose_config import expose
from examples.program_based_systems.explorers.program_based_systems_explorer import (
    BaseExplorerConfig,
    BaseIMGEPExplorer,
)


class InterferenceIMGEPConfig(BaseExplorerConfig):
    behavior_map: ObjectSpec = Field(
        object_spec(
            "examples.program_based_systems.examples.core_interferences.behavior_map.InterferenceBehaviorMap.InterferenceBehaviorMap"
        )
    )
    parameter_map: ObjectSpec = Field(
        object_spec(
            "examples.program_based_systems.examples.core_interferences.parameter_map.InterferenceParameterMap.InterferenceParameterMap",
            {
                "mixer": {
                    "path": "examples.program_based_systems.examples.core_interferences.parameter_map.mutator.ChunkProgramMixer",
                    "config": {
                        "num_parts": 2,
                    },
                },
            },
        )
    )


@expose
class InterferenceIMGEPExplorer(BaseIMGEPExplorer):
    config = InterferenceIMGEPConfig

    def __init__(self, *args, **kwargs):
        pass
