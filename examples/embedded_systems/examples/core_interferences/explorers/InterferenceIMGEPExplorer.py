from typing import Dict

from pydantic import Field

from adtool.utils.expose_config.expose_config import expose
from adtool.examples.embedded_systems.explorers.embedded_systems_explorer import (
    BaseExplorerConfig,
    BaseIMGEPExplorer,
)


class InterferenceIMGEPConfig(BaseExplorerConfig):
    behavior_map_config: Dict = Field(default_factory=lambda: {
        "path": "adtool.examples.embedded_systems.examples.core_interferences.behavior_map.InterferenceBehaviorMap.InterferenceBehaviorMap"
    })
    parameter_map_config: Dict = Field(default_factory=lambda: {
        "path": "adtool.examples.embedded_systems.examples.core_interferences.parameter_map.InterferenceParameterMap.InterferenceParameterMap",
        "mixer_config": {
            "path": "adtool.examples.embedded_systems.examples.core_interferences.parameter_map.mutator.ChunkProgramMixer",
            "num_parts": 2,
        },
    })


@expose
class InterferenceIMGEPExplorer(BaseIMGEPExplorer):
    config = InterferenceIMGEPConfig

    def __init__(self, *args, **kwargs):
        pass
