#!/usr/bin/env python3
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from examples.program_based_systems.systems.program_based_systems_system import BaseProgramSystem
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.factory import ObjectSpec, instantiate_object, object_spec
from examples.program_based_systems.examples.core_interferences.helpers.interference_visualizer import (
    render_interference_dashboard,
)
from examples.program_based_systems.examples.core_interferences.helpers.interference_normalization import (
    normalize_instruction_program,
)
from examples.program_based_systems.examples.core_interferences.types import (
    InterferenceDynamicParams,
    InterferenceParamsPayload,
    InterferenceSimulatorConfig,
    InterferenceSimulatorRunnerConfig,
)


class InterferenceConfig(BaseModel):
    simulator_config: InterferenceSimulatorConfig = Field(
        object_spec(
            "examples.program_based_systems.examples.core_interferences.systems.simulator.Sim3Backend",
            {
                "cycles": 80,
                "num_banks": 4,
                "num_addr": 41,
            },
        )
    )
    simulator_runner_config: InterferenceSimulatorRunnerConfig = Field(
        object_spec(
            "examples.program_based_systems.examples.core_interferences.systems.runner.DefaultEnvSimulatorRunner"
        )
    )


@expose
class InterferenceSystem(BaseProgramSystem):
    """System wrapper around interference environment."""

    config = InterferenceConfig

    def __init__(
            self,
            simulator_config: Optional[ObjectSpec] = None,
            simulator_runner_config: Optional[ObjectSpec] = None,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if simulator_config is None and simulator_runner_config is None:
            simulator_runner_config = self.config.simulator_runner_config
            simulator_config = self.config.simulator_config
        elif simulator_config is None or simulator_runner_config is None:
            raise ValueError(
                "Both simulator_config and simulator_runner_config must be provided together."
            )

        simulator = instantiate_object(
            simulator_config,
            object_name="simulator",
        )
        self.simulator_runner = instantiate_object(
            simulator_runner_config,
            simulator=simulator,
            object_name="simulator runner",
        )

    def map(self, input: Dict) -> Dict:
        # Copy input dict to avoid mutating upstream payload references.
        data = dict(input)
        params_payload: InterferenceParamsPayload = data["params"]
        dynamic_params = params_payload["dynamic_params"]

        params: InterferenceDynamicParams = {
            "core0": normalize_instruction_program(
                dynamic_params["core0"],
                strict=True,
                context="InterferenceSystem.core0",
            ),
            "core1": normalize_instruction_program(
                dynamic_params["core1"],
                strict=True,
                context="InterferenceSystem.core1",
            ),
        }

        # Delegates simulation execution to a path-configured runner strategy.
        output = self.simulator_runner.run(params)
        data["output"] = output

        return data

    def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:
        output = data_dict.get("output", {}) if isinstance(
            data_dict, dict) else {}

        if not output:
            return self.render_fallback()

        dashboard_png = render_interference_dashboard(data_dict)
        return [(dashboard_png, "png")]
