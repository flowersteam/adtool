#!/usr/bin/env python3
import io
from typing import Any, Dict, Optional, Tuple

from examples.embedded_systems.helpers.module_factory import make_module
import imageio
import numpy as np
from pydantic import BaseModel, Field

from adtool.systems.System import System
from adtool.utils.expose_config.expose_config import expose
from examples.embedded_systems.examples.core_interferences.helpers.interference_visualizer import (
    render_interference_dashboard,
)
from examples.embedded_systems.examples.core_interferences.helpers.interference_normalization import (
    normalize_instruction_program,
)
from examples.embedded_systems.examples.core_interferences.types import (
    InterferenceDynamicParams,
    InterferenceParamsPayload,
    InterferenceSimulatorConfig,
    InterferenceSimulatorRunnerConfig,
)


class InterferenceConfig(BaseModel):
    simulator_config: InterferenceSimulatorConfig = Field(
        default_factory=lambda: {
            "path": "examples.embedded_systems.examples.core_interferences.simulator.Sim3Backend",
            "cycles": 80,
            "num_banks": 4,
            "num_addr": 41,
        }
    )
    simulator_runner_config: InterferenceSimulatorRunnerConfig = Field(
        default_factory=lambda: {
            "path": "examples.embedded_systems.examples.core_interferences.simulator_runners.DefaultEnvSimulatorRunner",
        }
    )


@expose
class InterferenceSystem(System):
    """System wrapper around interference environment."""

    config = InterferenceConfig

    def __init__(
            self,
            simulator_config: Optional[InterferenceSimulatorConfig] = None,
            simulator_runner_config: Optional[InterferenceSimulatorRunnerConfig] = None,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if simulator_config is None and simulator_runner_config is None:
            simulator_runner_config = dict(self.config.simulator_runner_config)
            simulator_config = dict(self.config.simulator_config)
        elif simulator_config is None or simulator_runner_config is None:
            raise ValueError(
                "Both simulator_config and simulator_runner_config must be provided together."
            )

        simulator = make_module(
            "simulator",
            **simulator_config,
        )
        self.simulator_runner = make_module(
            "simulator_runner",
            simulator=simulator,
            **simulator_runner_config,
        )
        self._fallback_frame = np.zeros((64, 64, 3), dtype=np.uint8)

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
            byte_img = io.BytesIO()
            imageio.imwrite(byte_img, self._fallback_frame, format="png")
            byte_img.seek(0)
            return [(byte_img.getvalue(), "png")]

        dashboard_png = render_interference_dashboard(data_dict)
        return [(dashboard_png, "png")]
