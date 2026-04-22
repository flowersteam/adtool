#!/usr/bin/env python3
import io
from typing import Any, Dict, Tuple

import imageio
import numpy as np
from pydantic import BaseModel, Field

from adtool.systems.System import System
from adtool.utils.expose_config.expose_config import expose
from examples.core_interference.helpers.envi import Env
from examples.core_interference.helpers.interference_visualizer import (
	render_interference_dashboard,
)
from examples.core_interference.helpers.modifiers.normalization import (
	normalize_instruction_program,
)
from examples.core_interference.types import (
	InterferenceDynamicParams,
	InterferenceParamsPayload,
)

class InterferenceConfig(BaseModel):
	cycles: int = Field(80, ge=1, le=100000)
	num_banks: int = Field(4, ge=1, le=64)
	num_addr: int = Field(41, ge=4, le=4096)


@expose
class InterferenceSystem(System):
	"""System wrapper around interference environment."""

	config = InterferenceConfig

	def __init__(
		self,
		cycles: int = 80,
		num_banks: int = 4,
		num_addr: int = 41,
		*args,
		**kwargs,
	) -> None:
		super().__init__(*args, **kwargs)
		self.cycles = cycles
		self.num_banks = num_banks
		self.num_addr = num_addr
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

		env = Env(
			cycles=self.cycles,
			num_banks=self.num_banks,
			num_addr=self.num_addr,
		)
		# A fresh Env per call keeps each rollout independent from previous
		# simulator side effects/state.
		# Env execution produces the structured observation consumed by
		# InterferenceBehaviorMap.
		output = env(params)
		data["output"] = output

		return data

	def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:
		output = data_dict.get("output", {}) if isinstance(data_dict, dict) else {}

		if not output:
			byte_img = io.BytesIO()
			imageio.imwrite(byte_img, self._fallback_frame, format="png")
			byte_img.seek(0)
			return [(byte_img.getvalue(), "png")]

		dashboard_png = render_interference_dashboard(data_dict)
		return [(dashboard_png, "png")]
