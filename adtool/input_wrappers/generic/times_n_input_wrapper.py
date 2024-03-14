import numpy as np
from addict import Dict
from adtool.input_wrappers import BaseInputWrapper
from adtool.utils.spaces import BoxSpace, DictSpace

from adtool.utils.expose_config.defaults import Defaults, defaults
from dataclasses import dataclass, field

#@IntegerConfigParameter(name="n", default=1)

@dataclass
class TimesNInputWrapperConfig(Defaults):
    n: int = defaults(1)

@TimesNInputWrapperConfig.expose_config()
class TimesNInputWrapper(BaseInputWrapper):

    input_space = DictSpace(
        input_parameter=BoxSpace(low=-np.inf, high=np.inf, shape=())
    )

    def __init__(self, wrapped_output_space_key: str, **kwargs) -> None:
        super().__init__(wrapped_output_space_key, **kwargs)
        assert len(self.input_space) == 1
        if not isinstance(wrapped_output_space_key, str):
            raise TypeError(
                "wrapped_output_space_key must be a single string indicating the key of the space to wrap."
            )

        # Change key name to avoid issues with multiple same wrappers stacked
        new_key = "Times{0}_{1}".format(self.config["n"], wrapped_output_space_key)
        self.input_space[new_key] = self.input_space[self._initial_input_space_keys[0]]
        del self.input_space[self._initial_input_space_keys[0]]
        self._initial_input_space_keys = [new_key]

    def map(self, input: Dict, is_input_new_discovery, **kwargs) -> Dict:
        input[self._wrapped_output_space_key] = (
            input[self._initial_input_space_keys[0]] * self.config["n"]
        )
        del input[self._initial_input_space_keys[0]]
        return input
