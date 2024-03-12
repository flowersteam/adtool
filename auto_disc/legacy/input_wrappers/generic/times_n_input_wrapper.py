import numpy as np
from addict import Dict
from auto_disc.legacy.input_wrappers import BaseInputWrapper
from auto_disc.legacy.utils.config_parameters import IntegerConfigParameter
from auto_disc.legacy.utils.spaces import BoxSpace, DictSpace


@IntegerConfigParameter(name="n", default=1)
class TimesNInputWrapper(BaseInputWrapper):
    CONFIG_DEFINITION = {}

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
