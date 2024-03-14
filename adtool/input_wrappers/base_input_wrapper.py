from copy import deepcopy

from adtool import BaseAutoDiscModule
from adtool.utils.spaces import DictSpace


class BaseInputWrapper(BaseAutoDiscModule):
    """Base class to map the parameters sent by the explorer to the system's input space"""

    input_space = DictSpace()

    def __init__(self, wrapped_output_space_key=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_space = deepcopy(self.input_space)
        self.input_space.initialize(self)
        self._wrapped_output_space_key = wrapped_output_space_key
        self._initial_input_space_keys = [key for key in self.input_space]

    def initialize(self, output_space: DictSpace) -> None:
        """
        Defines input and output space for the input wrapper.
        """
        self.output_space = output_space
        for key in iter(output_space):
            if key != self._wrapped_output_space_key:
                self.input_space[key] = output_space[key]

    def map(self, input, is_input_new_discovery, **kwargs):
        """
        Map the input parameters (from the explorer) to the output parameters (sytem input)

        Args:
            input: input parameters
            is_input_new_discovery: indicates if it is a new discovery
        """
        raise NotImplementedError
