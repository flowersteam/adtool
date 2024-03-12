from typing import Dict

from auto_disc.legacy.input_wrappers import BaseInputWrapper


class DummyInputWrapper(BaseInputWrapper):
    """
    Empty InputWrapper used when no wrapper should be used.
    """

    def __init__(self, wrapped_output_space_key: str = None, **kwargs) -> None:
        super().__init__(wrapped_output_space_key=wrapped_output_space_key, **kwargs)

    def map(self, input: Dict, is_input_new_discovery, **kwargs) -> Dict:
        return input
