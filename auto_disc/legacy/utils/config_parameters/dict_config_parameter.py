import typing

from auto_disc.legacy.utils.config_parameters import BaseConfigParameter


class DictConfigParameter(BaseConfigParameter):
    """
    Decorator to add a dict config parameter to a class.
    """

    def __init__(self, name: str, default: typing.Dict = {}) -> None:
        """
        Init a dict config parameter.

        Args:
            name: name of config parameter
            default: default value of the config parameter
        """
        super().__init__(name, default)

    def check_value_to_set(self, value: typing.Dict) -> bool:
        """
        Check if the value is indeed a dict

        Args:
            value: current value of the config parameter
        Returns:
           The return value is True if the value is a dict
        """
        assert isinstance(value, dict), "Passed value is not a dict"
        return True

    def __call__(self, original_class):
        """
        Define correctly an autodisc modules considering the dict (decorator) config parameter

        Args:
            original_class: The class of the module we want to define
        Returns:
            new_class: The updated class with the config parameter

        """
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]["type"] = "DICT"

        return new_class
