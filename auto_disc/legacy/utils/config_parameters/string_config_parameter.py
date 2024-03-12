import typing

from auto_disc.legacy.utils.config_parameters import BaseConfigParameter


class StringConfigParameter(BaseConfigParameter):
    """
    Decorator to add a string config parameter to a class.
    Uses a list of possible values to choose among.
    """

    def __init__(
        self, name: str, default: str = "", possible_values: typing.List[str] = None
    ) -> None:
        """
        Init a str config parameter. Define the list of all possible values.

        Args:
            name: name of config parameter
            default: default value of the config parameter
            possible_values: List of all possible string
        """
        self._possible_values = possible_values
        if possible_values is not None and default not in possible_values:
            raise Exception("Default value not in possible values.")

        super().__init__(name, default)

    def check_value_to_set(self, value: str) -> bool:
        """
        Check if the value is one of the possible values

        Args:
            value: current value of the config parameter
        Returns:
           The return value is True if the value is in possible values
        """
        assert isinstance(value, str), "Passed value is not a string"
        if self._possible_values is not None:
            if value in self._possible_values:
                return True
            else:
                raise Exception(
                    "Chosen value ({0}) does not belong to the authorized list ({1}).".format(
                        value, self._possible_values
                    )
                )

        return True

    def __call__(self, original_class: type) -> type:
        """
        Define correctly an autodisc modules considering the string (decorator) config parameter

        Args:
            original_class: The class of the module we want to define
        Returns:
            new_class: The updated class with the config parameter

        """
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]["type"] = "STRING"
        new_class.CONFIG_DEFINITION[self._name][
            "possible_values"
        ] = self._possible_values

        return new_class
