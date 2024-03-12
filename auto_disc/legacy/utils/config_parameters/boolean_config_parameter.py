from auto_disc.legacy.utils.config_parameters import BaseConfigParameter


class BooleanConfigParameter(BaseConfigParameter):
    """
    Decorator to add a boolean config parameter to a class.
    """

    def __init__(self, name: str, default: bool) -> None:
        """
        Init a boolean config parameter. Define all possible values

        Args:
            name: name of config parameter
            default: default value of the config parameter
        """
        self._possible_values = [True, False]
        super().__init__(name, default)

    def check_value_to_set(self, value: bool) -> bool:
        """
        Check if the value is one of the possible values

        Args:
            value: current value of the config parameter
        Returns:
           The return value is True if the value is a boolean else an exception was raised
        """
        if value in self._possible_values:
            return True
        else:
            raise Exception("Chosen value ({0}) is not a boolean.".format(value))

    def __call__(self, original_class: type) -> type:
        """
        Define correctly an autodisc modules considering the boolean (decorator) config parameter

        Args:
            original_class: The class of the module we want to define
        Returns:
            new_class: The updated class with the config parameter

        """
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]["type"] = "BOOLEAN"
        new_class.CONFIG_DEFINITION[self._name][
            "possible_values"
        ] = self._possible_values

        return new_class
