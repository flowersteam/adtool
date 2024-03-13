from auto_disc.legacy.utils.config_parameters import BaseConfigParameter
import sys

class DecimalConfigParameter(BaseConfigParameter):
    """
    Decorator to add a decimal config parameter to a class.
    Uses a 'min' and 'max' to bound the authorized values.
    If a bound is set to 'None', this means no bound.
    """

    def __init__(
        self, name: str, default: float, min: float = None, max: float = None
    ) -> None:
        """
        Init a decimal config parameter. Define the bounds of the value

        Args:
            name: name of config parameter
            default: default value of the config parameter
            min: the lower bound
            max: the upper limit
        """
        print("DECIMAL CONFIG PARAMETER", name, default, min, max, file=sys.stderr)
        self._min = min
        self._max = max
        super().__init__(name, default)

    def check_value_to_set(self, value):
        """
        Check if the value is between the two bounds

        Args:
            value: current value of the config parameter
        Returns:
           The return value is True if the value is between the two bounds else an exception is thrown
        """
        if self._min is not None and value < self._min:
            raise Exception(
                "Chosen value ({0}) is lower than the minimum authorized ({1}).".format(
                    value, self._min
                )
            )

        if self._max is not None and value > self._max:
            raise Exception(
                "Chosen value ({0}) is lower than the maximum authorized ({1}).".format(
                    value, self._max
                )
            )

        return True

    def __call__(self, original_class):
        """
        Define correctly an autodisc modules considering the decimal (decorator) config parameter

        Args:
            original_class: The class of the module we want to define
        Returns:
            new_class: The updated class with the config parameter

        """
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]["type"] = "DECIMAL"
        new_class.CONFIG_DEFINITION[self._name]["min"] = self._min
        new_class.CONFIG_DEFINITION[self._name]["max"] = self._max

        return new_class
