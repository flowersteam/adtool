from auto_disc.legacy.utils.config_parameters import DecimalConfigParameter


class IntegerConfigParameter(DecimalConfigParameter):
    """
    Decorator to add an integer config parameter to a class.
    Uses a 'min' and 'max' to bound the authorized values.
    If a bound is set to 'None', this means no bound.
    If a decimal value is passed, it will be rounded.
    """

    def __init__(
        self, name: str, default: int, min: int = None, max: int = None
    ) -> None:
        """
        Init a int config parameter. Define the bounds of the value

        Args:
            name: name of config parameter
            default: default value of the config parameter
            min: the lower bound
            max: the upper limit
        """
        super().__init__(
            name,
            round(default),
            min=round(min) if min else min,
            max=round(max) if max else max,
        )

    def check_value_to_set(self, value: int) -> bool:
        """
        Check if the value is between the two bounds

        Args:
            value: current value of the config parameter
        Returns:
           The return value is True if the value is between the two bounds else an exception is thrown
        """
        return super().check_value_to_set(round(value))

    def __call__(self, original_class: type) -> type:
        """
        Define correctly an autodisc modules considering the int (decorator) config parameter

        Args:
            original_class: The class of the module we want to define
        Returns:
            new_class: The updated class with the config parameter

        """
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]["type"] = "INTEGER"

        return new_class
