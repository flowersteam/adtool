from typing import Any

from addict import Dict

import sys

class BaseConfigParameter:
    """
    Decorator to add a config parameter to a class.
    """

    def __init__(self, name: str, default: Any) -> None:
        """
        Init a config parameter

        Args:
            name: name of config parameter
            default: default value of the config parameter
        """
        assert self.check_value_to_set(default)
        self._default = default
        self._name = name

    def check_value_to_set(self, value: Any) -> bool:
        """
        Check if the value is one of the possible values

        Args:
            value: current value of the config parameter
        Returns:
           The return value is always True. All values are accepted
        """
        return True

    def __call__(self, original_class: type) -> type:
        """
        Define correctly an autodisc modules considering the (decorator) config parameter

        Args:
            original_class: The class of the module we want to define
        Returns:
            original_class: The updated class with the config parameter

        """
        original_init = original_class.__init__
        # Make copy of original __init__, so we can call it without recursion
        default_value = self._default
        name = self._name
        check_value = self.check_value_to_set

        def __init__(self, *args, **kws) -> None:
            value_to_set = default_value
            if name in kws:
                assert check_value(kws[name])
                value_to_set = kws[name]
                del kws[name]

            if not hasattr(self, "config"):  # Initialize config if not done
                self.config = Dict()

            self.config[name] = value_to_set  # add entry to config
            # setattr(self, name, value_to_set)

            original_init(self, *args, **kws)  # Call the original __init__

        original_class.__init__ = __init__  # Set the class' __init__ to the new one

        # Add the parameter to the config definition
        if not hasattr(original_class, "CONFIG_DEFINITION"):
            raise Exception(
                "Class {} should define an empty static CONFIG_DEFINITION: `CONFIG_DEFINITION = {}`".format(
                    original_class
                )
            )
        original_class.CONFIG_DEFINITION[name] = {"default": default_value}

        return original_class
