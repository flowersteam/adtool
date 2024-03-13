import os
from os import environ as env
from typing import Union, get_type_hints


def _parse_bool(val: Union[str, bool]) -> bool:  # pylint: disable=E1136
    return val if type(val) == bool else val.lower() in ["true", "yes", "1"]


class ConfigError(Exception):
    pass


class Config:
    EXPEDB_CALLER_HOST: str = "127.0.0.1"
    EXPEDB_CALLER_PORT: str = "5001"

    def __init__(self):
        for field in self.__annotations__:
            default_value = getattr(self, field, None)
            if default_value is None and env.get(field) is None:
                raise ConfigError("The {} field is required".format(field))

            var_type = get_type_hints(Config)[field]

            try:
                if var_type == bool:
                    value = _parse_bool(env.get(field, default_value))
                else:
                    value = var_type(env.get(field, default_value))

                self.__setattr__(field, value)
            except:
                raise ConfigError(
                    'Unable to cast value of "{}" to type "{}" for "{}" field'.format(
                        env[field], var_type, field
                    )
                )
