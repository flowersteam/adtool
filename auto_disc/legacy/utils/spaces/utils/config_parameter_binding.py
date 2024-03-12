from __future__ import annotations

import json
import math
import typing
from typing import Any


class ConfigParameterBinding:
    """
    Allows to bind some properties of a space to the value of a config parameter that belongs to the config of the instance.
    """

    def __init__(self, parameter_name: str) -> None:
        """
        Initialize ConfigParamter by defining its name and the operations to perform on it (basic none)

        Args:
            parameter_name: the name of the parameter
        """
        self.parameter_name = parameter_name
        self._operations = []

    def _apply_operation(self, val1: Any, val2: Any, operator: str) -> Any:
        """
        Perform the previously recorded operation between 2 value dependign on the operator.

        Args:
            val1 : the first value involved in the calculation
            val2 : the second value involved in the calculation
            operator: the operator to define witch operation we have to do

        Returns:
            the result of the calculation
        """
        if operator == "+":
            return val1 + val2
        elif operator == "-":
            return val1 - val2
        elif operator == "*":
            return val1 * val2
        elif operator == "/":
            return val1 / val2
        elif operator == "//":
            return val1 // val2

    def __get__(self, obj: object, objtype: type = None) -> Any:
        """
        Acces to the result of an operation including one or more ConfigParameterBinding when we actually appply the binding

        Args:
            obj: an autodisc module (e.g. system, explorer) in which the configparameters need binding
            objtype: a type
        Returns:
            value: the result of the calculation
        """
        value = obj.config[self.parameter_name]
        for operation in self._operations:
            other = operation[0]
            operator = operation[1]
            if isinstance(other, ConfigParameterBinding):
                other = other.__get__(obj)
            value = self._apply_operation(value, other, operator)

        return value

    def __add__(self, other: ConfigParameterBinding) -> ConfigParameterBinding:
        """
        Append add operation to the configParameter

        Args:
            other: The second variables involve in the operation (with self)
        Returns:
            self: return self with it new operation
        """
        self._operations.append((other, "+"))
        return self

    def __radd__(self, other: ConfigParameterBinding) -> ConfigParameterBinding:
        """
        overwrite __radd__. Append add operation to the configParameter

        Args:
            other: The second variables involve in the operation (with self)
        Returns:
            self: return self with it new operation
        """
        return self.__add__(other)

    def __sub__(self, other: ConfigParameterBinding) -> ConfigParameterBinding:
        """
        Append sub operation to the configParameter

        Args:
            other: The second variables involve in the operation (with self)
        Returns:
            self: return self with it new operation
        """
        self._operations.append((other, "-"))
        return self

    def __rsub__(self, other: ConfigParameterBinding) -> ConfigParameterBinding:
        """
        overwrite __rsub__. Append sub operation to the configParameter

        Args:
            other: The second variables involve in the operation (with self)
        Returns:
            self: return self with it new operation
        """
        return self.__sub__(other)

    def __mul__(self, other: ConfigParameterBinding) -> ConfigParameterBinding:
        """
        Append mul operation to the configParameter

        Args:
            other: The second variables involve in the operation (with self)
        Returns:
            self: return self with it new operation
        """
        self._operations.append((other, "*"))
        return self

    def __rmul__(self, other: ConfigParameterBinding) -> ConfigParameterBinding:
        """
        overwrite __rmul__. Append mul operation to the configParameter

        Args:
            other: The second variables involve in the operation (with self)
        Returns:
            self: return self with it new operation
        """
        return self.__mul__(other)

    def __div__(self, other: ConfigParameterBinding) -> ConfigParameterBinding:
        """
        Append div operation to the configParameter

        Args:
            other: The second variables involve in the operation (with self)
        Returns:
            self: return self with it new operation
        """
        self._operations.append((other, "/"))
        return self

    def __rdiv__(self, other: ConfigParameterBinding) -> ConfigParameterBinding:
        """
        overwrite __rdiv__. Append div operation to the configParameter

        Args:
            other: The second variables involve in the operation (with self)
        Returns:
            self: return self with it new operation
        """
        return self.__div__(other)

    def __floordiv__(self, other: ConfigParameterBinding) -> ConfigParameterBinding:
        """
        Append floordiv operation to the configParameter

        Args:
            other: The second variables involve in the operation (with self)
        Returns:
            self: return self with it new operation
        """
        self._operations.append((other, "//"))
        return self

    def __rfloordiv__(self, other: ConfigParameterBinding) -> ConfigParameterBinding:
        """
        overwrite __rfloordiv__. Append floordiv operation to the configParameter

        Args:
            other: The second variables involve in the operation (with self)
        Returns:
            self: return self with it new operation
        """
        return self.__floordiv__(other)

    def to_json(self) -> str:
        """
        transforms the binding into a string to make it readable

        Returns:
            binding: the binding definition as string
        """
        binding = "binding." + self.parameter_name
        for operation in self._operations:
            other = operation[0]
            operator = operation[1]
            if isinstance(other, ConfigParameterBinding):
                other = other.to_json()
            binding = "({0}, {1}, {2})".format(binding, other, operator)
        return binding
