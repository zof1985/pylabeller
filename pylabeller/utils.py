"""UTILS MODULE"""


#! IMPORTS


from types import FunctionType, MethodType
from typing import Any


#! FUNCTIONS


def check_type(obj: object, typ: type) -> bool:
    """ensure the object is of the provided type/s"""
    if not isinstance(obj, typ):
        raise TypeError(f"{obj} must be an instance of {typ}.")
    return True


#! CLASSES


class Signal:
    """
    class allowing to generate event signals and connect them to functions.
    """

    # ****** VARIABLES ****** #

    _connected_fun = None

    # ****** CONSTRUCTOR ****** #

    def __init__(self) -> None:
        self._connected_fun = None

    # ****** PROPERTIES ****** #

    @property
    def connected_function(self) -> FunctionType | MethodType | None:
        """return the function connected to this signal."""
        return self._connected_fun

    # ****** METHODS ****** #

    def emit(self, *args, **kwargs) -> Any | None:
        """emit the signal with the provided parameters."""
        if self.connected_function is None:
            return None
        return self.connected_function(*args, **kwargs)

    def connect(self, fun: FunctionType | MethodType) -> None:
        """
        connect a function/method to the actual signal

        Parameters
        ----------
        fun: FunctionType | MethodType
            the function to be connected to the signal.
        """
        check_type(fun, (MethodType, FunctionType))
        self._connected_fun = fun

    def disconnect(self) -> None:
        """disconnect the signal from the actual function."""
        self._connected_fun = None

    def is_connected(self) -> bool:
        """check whether the signal is connected to a function."""
        return self.is_connected is not None
