"""
Helpers for dealing with NumPy arrays.

Originally copied from openff-nagl.
"""

from typing import Annotated

import numpy
from numpy.typing import NDArray
from openff.toolkit import Quantity
from pydantic import BeforeValidator, WrapSerializer


def _strip_units(val: list[float] | Quantity | NDArray) -> NDArray:
    if hasattr(val, "magnitude"):
        unitless_val = val.magnitude
    else:
        unitless_val = val

    return numpy.asarray(unitless_val).reshape((-3, 3))


def _array_serializer(val: NDArray[numpy.float64], nxt) -> list[float]:
    return val.flatten().tolist()


CoordinateArray = Annotated[
    NDArray[numpy.float64],
    BeforeValidator(_strip_units),
    WrapSerializer(_array_serializer),
]

Array = CoordinateArray
