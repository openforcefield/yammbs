"""Helpers for dealing with NumPy arrays."""

from typing import Annotated

import numpy
from numpy.typing import NDArray
from openff.toolkit import Quantity
from pydantic import BeforeValidator


def _strip_units(val: list[float] | Quantity | NDArray) -> NDArray:
    if hasattr(val, "magnitude"):
        unitless_val = val.magnitude
    else:
        unitless_val = val

    return numpy.asarray(unitless_val).reshape((-3, 3))


CoordinateArray = Annotated[numpy.ndarray, BeforeValidator(_strip_units)]

Array = CoordinateArray
