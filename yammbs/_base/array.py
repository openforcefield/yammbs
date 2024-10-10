"""Helpers for dealing with NumPy arrays."""

from typing import Annotated

import numpy
from numpy.typing import NDArray
from openff.toolkit import Quantity
from pydantic import BeforeValidator


def _strip_units(val: Quantity | NDArray) -> NDArray:
    if hasattr(val, "magnitude"):
        return val.magnitude
    else:
        return val


Array = Annotated[numpy.ndarray, BeforeValidator(_strip_units)]
