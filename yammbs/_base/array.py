"""Helpers for dealing with NumPy arrays."""

from typing import Annotated

from numpy.typing import NDArray
from openff.toolkit import Quantity
from pydantic import BeforeValidator


def _strip_units(val: list[float] | Quantity | NDArray) -> list[float]:
    if hasattr(val, "magnitude"):
        return val.magnitude
    else:
        return val


Array = Annotated[list, BeforeValidator(_strip_units)]
