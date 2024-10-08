"""Copied from openff-nagl."""

from typing import Any, List, Union

import numpy
from pydantic import BaseModel, ConfigDict

FloatArrayLike = Union[List, numpy.ndarray, float]


def round_floats(
    obj: FloatArrayLike,
    decimals: int = 8,
) -> FloatArrayLike:
    rounded = numpy.around(obj, decimals)
    threshold = 5 ** (1 - decimals)
    if isinstance(rounded, numpy.ndarray):
        rounded[numpy.abs(rounded) < threshold] = 0.0
    elif numpy.abs(rounded) < threshold:
        rounded = 0.0
    return rounded


class MutableModel(BaseModel):
    """
    Base class that all classes should subclass.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        frozen=False,
    )

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)


class ImmutableModel(MutableModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        frozen=True,
    )
