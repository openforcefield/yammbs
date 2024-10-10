"""
Helpers for dealing with Pydantic.

Originally copied from openff-nagl.
"""

import hashlib
import json
from typing import Any, ClassVar, Union

import numpy
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

FloatArrayLike = Union[list[float], NDArray[numpy.float64], float]


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

    _hash_fields: ClassVar[list[str] | None] = None
    _float_fields: ClassVar[list[str]] = list()
    _float_decimals: ClassVar[int] = 8

    _hash_int: int | None = None
    _hash_str: str | None = None

    def __init__(self, *args, **kwargs):
        self.__pre_init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.__post_init__(*args, **kwargs)

    def __pre_init__(self, *args, **kwargs):
        pass

    def __post_init__(self, *args, **kwargs):
        pass

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)

    def __hash__(self):
        if self._hash_int is None:
            mash = self.get_hash()
            self._hash_int = int(mash, 16)
        return self._hash_int

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_hash(self) -> str:
        """Returns string hash of the object"""
        if self._hash_str is None:
            dumped = self.dump_hashable(decimals=self._float_decimals)
            mash = hashlib.sha1()
            mash.update(dumped.encode("utf-8"))
            self._hash_str = mash.hexdigest()

        return self._hash_str

    def hash_dict(self) -> dict[str, Any]:
        """Create dictionary from hash fields and sort alphabetically"""
        if self._hash_fields:
            hashdct = self.model_dump(include=set(self._hash_fields))
        else:
            hashdct = self.model_dump()

        return {key: hashdct[key] for key in sorted(hashdct)}

    def dump_hashable(self, decimals: int | None = None):
        """
        Serialize object to a JSON formatted string

        Unlike model_dump_json(), this method only includes hashable fields,
        sorts them alphabetically, and optionally rounds floats.
        """
        data = self.hash_dict()

        if decimals is not None:
            for field in self._float_fields:
                if field in data:
                    data[field] = round_floats(data[field], decimals=decimals)

            with numpy.printoptions(precision=16):
                return json.dumps(data)

        return json.dumps(data)


class ImmutableModel(MutableModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        frozen=True,
    )
