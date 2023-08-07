import numpy
import pandas

from ibstore._base.array import Array
from ibstore._base.base import ImmutableModel


class DDE(ImmutableModel):
    qcarchive_id: str
    force_field: str
    difference: float


class DDECollection(list):
    def to_dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            [dde.difference for dde in self],
            index=pandas.Index([dde.qcarchive_id for dde in self]),
            columns=["difference"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)


class RMSD(ImmutableModel):
    qcarchive_id: str
    force_field: str
    rmsd: float


class RMSDCollection(list):
    def to_dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            [rmsd.rmsd for rmsd in self],
            index=pandas.Index([rmsd.qcarchive_id for rmsd in self]),
            columns=["rmsd"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)


def get_rmsd(
    reference: Array,
    target: Array,
) -> float:
    assert (
        reference.shape == target.shape
    ), "reference and target must have the same shape"

    return numpy.sqrt(numpy.sum((reference - target) ** 2) / len(reference))
