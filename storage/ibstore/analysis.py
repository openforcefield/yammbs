import numpy
import pandas
from openff.toolkit import Molecule

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
    molecule: Molecule,
    reference: Array,
    target: Array,
) -> float:
    """Compute the RMSD between two sets of coordinates."""
    from openeye import oechem
    from openff.units import Quantity, unit

    molecule1 = Molecule(molecule)
    molecule1.add_conformer(Quantity(reference, unit.angstrom))

    molecule2 = Molecule(molecule)
    molecule2.add_conformer(Quantity(target, unit.angstrom))

    # oechem appears to not support named arguments, but it's hard to tell
    # since the Python API is not documented
    return oechem.OERMSD(molecule1.to_openeye(), molecule2.to_openeye(), True, True, True,)

def _get_rmsd(
    reference: Array,
    target: Array,
) -> float:
    """Native, naive implementation of RMSD."""
    assert (
        reference.shape == target.shape
    ), "reference and target must have the same shape"

    return numpy.sqrt(numpy.sum((reference - target) ** 2) / len(reference))
