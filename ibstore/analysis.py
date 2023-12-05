import numpy
import pandas
from openff.toolkit import Molecule

from ibstore._base.array import Array
from ibstore._base.base import ImmutableModel


class DDE(ImmutableModel):
    qcarchive_id: int
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
    qcarchive_id: int
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


class TFD(ImmutableModel):
    qcarchive_id: int
    force_field: str
    tfd: float


class TFDCollection(list):
    def to_dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            [tfd.tfd for tfd in self],
            index=pandas.Index([tfd.qcarchive_id for tfd in self]),
            columns=["tfd"],
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
    return oechem.OERMSD(
        molecule1.to_openeye(),
        molecule2.to_openeye(),
        True,
        True,
        True,
    )


def _get_rmsd(
    reference: Array,
    target: Array,
) -> float:
    """Native, naive implementation of RMSD."""
    assert (
        reference.shape == target.shape
    ), "reference and target must have the same shape"

    return numpy.sqrt(numpy.sum((reference - target) ** 2) / len(reference))


def get_tfd(
    molecule: Molecule,
    reference: Array,
    target: Array,
) -> float:
    def _rdmol(
        molecule: Molecule,
        conformer: Array,
    ):
        from copy import deepcopy

        from openff.units import Quantity, unit

        # TODO: Do we need to remap indices?
        if False:
            # def _rdmol(inchi_key, mapped_smiles, ...)

            molecule = Molecule.from_inchi(inchi_key)  # noqa

            molecule_from_smiles = Molecule.from_mapped_smiles(mapped_smiles)  # noqa

            are_isomorphic, atom_map = Molecule.are_isomorphic(
                molecule,
                molecule_from_smiles,
                return_atom_map=True,
            )

            assert are_isomorphic, (
                "Molecules from InChi and mapped SMILES are not isomorphic:\n"
                f"\tinchi_key={inchi_key}\n"  # noqa
                f"\tmapped_smiles={mapped_smiles}"  # noqa
            )

            molecule.remap(mapping_dict=atom_map)

        molecule = deepcopy(molecule)

        molecule.add_conformer(
            Quantity(conformer, unit.angstrom),
        )

        return molecule.to_rdkit()

    from rdkit.Chem import TorsionFingerprints

    return TorsionFingerprints.GetTFDBetweenMolecules(
        _rdmol(molecule, reference),
        _rdmol(molecule, target),
    )
