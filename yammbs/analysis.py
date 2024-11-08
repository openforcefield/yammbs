from typing import TYPE_CHECKING

import numpy
from openff.toolkit import Molecule
from openff.units import Quantity, unit

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel

if TYPE_CHECKING:
    from pandas import DataFrame


class DDE(ImmutableModel):
    qcarchive_id: int
    difference: float | None


class DDECollection(list[DDE]):
    def to_dataframe(self) -> "DataFrame":
        import pandas

        return pandas.DataFrame(
            [dde.difference for dde in self],
            index=pandas.Index([dde.qcarchive_id for dde in self]),
            columns=["difference"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)


class RMSD(ImmutableModel):
    qcarchive_id: int
    rmsd: float


class RMSDCollection(list[RMSD]):
    def to_dataframe(self) -> "DataFrame":
        import pandas

        return pandas.DataFrame(
            [rmsd.rmsd for rmsd in self],
            index=pandas.Index([rmsd.qcarchive_id for rmsd in self]),
            columns=["rmsd"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)


class ICRMSD(ImmutableModel):
    qcarchive_id: int
    icrmsd: dict[str, float]


class ICRMSDCollection(list):
    def to_dataframe(self) -> "DataFrame":
        import pandas

        return pandas.DataFrame(
            [
                (
                    icrmsd.icrmsd["Bond"],
                    icrmsd.icrmsd["Angle"],
                    icrmsd.icrmsd.get("Dihedral", pandas.NA),
                    icrmsd.icrmsd.get("Improper", pandas.NA),
                )
                for icrmsd in self
            ],
            index=pandas.Index([rmsd.qcarchive_id for rmsd in self]),
            columns=["Bond", "Angle", "Dihedral", "Improper"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)


class TFD(ImmutableModel):
    qcarchive_id: int
    tfd: float


class TFDCollection(list):
    def to_dataframe(self) -> "DataFrame":
        import pandas

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
    molecule2 = Molecule(molecule)

    for molecule in (molecule1, molecule2):
        if molecule.conformers is not None:
            molecule.conformers.clear()

    molecule1.add_conformer(Quantity(reference, unit.angstrom))  # type: ignore[call-overload]

    molecule2.add_conformer(Quantity(target, unit.angstrom))  # type: ignore[call-overload]

    # oechem appears to not support named arguments, but it's hard to tell
    # since the Python API is not documented
    return oechem.OERMSD(
        molecule1.to_openeye(),
        molecule2.to_openeye(),
        True,
        True,
        True,
    )


def get_internal_coordinate_rmsds(
    molecule: Molecule,
    reference: Array,
    target: Array,
    _types: tuple[str, ...] = ("Bond", "Angle", "Dihedral", "Improper"),
) -> dict[str, float]:
    """Get internal coordinate RMSDs for one conformer of one molecule."""
    from geometric.internal import (
        Angle,
        Dihedral,
        Distance,
        OutOfPlane,
        PrimitiveInternalCoordinates,
    )

    from yammbs._forcebalance import compute_rmsd as forcebalance_rmsd
    from yammbs._molecule import _to_geometric_molecule

    if isinstance(reference, Quantity):
        reference = reference.m_as(unit.angstrom)

    if isinstance(target, Quantity):
        target = target.m_as(unit.angstrom)

    _generator = PrimitiveInternalCoordinates(
        _to_geometric_molecule(molecule=molecule, coordinates=target),
    )

    types: dict[str, type | None] = {
        _type: {
            "Bond": Distance,
            "Angle": Angle,
            "Dihedral": Dihedral,
            "Improper": OutOfPlane,
        }.get(_type)
        for _type in _types
    }

    internal_coordinates = {
        label: [
            (
                internal_coordinate.value(reference),
                internal_coordinate.value(target),
            )
            for internal_coordinate in _generator.Internals
            if isinstance(internal_coordinate, internal_coordinate_class)  # type: ignore[arg-type]
        ]
        for label, internal_coordinate_class in types.items()
    }

    internal_coordinate_rmsd = dict()

    for _type, _values in internal_coordinates.items():
        if len(_values) == 0:
            continue

        _qm_values, _mm_values = zip(*_values)

        qm_values = numpy.array(_qm_values)
        mm_values = numpy.array(_mm_values)

        # Converting from radians to degrees
        if _type in ["Angle", "Dihedral", "Improper"]:
            rmsd = forcebalance_rmsd(
                qm_values * 180 / numpy.pi,
                mm_values * 180 / numpy.pi,
                360,
            )
        else:
            rmsd = forcebalance_rmsd(qm_values, mm_values)

        internal_coordinate_rmsd[_type] = rmsd

    return internal_coordinate_rmsd


def get_tfd(
    molecule: Molecule,
    reference: Array,
    target: Array,
) -> float:
    def _rdmol(
        molecule: Molecule,
        conformer: Array,
    ):
        from openff.units import Quantity, unit

        molecule = Molecule(molecule)
        if molecule.conformers is not None:
            molecule.conformers.clear()

        molecule.add_conformer(Quantity(conformer, unit.angstrom))  # type: ignore[call-overload]

        return molecule.to_rdkit()

    from rdkit.Chem import TorsionFingerprints

    return TorsionFingerprints.GetTFDBetweenMolecules(
        _rdmol(molecule, reference),
        _rdmol(molecule, target),
    )
