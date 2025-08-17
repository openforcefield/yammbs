"""Analysis routines for optimizations."""

from typing import TYPE_CHECKING

import numpy
from openff.toolkit import Molecule, Quantity

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel

if TYPE_CHECKING:
    from pandas import DataFrame


class DDE(ImmutableModel):
    """A model containing relative energy differences (DDEs) of a molecule."""

    qcarchive_id: int
    difference: float | None


class DDECollection(list[DDE]):
    """A model containing DDEs of a dataset of molecules."""

    def to_dataframe(self) -> "DataFrame":
        """Convert the collection to a pandas DataFrame."""
        import pandas

        return pandas.DataFrame(
            [dde.difference for dde in self],
            index=pandas.Index([dde.qcarchive_id for dde in self]),
            columns=["difference"],
        )

    def to_csv(self, path: str):
        """Write the collection to a CSV file."""
        self.to_dataframe().to_csv(path)


class RMSD(ImmutableModel):
    """A model containing the root-mean-square deviations of conformers a molecule."""

    qcarchive_id: int
    rmsd: float


class RMSDCollection(list[RMSD]):
    """A model containing RMSDs of a dataset of molecules."""

    def to_dataframe(self) -> "DataFrame":
        """Convert the collection to a pandas DataFrame."""
        import pandas

        return pandas.DataFrame(
            [rmsd.rmsd for rmsd in self],
            index=pandas.Index([rmsd.qcarchive_id for rmsd in self]),
            columns=["rmsd"],
        )

    def to_csv(self, path: str):
        """Write the collection to a CSV file."""
        self.to_dataframe().to_csv(path)


class ICRMSD(ImmutableModel):
    """A model containing the internal coordinate RMSDs of a molecule."""

    qcarchive_id: int
    icrmsd: dict[str, float]


class ICRMSDCollection(list):
    """A model containing internal coordinate RMSDs of a dataset of molecules."""

    def to_dataframe(self) -> "DataFrame":
        """Convert the collection to a pandas DataFrame."""
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
        """Write the collection to a CSV file."""
        self.to_dataframe().to_csv(path)


class TFD(ImmutableModel):
    """A model containing the TFD of a molecule."""

    qcarchive_id: int
    tfd: float


class TFDCollection(list):
    """A model containing TFDs of a collection of molecules."""

    def to_dataframe(self) -> "DataFrame":
        """Convert the collection to a pandas DataFrame."""
        import pandas

        return pandas.DataFrame(
            [tfd.tfd for tfd in self],
            index=pandas.Index([tfd.qcarchive_id for tfd in self]),
            columns=["tfd"],
        )

    def to_csv(self, path: str):
        """Write the collection to a CSV file."""
        self.to_dataframe().to_csv(path)


def get_rmsd(
    molecule: Molecule,
    reference: Array,
    target: Array,
) -> float:
    """Compute the RMSD between two sets of coordinates."""
    from openeye import oechem

    molecule1 = Molecule(molecule)
    molecule2 = Molecule(molecule)

    for molecule in (molecule1, molecule2):
        if molecule.conformers is not None:
            molecule.conformers.clear()

    molecule1.add_conformer(Quantity(reference, "angstrom"))

    molecule2.add_conformer(Quantity(target, "angstrom"))

    # oechem appears to not support named arguments, but it's hard to tell
    # since the Python API is not documented
    return oechem.OERMSD(
        molecule1.to_openeye(),
        molecule2.to_openeye(),
        True,
        True,
        True,
    )


def get_internal_coordinates(
    molecule: Molecule,
    coordinates: Array,
    _types: tuple[str, ...] = ("Bond", "Angle", "Dihedral", "Improper"),
) -> dict[str, dict[tuple[int, ...], float]]:
    """Get internal coordinates of one conformers of a molecule using geomeTRIC.

    The return value is keyed by valence type (Bond, Angle, Dihedral, Improper). Each
    value is itself a dictionary containing key-val pairs of relevant atom indices and
    the corresponding "internal coordinate."

    The conformer(s) attached to the `molecule` argument are ignored, only the values in
    the `reference` and `target` arguments are used. The `molecule` argument is only used
    to determine the atom indices of the internal coordinates.

    Parameters
    ----------
    molecule : openff.toolkit.Molecule
        The molecule to get the internal coordinates for.
    reference : numpy.ndarray or openff.toolkit.Quantity
        The "reference" conformer to get the internal coordinates for. If unitless,
        assumed to be in Angstroms.
    target : numpy.ndarray or openff.toolkit.Quantity
        The "target" conformer to get the internal coordinates for. If unitless, assumed
        to be in Angstroms.

    Returns
    -------
    dict[str, dict[tuple[int, ...], float]]
        A dictionary of dictionaries containing the internal coordinates of the two
        conformers. The keys of the outer dictionary are the valence types (Bond, Angle,
        Dihedral, Improper). The keys of the inner dictionaries are tuples of atom
        indices. The values of the inner dictionaries are the corresponding internal
        coordinates.

    """
    from geometric.internal import (
        Angle,
        Dihedral,
        Distance,
        OutOfPlane,
        PrimitiveInternalCoordinates,
    )

    from yammbs._molecule import _to_geometric_molecule

    if isinstance(coordinates, Quantity):
        coordinates = coordinates.m_as("angstrom")

    _generator = PrimitiveInternalCoordinates(
        _to_geometric_molecule(molecule=molecule, coordinates=coordinates),
    )

    _mapping = {
        "Bond": Distance,
        "Angle": Angle,
        "Dihedral": Dihedral,
        "Improper": OutOfPlane,
    }
    types: dict[str, type] = {_type: _mapping[_type] for _type in _types}

    internal_coordinates: dict[str, dict[tuple[int, ...], float]] = dict()

    for label, internal_coordinate_class in types.items():
        internal_coordinates[label] = dict()

        for internal_coordinate in _generator.Internals:
            if not isinstance(internal_coordinate, internal_coordinate_class):
                continue

            elif isinstance(internal_coordinate, Distance):
                key = tuple(
                    (
                        internal_coordinate.a,
                        internal_coordinate.b,
                    ),
                )

            elif isinstance(internal_coordinate, Angle):
                key = tuple(
                    (
                        internal_coordinate.a,
                        internal_coordinate.b,
                        internal_coordinate.c,
                    ),
                )

            elif isinstance(internal_coordinate, Dihedral):
                key = tuple(
                    (
                        internal_coordinate.a,
                        internal_coordinate.b,
                        internal_coordinate.c,
                        internal_coordinate.d,
                    ),
                )

            elif isinstance(internal_coordinate, OutOfPlane):
                # geomeTRIC lists the central atom FIRST, but SMIRNOFF force fields list
                # the central atom SECOND. Re-ordering here to be consistent with SMIRNOFF
                # see PR #109 for more

                key = tuple(
                    (
                        internal_coordinate.b,  # NOTE!
                        internal_coordinate.a,  # NOTE!
                        internal_coordinate.c,
                        internal_coordinate.d,
                    ),
                )

            internal_coordinates[label].update(
                {tuple(int(index) for index in key): float(internal_coordinate.value(coordinates))},
            )

    return internal_coordinates


def get_internal_coordinate_differences(
    molecule: Molecule,
    reference: Array,
    target: Array,
    _types: tuple[str, ...] = ("Bond", "Angle", "Dihedral", "Improper"),
) -> dict[str, dict[tuple[int, ...], float]]:
    """Get internal coordinate differences between two conformers of one molecule.

    The behavior is identical to get_internal_coordinates, except that the return value
    is the difference in the relevant internal coordinate of each conformer, not the
    value of one conformer.

    The conformers attached to the `molecule` argument are ignored, only the values in
    the `reference` and `target` arguments are used. The `molecule` argument is only used
    to determine the atom indices of the internal coordinates.

    Parameters
    ----------
    molecule : openff.toolkit.Molecule
        The molecule to get the internal coordinates for.
    reference : numpy.ndarray or openff.toolkit.Quantity
        The "reference" conformer to get the internal coordinates for. If unitless,
        assumed to be in Angstroms.
    target : numpy.ndarray or openff.toolkit.Quantity
        The "target" conformer to get the internal coordinates for. If unitless, assumed
        to be in Angstroms.

    Returns
    -------
    dict[str, dict[tuple[int, ...], float]]
        A dictionary of dictionaries containing the internal coordinates of the two
        conformers. The keys of the outer dictionary are the valence types (Bond, Angle,
        Dihedral, Improper). The keys of the inner dictionaries are tuples of atom
        indices. The values of the inner dictionaries are differences of the internal
        coordinates between the "target" and "reference" conformer.

    """
    differences: dict[str, dict[tuple[int, ...], float]] = dict()

    (
        reference_coordinates,
        target_coordinates,
    ) = (
        get_internal_coordinates(
            molecule=molecule,
            coordinates=coordinates,
            _types=_types,
        )
        for coordinates in [reference, target]
    )

    # the keys of each dictionary (and nested dictionaries) should be the same
    for label, values_with_indices in reference_coordinates.items():
        differences[label] = dict()

        for indices, value in values_with_indices.items():
            differences[label][indices] = target_coordinates[label][indices] - value

    return differences


def get_internal_coordinate_rmsds(
    molecule: Molecule,
    reference: Array,
    target: Array,
    _types: tuple[str, ...] = ("Bond", "Angle", "Dihedral", "Improper"),
) -> dict[str, float]:
    """Get internal coordinate RMSDs between two conformers of one molecule.

    The behavior is identical to get_internal_coordinates, except that the return value
    is the RMSD of all internal coordinate differences of a particular type.

    The conformers attached to the `molecule` argument are ignored, only the values in
    the `reference` and `target` arguments are used. The `molecule` argument is only
    used to determine the atom indices of the internal coordinates.

    Parameters
    ----------
    molecule : openff.toolkit.Molecule
        The molecule to get the internal coordinates for.
    reference : numpy.ndarray or openff.toolkit.Quantity
        The "reference" conformer to get the internal coordinates for. If unitless,
        assumed to be in Angstroms.
    target : numpy.ndarray or openff.toolkit.Quantity
        The "target" conformer to get the internal coordinates for. If unitless, assumed
        to be in Angstroms.

    Returns
    -------
    dict[str, float]
        A dictionary containing the internal coordinate RMSDs between the two
        conformers, averaged over all internal coordinates of a given type, keyed by
        the name of that type.

    """
    from yammbs._forcebalance import compute_rmsd as forcebalance_rmsd

    (
        reference_coordinates,
        target_coordinates,
    ) = (
        get_internal_coordinates(
            molecule=molecule,
            coordinates=coordinates,
            _types=_types,
        )
        for coordinates in [reference, target]
    )

    internal_coordinate_rmsd = dict()

    for _type in reference_coordinates:
        qm_values = numpy.array([*reference_coordinates[_type].values()])
        mm_values = numpy.array([*target_coordinates[_type].values()])

        if len(qm_values) * len(mm_values) == 0:
            continue

        if len(qm_values) != len(mm_values):
            continue

        # Converting from radians to degrees
        if _type in ["Angle", "Dihedral", "Improper"]:
            rmsd = forcebalance_rmsd(
                qm_values * 180 / numpy.pi,
                mm_values * 180 / numpy.pi,
                360,
            )
        else:
            rmsd = forcebalance_rmsd(qm_values, mm_values)

        internal_coordinate_rmsd[_type] = float(rmsd)

    return internal_coordinate_rmsd


def get_tfd(
    molecule: Molecule,
    reference: Array,
    target: Array,
) -> float:
    """Get the torsion fingerprint deviation between two conformers of one molecule."""

    def _rdmol(
        molecule: Molecule,
        conformer: Array,
    ):
        molecule = Molecule(molecule)
        if molecule.conformers is not None:
            molecule.conformers.clear()

        molecule.add_conformer(Quantity(conformer, "angstrom"))

        return molecule.to_rdkit()

    from rdkit.Chem import TorsionFingerprints

    return TorsionFingerprints.GetTFDBetweenMolecules(
        _rdmol(molecule, reference),
        _rdmol(molecule, target),
    )
