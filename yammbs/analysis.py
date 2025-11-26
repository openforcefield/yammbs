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
    reference: Array,
    target: Array,
    _types: tuple[str, ...] = ("Bond", "Angle", "Dihedral", "Improper"),
) -> dict[str, dict[tuple[int, ...], tuple[float, float]]]:
    """Get internal coordinates of two conformers of the same molecule using MDAnalysis.

    The return value is keyed by valence type (Bond, Angle, Dihedral, Improper). Each
    value is itself a dictionary containing key-val pairs of relevant atom indices and a
    2-length tuple of the internal coordinates. The first tuple is the internal coordinate
    of the reference conformer, the second is the internal coordinate of the target.

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
    dict[str, dict[tuple[int, ...], tuple[int, int]]]
        A dictionary of dictionaries containing the internal coordinates of the two
        conformers. The keys of the outer dictionary are the valence types (Bond, Angle,
        Dihedral, Improper). The keys of the inner dictionaries are tuples of atom
        indices. The values of the inner dictionaries are tuples of the internal
        coordinates, the first value corresponding to the "reference" conformer and the
        second to the "target" conformer.

    """
    from yammbs._mdanalysis import get_angles, get_bonds, get_improper_torsions, get_proper_torsions

    reference_molecule = Molecule(molecule)
    target_molecule = Molecule(molecule)

    reference_molecule.clear_conformers()
    target_molecule.clear_conformers()

    reference_molecule.add_conformer(Quantity(reference, "angstrom"))
    target_molecule.add_conformer(Quantity(target, "angstrom"))

    internal_coordinates: dict[str, dict[tuple[int, ...], tuple[float, float]]] = dict()

    key_func_mapping = {
        "Bond": get_bonds,
        "Angle": get_angles,
        "Dihedral": get_proper_torsions,
        "Improper": get_improper_torsions,
    }

    for _type in _types:
        try:
            reference_values = key_func_mapping[_type](reference_molecule)
            target_values = key_func_mapping[_type](target_molecule)

            assert reference_values.keys() == target_values.keys()

            internal_coordinates[_type] = dict()

            try:
                key = next(iter(reference_values.keys()))
            except StopIteration:
                # likely means there are no internal coordinates of this type,
                # i.e. water having no torsions
                continue

            internal_coordinates[_type][key] = (reference_values[key], target_values[key])

        except KeyError:
            raise ValueError(f"Unknown internal coordinate type: {_type}")

    return internal_coordinates


def get_internal_coordinate_differences(
    molecule: Molecule,
    reference: Array,
    target: Array,
    _types: tuple[str, ...] = ("Bond", "Angle", "Dihedral", "Improper"),
) -> dict[str, dict[tuple[int, ...], float]]:
    """Get internal coordinate differences between two conformers of one molecule.

    The behavior is identical to get_internal_coordinates, except that the return value
    is the difference in the relevant internal coordinate of each conformers, not the
    two values themselves.

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

    internal_coordinates = get_internal_coordinates(
        molecule=molecule,
        reference=reference,
        target=target,
        _types=_types,
    )

    for label, values_with_indices in internal_coordinates.items():
        differences[label] = dict()

        for indices, values in values_with_indices.items():
            differences[label][indices] = values[1] - values[0]

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

    internal_coordinates = get_internal_coordinates(
        molecule=molecule,
        reference=reference,
        target=target,
        _types=_types,
    )

    internal_coordinate_rmsd = dict()

    for _type, values_with_indices in internal_coordinates.items():
        if len(values_with_indices) == 0:
            continue

        _qm_values = [value[0] for value in values_with_indices.values()]
        _mm_values = [value[1] for value in values_with_indices.values()]

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
