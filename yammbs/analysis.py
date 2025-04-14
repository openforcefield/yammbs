from typing import TYPE_CHECKING

import numpy
from openff.toolkit import Molecule, Quantity
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import GetAngleDeg, GetBondLength, GetDihedralDeg

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


def _shift_angle(angle: float) -> float:
    """Shift angles by +/- 180 degrees if they're close to +/- 180."""
    # handle being close to both 180 and -180
    if abs(abs(angle) - 180) < 3:
        # subtract 180 if ~180, subtract -180 if ~-180
        return float(angle - numpy.sign(angle) * 180)
    else:
        # otherwise, return the angle as is
        return float(angle)


def _get_bond_length(
    conformers: tuple[Chem.Conformer, Chem.Conformer],
    atom1_index: int,
    atom2_index: int,
) -> tuple[float, float]:
    """Get the bond length between two atoms in a conformer."""
    return tuple(
        GetBondLength(
            conformer,
            atom1_index,
            atom2_index,
        )
        for conformer in conformers
    )


def _get_angle_angle(
    conformers: tuple[Chem.Conformer, Chem.Conformer],
    atom1_index: int,
    atom2_index: int,
    atom3_index: int,
) -> tuple[float, float]:
    """Get the angle, in degrees, between three atoms in a conformer."""
    return tuple(
        _shift_angle(
            GetAngleDeg(
                conformer,
                atom1_index,
                atom2_index,
                atom3_index,
            ),
        )
        for conformer in conformers
    )


def _get_dihedral_angle(
    conformers: tuple[Chem.Conformer, Chem.Conformer],
    atom1_index: int,
    atom2_index: int,
    atom3_index: int,
    atom4_index: int,
) -> tuple[float, float]:
    """Get the dihedral angle, in degrees, between four atoms in a conformer."""
    return tuple(
        _shift_angle(
            GetDihedralDeg(
                conformer,
                atom1_index,
                atom2_index,
                atom3_index,
                atom4_index,
            ),
        )
        for conformer in conformers
    )


def get_internal_coordinates(
    molecule: Molecule,
    reference: Array,
    target: Array,
    _types: tuple[str, ...] = ("Bond", "Angle", "Dihedral", "Improper"),
) -> dict[str, dict[tuple[int, ...], tuple[float, float]]]:
    """Get internal coordinates of two conformers using The RDKit.

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
    if isinstance(reference, Quantity):
        reference = reference.m_as("angstrom")

    if isinstance(target, Quantity):
        target = target.m_as("angstrom")

    molecule = Molecule(molecule)
    molecule.clear_conformers()
    molecule.add_conformer(Quantity(reference, "angstrom"))
    molecule.add_conformer(Quantity(target, "angstrom"))

    rdmol = molecule.to_rdkit()

    if isinstance(reference, Quantity):
        reference = target.m_as("angstrom")

    internal_coordinates: dict[str, dict[tuple[int, ...], tuple[float, float]]] = dict()

    if "Bond" in _types:
        internal_coordinates["Bond"] = {
            tuple((bond.atom1_index, bond.atom2_index)): _get_bond_length(
                (rdmol.GetConformer(0), rdmol.GetConformer(1)),
                bond.atom1_index,
                bond.atom2_index,
            )
            for bond in molecule.bonds
        }

    if "Angle" in _types:
        internal_coordinates["Angle"] = {
            tuple(
                (
                    angle[0].molecule_atom_index,
                    angle[1].molecule_atom_index,
                    angle[2].molecule_atom_index,
                ),
            ): _get_angle_angle(
                (rdmol.GetConformer(0), rdmol.GetConformer(1)),
                angle[0].molecule_atom_index,
                angle[1].molecule_atom_index,
                angle[2].molecule_atom_index,
            )
            for angle in molecule.angles
        }

    if "Dihedral" in _types:
        internal_coordinates["Dihedral"] = {
            # angle between i-j-k and j-k-l planes,
            tuple(
                (
                    dihedral[0].molecule_atom_index,
                    dihedral[1].molecule_atom_index,
                    dihedral[2].molecule_atom_index,
                    dihedral[3].molecule_atom_index,
                ),
            ): _get_dihedral_angle(
                (rdmol.GetConformer(0), rdmol.GetConformer(1)),
                dihedral[0].molecule_atom_index,
                dihedral[1].molecule_atom_index,
                dihedral[2].molecule_atom_index,
                dihedral[3].molecule_atom_index,
            )
            for dihedral in molecule.propers
        }

    if "Improper" in _types:
        internal_coordinates["Improper"] = {
            # angle between i-j-k and j-k-l planes, except the bonding is i-j, j-k, j-l
            # TODO: is trefoil averaging baked into the Molecule.smirnoff_impropers generator?
            tuple(
                (
                    improper[0].molecule_atom_index,
                    improper[1].molecule_atom_index,
                    improper[2].molecule_atom_index,
                    improper[3].molecule_atom_index,
                ),
            ): _get_dihedral_angle(
                (rdmol.GetConformer(0), rdmol.GetConformer(1)),
                improper[0].molecule_atom_index,
                improper[1].molecule_atom_index,  # central atom
                improper[2].molecule_atom_index,
                improper[3].molecule_atom_index,
            )
            for improper in molecule.smirnoff_impropers
        }

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
