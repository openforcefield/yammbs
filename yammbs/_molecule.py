"""Molecule conversion utilities"""

from typing import TYPE_CHECKING

from openff.toolkit import Molecule, Quantity

from yammbs._base.array import Array

if TYPE_CHECKING:
    from geometric.molecule import Molecule as GeometricMolecule


def _to_geometric_molecule(
    molecule: Molecule,
    coordinates: Array,
) -> "GeometricMolecule":
    from geometric.molecule import Molecule as GeometricMolecule

    geometric_molecule = GeometricMolecule()

    geometric_molecule.Data = {
        "resname": ["UNK"] * molecule.n_atoms,
        "resid": [0] * molecule.n_atoms,
        "elem": [atom.symbol for atom in molecule.atoms],
        "bonds": [(bond.atom1_index, bond.atom2_index) for bond in molecule.bonds],
        "name": molecule.name,
        "xyzs": [coordinates],
    }

    return geometric_molecule


def _molecule_with_conformer_from_smiles(
    mapped_smiles: str,
    conformer: Array,
) -> Molecule:
    """Create a molecule from mapped SMILES and attach a single conformer."""
    molecule = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)
    molecule.add_conformer(Quantity(conformer, "angstrom"))

    return molecule
