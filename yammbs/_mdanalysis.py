import MDAnalysis
from openff.toolkit import Molecule


# TODO: Caching might be useful, but needs to account for two molecules having
#       identical (mapped) SMILES but (crucially!) different conformers.
#       If implementing this, make sure to use MAPPED SMILES as a key, which might
#       require defining another function
def from_openff(molecule: Molecule) -> MDAnalysis.Universe:
    return MDAnalysis.Universe(molecule.to_rdkit())


def get_bonds(molecule: Molecule) -> dict[tuple[int, int], float]:
    assert molecule.n_conformers == 1, "Need a conformer for coordinate analysis"

    universe = from_openff(molecule)

    result = dict()

    for bond in molecule.bonds:
        atom_indices = [bond.atom1_index, bond.atom2_index]
        bond_coords = universe.coord.positions[atom_indices]
        # Process angle using MDAnalysis
        result[tuple(atom_indices)] = MDAnalysis.lib.distances.calc_bonds(*bond_coords)

    return result


def get_angles(molecule: Molecule) -> dict[tuple[int, int, int], float]:
    assert molecule.n_conformers == 1, "Need a conformer for coordinate analysis"

    universe = from_openff(molecule)

    result = dict()

    for angle in molecule.angles:
        atom_indices = [molecule.atom_index(atom) for atom in angle]
        angle_coords = universe.coord.positions[atom_indices]
        # TODO: would this be faster if we gave it all angles at once?
        result[tuple(atom_indices)] = MDAnalysis.lib.distances.calc_angles(*angle_coords)

    return result


def get_proper_torsions(molecule: Molecule) -> dict[tuple[int, int, int, int], float]:
    assert molecule.n_conformers == 1, "Need a conformer for coordinate analysis"

    universe = from_openff(molecule)

    result = dict()

    for dihedral in molecule.propers:
        atom_indices = [molecule.atom_index(atom) for atom in dihedral]
        dihedral_coords = universe.coord.positions[atom_indices]
        result[tuple(atom_indices)] = MDAnalysis.lib.distances.calc_dihedrals(*dihedral_coords)

    return result


def get_improper_torsions(molecule: Molecule) -> dict[tuple[int, int, int, int], float]:
    assert molecule.n_conformers == 1, "Need a conformer for coordinate analysis"

    universe = from_openff(molecule)

    result = dict()

    for dihedral in molecule.smirnoff_impropers:
        # central atom is SECOND
        atom_indices = [molecule.atom_index(atom) for atom in dihedral]

        # calc_dihedrals builds planes using i-j-k and j-k-l, and following that definition
        # the central atom must be first (i)
        dihedral_coords = universe.coord.positions[
            [
                atom_indices[1],
                atom_indices[0],
                atom_indices[2],
                atom_indices[3],
            ]
        ]

        # but don't re-order when bookkeeping
        result[tuple(atom_indices)] = MDAnalysis.lib.distances.calc_dihedrals(*dihedral_coords)

    return result
