import MDAnalysis
from openff.toolkit import Molecule

def from_openff(molecule: Molecule) -> MDAnalysis.Universe:
    return MDAnalysis.Universe(molecule.to_rdkit())

def get_angles(molecule: Molecule) -> dict[tuple[int, int, int], float]:
    assert molecule.n_conformers == 1, "Need a conformer for coordinate analysis"

    universe = from_openff(molecule)

    result = dict()

    for angle in molecule.angles:
        atom_indices = [molecule.atom_index(atom) for atom in angle]
        angle_coords = universe.coord.positions[atom_indices]
        # TODO: would this be faster if we gave it all angles at once?
        result[tuple(atom_indices)] =  MDAnalysis.lib.distances.calc_angles(*angle_coords)

    return result