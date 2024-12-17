import numpy
from openff.toolkit import Molecule

from yammbs._molecule import _to_geometric_molecule


def test_to_geometric_molecule():
    molecule = Molecule.from_smiles("C1([H])CCCCN1")
    molecule.generate_conformers(n_conformers=1)

    geometric_molecule = _to_geometric_molecule(molecule, molecule.conformers[0].m)

    assert molecule.n_atoms == len(geometric_molecule.Data["elem"])
    assert molecule.n_bonds == len(geometric_molecule.Data["bonds"])

    assert numpy.allclose(
        molecule.conformers[0].m_as("angstrom"),
        geometric_molecule.Data["xyzs"][0],
    )
