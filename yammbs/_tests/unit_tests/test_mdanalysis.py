import numpy
import pytest
from openff.toolkit import Molecule

from yammbs._mdanalysis import from_openff


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)O", "C1CC1N"])
def test_valences_match(smiles):
    molecule = Molecule.from_smiles(smiles)
    openff_bonds = molecule.bonds

    universe = from_openff(molecule)
    mdanalysis_bonds = universe.bonds

    assert len(openff_bonds) == len(mdanalysis_bonds)

    openff_array = numpy.sort(
        numpy.array(
            [[bond.atom1_index, bond.atom2_index] for bond in openff_bonds],
            dtype=numpy.int32,
        ),
        axis=0,
    )
    mdanalysis_array = numpy.sort(
        numpy.array(
            [bond.indices for bond in mdanalysis_bonds],
            dtype=numpy.int32,
        ),
        axis=0,
    )

    numpy.testing.assert_allclose(openff_array, mdanalysis_array)
