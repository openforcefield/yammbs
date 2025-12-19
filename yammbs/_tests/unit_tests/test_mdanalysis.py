import pytest

import numpy

from yammbs._mdanalysis import from_openff
from openff.toolkit.topology.molecule import Bond
from openff.toolkit import Molecule
from MDAnalysis.core.topologyobjects import TopologyGroup
def assert_bonds_match(
    openff_bonds: list[Bond],
    mdanalysis_bonds: TopologyGroup,
):
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
            [bond.indices for bond in mdanalysis_bonds], dtype=numpy.int32,
        ),
        axis=0,
    )

    numpy.testing.assert_allclose(openff_array, mdanalysis_array)



@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)O", "C1CC1N"])
def test_valences_match(smiles):
    molecule = Molecule.from_smiles(smiles)

    universe = from_openff(molecule)

    assert_bonds_match(molecule.bonds, universe.bonds)

# For each valence term (% confusion about impropers)
#  * Make sure basic behavior matches OpenFF <-> MDAnalysi
#  * Make sure bad conformers don't change graphs
