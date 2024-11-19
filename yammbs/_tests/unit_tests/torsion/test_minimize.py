import pytest
from openff.toolkit import Molecule

from yammbs.torsion._minimize import ConstrainedMinimizationInput, _minimize_constrained


@pytest.fixture
def ethane_input() -> ConstrainedMinimizationInput:
    ethane = Molecule.from_smiles("CC")
    ethane.generate_conformers(n_conformers=1)

    return ConstrainedMinimizationInput(
        mapped_smiles=ethane.to_smiles(mapped=True),
        dihedral_indices=[0, 1, 2, 3],
        force_field="openff-2.0.0",
        coordinates=ethane.conformers[0].m_as("angstrom"),
        grid_id=15.0,  # arbitrary
    )


def test_minimization_basic(ethane_input):
    result = _minimize_constrained(ethane_input)

    # these models don't track the (MM) energy before the (constrained) minimization
    # is there any reason to?

    # just check the minimization succeeded and produced non-NaN
    assert isinstance(result.energy, float)
