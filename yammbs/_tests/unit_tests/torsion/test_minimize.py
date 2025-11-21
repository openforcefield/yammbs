import pytest
from openff.toolkit import Molecule
from yammbs.analysis import get_rmsd

from yammbs.torsion._minimize import (
    ConstrainedMinimizationInput,
    ConstrainedMinimizationResult,
    _minimize_constrained,
)


@pytest.fixture
def pentane_molecule() -> Molecule:
    """Create pentane molecule with a conformer."""
    pentane = Molecule.from_smiles("CCCCC")
    pentane.generate_conformers(n_conformers=1)
    return pentane


@pytest.fixture
def pentane_unrestrained_minimization_result(
    pentane_molecule,
) -> ConstrainedMinimizationResult:
    """Create pentane input with restrain_k set to 0.0."""
    min_input = ConstrainedMinimizationInput(
        torsion_id=100,
        mapped_smiles=pentane_molecule.to_smiles(mapped=True),
        dihedral_indices=[0, 1, 2, 3],
        force_field="openff-2.0.0",
        coordinates=pentane_molecule.conformers[0].m_as("angstrom"),
        grid_id=15.0,  # arbitrary
        restrain_k=0.0,
    )

    result = _minimize_constrained(min_input)

    return result


@pytest.fixture
def pentane_restrained_minimization_result(
    pentane_molecule,
) -> ConstrainedMinimizationResult:
    """Create pentane input with restrain_k set to 5.0."""
    min_input = ConstrainedMinimizationInput(
        torsion_id=100,
        mapped_smiles=pentane_molecule.to_smiles(mapped=True),
        dihedral_indices=[0, 1, 2, 3],
        force_field="openff-2.0.0",
        coordinates=pentane_molecule.conformers[0].m_as("angstrom"),
        grid_id=15.0,  # arbitrary
        restrain_k=5.0,
    )

    result = _minimize_constrained(min_input)

    return result


def test_minimization_basic(pentane_restrained_minimization_result):

    # these models don't track the (MM) energy before the (constrained) minimization
    # is there any reason to?

    # just check the minimization succeeded and produced non-NaN
    assert isinstance(pentane_restrained_minimization_result.energy, float)


def test_restraining_reduces_rmsd(
    pentane_molecule,
    pentane_unrestrained_minimization_result,
    pentane_restrained_minimization_result,
):
    """Test that restraints reduce coordinate deviation from initial structure."""
    reference_coords = pentane_molecule.conformers[0].m_as("angstrom")

    unrestrained_rmsd = get_rmsd(
        pentane_molecule,
        reference=reference_coords,
        target=pentane_unrestrained_minimization_result.coordinates,
    )
    restrained_rmsd = get_rmsd(
        pentane_molecule,
        reference=reference_coords,
        target=pentane_restrained_minimization_result.coordinates,
    )
    assert restrained_rmsd < unrestrained_rmsd
    print(
        f"Unrestrained RMSD: {unrestrained_rmsd:.4f} Å, Restrained RMSD: {restrained_rmsd:.4f} Å",
    )
