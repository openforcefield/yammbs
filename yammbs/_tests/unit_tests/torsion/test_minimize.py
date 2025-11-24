import pytest
from openff.toolkit import Molecule
import numpy as np
import openmm

from yammbs.torsion._minimize import (
    ConstrainedMinimizationInput,
    ConstrainedMinimizationResult,
    _run_minimization_constrained,
)


@pytest.fixture
def ethane() -> Molecule:
    """Return an ethane with a bad conformer."""
    ethane = Molecule.from_smiles("[H:3][C:1]([H:4])([H:5])[C:2]([H:6])([H:7])[H:8]")

    positions = (
        np.array(
            [
                [0.24658203, -0.60449219, 0.3815918],
                [-0.24658203, 0.60400391, -0.38183594],
                [1.33798838, -0.59356904, 0.45473218],
                [-0.0564872, -1.52736127, -0.12180027],
                [-0.16626501, -0.61344373, 1.39503479],
                [-1.33789062, 0.59326172, -0.4550781],
                [0.16674803, 0.61376953, -1.39453125],
                [0.05578615, 1.52734375, 0.12237547],
            ],
        )
        * openmm.unit.angstrom
    )

    ethane.add_conformer(positions)

    return ethane


@pytest.fixture
def ethane_openmm_result() -> ConstrainedMinimizationResult:
    ethane = Molecule.from_smiles("CC")
    ethane.generate_conformers(n_conformers=1)

    min_input = ConstrainedMinimizationInput(
        torsion_id=100,
        mapped_smiles=ethane.to_smiles(mapped=True),
        dihedral_indices=[2, 0, 1, 5],
        force_field="openff-2.0.0",
        coordinates=ethane.conformers[0].m_as("angstrom"),
        grid_id=180.0,  # arbitrary
        method="openmm",
    )

    return _run_minimization_constrained(min_input)


@pytest.fixture
def ethane_geometric_result() -> ConstrainedMinimizationResult:
    ethane = Molecule.from_smiles("CC")
    ethane.generate_conformers(n_conformers=1)

    min_input = ConstrainedMinimizationInput(
        torsion_id=100,
        mapped_smiles=ethane.to_smiles(mapped=True),
        dihedral_indices=[2, 0, 1, 5],
        force_field="openff-2.0.0",
        coordinates=ethane.conformers[0].m_as("angstrom"),
        grid_id=180.0,  # arbitrary
        method="geometric",
    )

    return _run_minimization_constrained(min_input)


def test_minimization_basic_openmm(ethane_openmm_result):

    # these models don't track the (MM) energy before the (constrained) minimization
    # is there any reason to?

    # just check the minimization succeeded and produced non-NaN
    assert isinstance(ethane_openmm_result.energy, float)


def test_minimization_basic_geometric(ethane_geometric_result):
    assert isinstance(ethane_geometric_result.energy, float)


def test_minimizations_similar_results(
    ethane_openmm_result,
    ethane_geometric_result,
):
    """Check that the two minimization methods give similar results."""
    energy_diff = abs(ethane_openmm_result.energy - ethane_geometric_result.energy)
    assert energy_diff < 0.1  # kcal/mol
