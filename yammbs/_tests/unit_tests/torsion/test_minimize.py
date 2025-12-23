import pytest
from openff.toolkit import Molecule
import numpy as np
import openmm
from yammbs.analysis import get_rmsd
import copy

from yammbs.torsion._minimize import (
    ConstrainedMinimizationInput,
    ConstrainedMinimizationResult,
    _run_minimization_constrained,
)


@pytest.fixture
def pentane_molecule() -> Molecule:
    """Return an ethane with one conformer."""
    pentane = Molecule.from_mapped_smiles(
        "[C:1]([H:6])([H:7])([H:8])[C:2]([H:9])([H:10])[C:3]([H:11])([H:12])[C:4]([H:13])([H:14])[C:5]([H:15])([H:16])[H:17]",
    )

    positions = (
        np.array(
            [
                [-0.91508573, 0.29544935, -2.34612298],
                [-0.71728516, -0.49414062, -1.06152344],
                [0.11737061, 0.28637695, -0.04742432],
                [0.31494141, -0.50927734, 1.24316406],
                [1.13239932, 0.24814101, 2.24388671],
                [-1.51401126, -0.27057582, -3.0674386],
                [-1.43489695, 1.23460948, -2.13688302],
                [0.0461266, 0.53576541, -2.81141567],
                [-1.6965872, -0.73367834, -0.63261503],
                [-0.224279, -1.44469547, -1.29724395],
                [-0.3737793, 1.24023438, 0.18041992],
                [1.09472656, 0.53173828, -0.48168945],
                [0.8140077, -1.46017945, 1.02239442],
                [-0.65867734, -0.74810833, 1.68678439],
                [1.26303411, -0.33726859, 3.16079259],
                [0.6256721, 1.18306684, 2.50701475],
                [2.11578083, 0.49720106, 1.83249044],
            ],
        )
        * openmm.unit.angstrom
    )

    pentane.add_conformer(positions)

    return pentane


@pytest.fixture
def base_minimization_input(pentane_molecule) -> ConstrainedMinimizationInput:
    return ConstrainedMinimizationInput(
        torsion_id=100,  # Arbitrary
        mapped_smiles=pentane_molecule.to_smiles(mapped=True),
        dihedral_indices=[0, 1, 2, 3],
        force_field="openff-2.0.0",
        coordinates=pentane_molecule.conformers[0].m_as("angstrom"),
        grid_id=180.0,
        method="openmm",
    )


@pytest.fixture
def pentane_openmm_unrestrained_result(
    pentane_molecule, base_minimization_input,
) -> ConstrainedMinimizationResult:

    return _run_minimization_constrained(base_minimization_input)


@pytest.fixture
def pentane_geometric_unrestrained_result(
    pentane_molecule, base_minimization_input,
) -> ConstrainedMinimizationResult:

    min_input_dict = base_minimization_input.model_dump()
    min_input_dict["method"] = "geometric"

    return _run_minimization_constrained(ConstrainedMinimizationInput(**min_input_dict))


@pytest.fixture
def pentane_openmm_restrained_result(
    pentane_molecule, base_minimization_input,
) -> ConstrainedMinimizationResult:

    min_input = copy.deepcopy(base_minimization_input)
    min_input_dict = base_minimization_input.model_dump()
    min_input_dict["restraint_k"] = 1.0  # apply restraints

    return _run_minimization_constrained(ConstrainedMinimizationInput(**min_input_dict))


@pytest.fixture
def pentane_geometric_restrained_result(
    pentane_molecule, base_minimization_input,
) -> ConstrainedMinimizationResult:

    min_input_dict = base_minimization_input.model_dump()
    min_input_dict["method"] = "geometric"
    min_input_dict["restraint_k"] = 1.0  # apply restraints

    return _run_minimization_constrained(ConstrainedMinimizationInput(**min_input_dict))


def test_minimization_basic_openmm(pentane_openmm_unrestrained_result):
    # these models don't track the (MM) energy before the (constrained) minimization
    # is there any reason to?

    # just check the minimization succeeded and produced non-NaN
    assert isinstance(pentane_openmm_unrestrained_result.energy, float)


def test_minimization_basic_geometric(pentane_geometric_unrestrained_result):
    assert isinstance(pentane_geometric_unrestrained_result.energy, float)


def test_minimizations_similar_results(
    pentane_openmm_unrestrained_result,
    pentane_geometric_unrestrained_result,
):
    """Check that the two minimization methods give similar results."""
    energy_diff = abs(
        pentane_openmm_unrestrained_result.energy
        - pentane_geometric_unrestrained_result.energy,
    )
    assert energy_diff < 0.1  # kcal/mol


@pytest.mark.parametrize("method", ["openmm", "geometric"])
def test_restraining_reduces_rmsd(
    method,
    pentane_molecule,
    pentane_openmm_restrained_result,
    pentane_geometric_restrained_result,
    pentane_openmm_unrestrained_result,
    pentane_geometric_unrestrained_result,
):
    """Test that restraints reduce coordinate deviation from initial structure."""
    reference_coords = pentane_molecule.conformers[0].m_as("angstrom")

    if method == "openmm":
        pentane_restrained_minimization_result = pentane_openmm_restrained_result
        pentane_unrestrained_minimization_result = pentane_openmm_unrestrained_result
    else:
        pentane_restrained_minimization_result = pentane_geometric_restrained_result
        pentane_unrestrained_minimization_result = pentane_geometric_unrestrained_result

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
