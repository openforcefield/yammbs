"""Tests torsion minimization."""

import pytest
from openff.toolkit import ForceField, Molecule

from yammbs.analysis import get_rmsd
from yammbs.torsion._minimize import (
    ConstrainedMinimizationInput,
    ConstrainedMinimizationResult,
    _minimize_constrained,
)
from yammbs.torsion._store import TorsionStore
from yammbs.torsion.inputs import QCArchiveTorsionDataset, QCArchiveTorsionProfile


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
    """Create pentane input with restraint_k set to 0.0."""
    min_input = ConstrainedMinimizationInput(
        torsion_id=100,
        mapped_smiles=pentane_molecule.to_smiles(mapped=True),
        dihedral_indices=[0, 1, 2, 3],
        force_field="openff-2.0.0",
        coordinates=pentane_molecule.conformers[0].m_as("angstrom"),
        grid_id=15.0,  # arbitrary
        restraint_k=0.0,
    )

    result = _minimize_constrained(min_input)

    return result


@pytest.fixture
def pentane_restrained_minimization_result(
    pentane_molecule,
) -> ConstrainedMinimizationResult:
    """Create pentane input with restraint_k set to 5.0."""
    min_input = ConstrainedMinimizationInput(
        torsion_id=100,
        mapped_smiles=pentane_molecule.to_smiles(mapped=True),
        dihedral_indices=[0, 1, 2, 3],
        force_field="openff-2.0.0",
        coordinates=pentane_molecule.conformers[0].m_as("angstrom"),
        grid_id=15.0,  # arbitrary
        restraint_k=5.0,
    )

    result = _minimize_constrained(min_input)

    return result


def test_minimization_basic(pentane_restrained_minimization_result):
    """Test basic functionality of constrained minimization, only inspecting fixture."""
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


@pytest.mark.parametrize("failure_case", ["valence", "charge"])
def test_failed_minimizations(tmp_path, capsys, failure_case):
    """Test that failed MM minimizations in a torsion store are handled gracefully."""
    SMILES = "[H:5][C:1]([H:6])([H:7])[C:2]([H:8])([H:9])[C:3]([H:10])([H:11])[O:4][H:12]"
    FORCE_FIELD = (tmp_path / f"sage_missing_{failure_case}.offxml").as_posix()

    propanol = Molecule.from_mapped_smiles(SMILES)

    propanol.generate_conformers(n_conformers=1)

    profile = QCArchiveTorsionProfile(
        mapped_smiles=SMILES,
        dihedral_indices=[0, 1, 2, 3],
        qcarchive_id=12345,
        id=12345,
        coordinates={  # meaningless numbers, this test should fail before seeing coordinates
            -15.0: propanol.conformers[0].m_as("angstrom"),
            0.0: propanol.conformers[0].m_as("angstrom"),
            15.0: propanol.conformers[0].m_as("angstrom"),
        },
        energies={
            -15.0: -100.0,
            0.0: -110.0,
            15.0: -105.0,
        },
    )

    store = TorsionStore.from_torsion_dataset(
        dataset=QCArchiveTorsionDataset(
            tag="failed minimizations test",
            version=1,
            qm_torsions=[profile],
        ),
        database_name=(tmp_path / "failed_minimizations.sqlite").as_posix(),
    )

    assert len(store.get_qm_points_by_torsion_id(12345)) == 3
    assert store.get_qm_energies_by_torsion_id(12345) == {-15.0: -100.0, 0.0: -110.0, 15.0: -105.0}

    sage = ForceField("openff-2.0.0.offxml")

    match failure_case:
        case "valence":
            sage.deregister_parameter_handler("ProperTorsions")
            sage.get_parameter_handler("ProperTorsions")
        case "charge":
            sage.deregister_parameter_handler("ToolkitAM1BCC")

    sage.to_file(FORCE_FIELD)

    store.optimize_mm(force_field=FORCE_FIELD)

    # if some molecules fail in a force field, it probably shows up here, but for our dataset
    # all (one) molecule(s) fail so it doesn't
    assert len(store.get_force_fields()) == 0

    assert len(store.get_mm_energies_by_torsion_id(12345, force_field=FORCE_FIELD)) == 0

    # caplog fixture is supposed to handle this, but capsys is more reliable
    assert failure_case in capsys.readouterr().err, capsys.readouterr().err
