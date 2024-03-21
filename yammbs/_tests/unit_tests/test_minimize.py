import numpy
import pytest
from openff.toolkit import ForceField, Molecule
from openff.units import unit

from yammbs import MoleculeStore
from yammbs._minimize import MinimizationInput, _run_openmm
from yammbs.cached_result import CachedResultCollection


@pytest.fixture()
def ethane() -> Molecule:
    """Return an ethane with a bad conformer"""
    ethane = Molecule.from_smiles("CC")
    ethane.generate_conformers(n_conformers=1)

    return ethane


@pytest.fixture()
def perturbed_ethane(ethane) -> Molecule:
    """Return an ethane with a bad conformer"""
    ethane._conformers[0] *= 1.2

    return ethane


def basic_input(force_field="openff-1.0.0") -> MinimizationInput:
    ethane = Molecule.from_smiles("CC")
    ethane.generate_conformers(n_conformers=1)

    return MinimizationInput(
        qcarchive_id="test",
        force_field=force_field,
        mapped_smiles=ethane.to_smiles(mapped=True),
        coordinates=ethane.conformers[0].m_as(unit.angstrom),
    )


@pytest.fixture()
def perturbed_input(perturbed_ethane) -> MinimizationInput:
    return MinimizationInput(
        qcarchive_id="test",
        force_field="openff-1.0.0",
        mapped_smiles=perturbed_ethane.to_smiles(mapped=True),
        coordinates=perturbed_ethane.conformers[0].m_as(unit.angstrom),
    )


def test_minimization_basic(perturbed_input):
    # This C-C bond distance should be ~1.5 * 1.2 ~= 1.8
    initial = (perturbed_input.coordinates[0] - perturbed_input.coordinates[1],)
    assert numpy.linalg.norm(initial) > 1.6

    result = _run_openmm(perturbed_input)

    for attr in (
        "qcarchive_id",
        "mapped_smiles",
    ):
        assert getattr(perturbed_input, attr) == getattr(result, attr)

    # Check that the C-C bond distance has settled to ~1.5 A
    final = result.coordinates[0] - result.coordinates[1]
    assert 1.5 < numpy.linalg.norm(final) < 1.6


def test_same_force_field_same_results():
    energy1 = _run_openmm(basic_input("openff-1.0.0")).energy
    energy2 = _run_openmm(basic_input("openff-1.0.0")).energy

    assert energy1 == energy2


def test_different_force_fields_different_results():
    energy1 = _run_openmm(basic_input("openff-1.0.0")).energy
    energy2 = _run_openmm(basic_input("openff-2.0.0")).energy

    assert energy1 != energy2


def test_plugin_loadable(ethane):
    _run_openmm(
        MinimizationInput(
            qcarchive_id="test",
            force_field="de-force-1.0.1",
            mapped_smiles=ethane.to_smiles(mapped=True),
            coordinates=ethane.conformers[0].m_as(unit.angstrom),
        ),
    )


@pytest.mark.timeout(1)
def test_cached_force_fields_load_quickly():
    """Test that cached force fields are loaded quickly."""
    from yammbs._minimize import _lazy_load_force_field

    # timeout includes the time it takes to load it the first time, but that should be << 1 second
    [_lazy_load_force_field("openff-1.0.0") for _ in range(1000)]


def test_finds_local_force_field(ethane, tmp_path):
    ForceField("openff_unconstrained-1.2.0.offxml").to_file(tmp_path / "fOOOO.offxml")

    _run_openmm(
        MinimizationInput(
            qcarchive_id="test",
            force_field=(tmp_path / "fOOOO.offxml").as_posix(),
            mapped_smiles=ethane.to_smiles(mapped=True),
            coordinates=ethane.conformers[0].m_as(unit.angstrom),
        ),
    )


def test_plugin_not_needed_to_use_mainline_force_field(monkeypatch, ethane):
    from deforcefields import deforcefields

    assert len(deforcefields.get_forcefield_paths()) > 0

    def return_no_paths():
        return list()

    monkeypatch.setattr(deforcefields, "get_forcefield_paths", return_no_paths)

    assert len(deforcefields.get_forcefield_paths()) == 0

    _run_openmm(
        MinimizationInput(
            qcarchive_id="test",
            force_field="openff-1.0.0",
            mapped_smiles=ethane.to_smiles(mapped=True),
            coordinates=ethane.conformers[0].m_as(unit.angstrom),
        ),
    )


def test_partially_minimized(tiny_cache, tmp_path, guess_n_processes):
    """
    Test that minimizing with one force field produces expected results

    See https://github.com/mattwthompson/ib/pull/21#discussion_r1511804909
    """

    def get_n_mm_conformers(store, ff):
        molecule_ids = store.get_molecule_ids()
        return sum(
            len(
                store.get_mm_conformers_by_molecule_id(
                    id=molecule_id,
                    force_field=ff,
                ),
            )
            for molecule_id in molecule_ids
        )

    def get_n_results(store) -> tuple[int, ...]:
        return (
            get_n_mm_conformers(store, "openff-1.0.0"),
            get_n_mm_conformers(store, "openff-2.0.0"),
        )

    # No need to minimize all 200 records twice ...
    tinier_cache = CachedResultCollection()

    # ... so slice out some really small molecules (< 9 heavy atoms)
    # which should be 12 molecules for this dataset
    for result in tiny_cache.inner:
        molecule = Molecule.from_mapped_smiles(result.mapped_smiles)
        if len([atom for atom in molecule.atoms if atom.atomic_number > 1]) < 9:
            tinier_cache.inner.append(result)

    tinier_store = MoleculeStore.from_cached_result_collection(
        tinier_cache,
        database_name=(tmp_path / "tiny.sqlite").as_posix(),
    )

    assert get_n_results(tinier_store) == (0, 0)

    tinier_store.optimize_mm(force_field="openff-1.0.0", n_processes=guess_n_processes)

    assert get_n_results(tinier_store) == (12, 0)

    tinier_store.optimize_mm(force_field="openff-2.0.0", n_processes=guess_n_processes)

    assert get_n_results(tinier_store) == (12, 12)
