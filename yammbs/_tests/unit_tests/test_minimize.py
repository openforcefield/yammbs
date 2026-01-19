import platform

import numpy
import pytest
from openff.toolkit import ForceField, Molecule, unit
from openff.toolkit import __version__ as __toolkit_version__

from yammbs import MoleculeStore
from yammbs._minimize import MinimizationInput, _run_openmm
from yammbs.inputs import QCArchiveDataset


@pytest.fixture
def ethane() -> Molecule:
    """Return an ethane with a bad conformer."""
    ethane = Molecule.from_smiles("CC")
    ethane.generate_conformers(n_conformers=1)

    return ethane


@pytest.fixture
def perturbed_ethane(ethane) -> Molecule:
    """Return an ethane with a bad conformer."""
    ethane._conformers[0] *= 1.2

    return ethane


def basic_input(force_field="openff-1.0.0") -> MinimizationInput:
    ethane = Molecule.from_smiles("CC")
    ethane.generate_conformers(n_conformers=1)

    return MinimizationInput(
        inchi_key=ethane.to_inchikey(),
        qcarchive_id=123485854848,
        force_field=force_field,
        mapped_smiles=ethane.to_smiles(mapped=True),
        coordinates=ethane.conformers[0].m_as(unit.angstrom),
    )


@pytest.fixture
def perturbed_input(perturbed_ethane) -> MinimizationInput:
    return MinimizationInput(
        inchi_key=perturbed_ethane.to_inchikey(),
        qcarchive_id=348483483,
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
        "inchi_key",
        "qcarchive_id",
        "mapped_smiles",
    ):
        assert getattr(perturbed_input, attr) == getattr(result, attr)

    # Check that the C-C bond distance has settled to ~1.5 A
    final = result.coordinates[0] - result.coordinates[1]
    assert 1.5 < numpy.linalg.norm(final) < 1.6


def test_minimization_unassigned_torsion(caplog):
    """Test that ``_run_openmm`` returns None and logs a warning when there are unassigned valence terms."""
    smiles = (
        "[H:6][C@@:5]([C:16](=[O:17])[O:18][F:19])([C@:4]([H:31])([C:2]"
        "([H:27])([C:1]([H:24])([H:25])[H:26])[C:3]([H:28])([H:29])[H:30])"
        "[C:20]([C:21]([H:44])([H:45])[H:46])([C:22]([H:47])([H:48])[H:49])"
        "[O:23][H:50])[N:7]([C:9](=[O:10])[O:11][C:12]([C:13]([H:35])([H:36])"
        "[H:37])([C:14]([H:38])([H:39])[H:40])[C:15]([H:41])([H:42])[H:43])"
        "[C:8]([H:32])([H:33])[H:34]"
    )
    mol = Molecule.from_mapped_smiles(smiles)
    mol.generate_conformers(n_conformers=1)
    min_input = MinimizationInput(
        inchi_key=mol.to_inchikey(),
        qcarchive_id=36955694,
        force_field="openff-1.0.0",
        mapped_smiles=mol.to_smiles(mapped=True),
        coordinates=mol.conformers[0].m_as(unit.angstrom),
    )

    result = _run_openmm(min_input)

    assert "unassigned valence terms" in caplog.text
    assert result is None


def test_same_force_field_same_results():
    energy1 = _run_openmm(basic_input("openff-1.0.0")).energy
    energy2 = _run_openmm(basic_input("openff-1.0.0")).energy

    assert energy1 == energy2


def test_different_force_fields_different_results():
    energy1 = _run_openmm(basic_input("openff-1.0.0")).energy
    energy2 = _run_openmm(basic_input("openff-2.0.0")).energy

    assert energy1 != energy2


def test_plugin_loadable(ethane):
    pytest.importorskip("deforcefields.deforcefields")

    _run_openmm(
        MinimizationInput(
            inchi_key=ethane.to_inchikey(),
            qcarchive_id=38483483483481384183412831832,
            force_field="de-force-1.0.1",
            mapped_smiles=ethane.to_smiles(mapped=True),
            coordinates=ethane.conformers[0].m_as(unit.angstrom),
        ),
    )


@pytest.mark.timeout(3)
def test_cached_force_fields_load_quickly():
    """Test that cached force fields are loaded quickly."""
    from yammbs._minimize import _lazy_load_force_field

    # timeout includes the time it takes to load it the first time, but that should be << 1 second
    [_lazy_load_force_field("openff-1.0.0") for _ in range(1000)]


def test_finds_local_force_field(ethane, tmp_path):
    ForceField("openff_unconstrained-1.2.0.offxml").to_file(tmp_path / "fOOOO.offxml")

    _run_openmm(
        MinimizationInput(
            inchi_key=ethane.to_inchikey(),
            qcarchive_id=5,
            force_field=(tmp_path / "fOOOO.offxml").as_posix(),
            mapped_smiles=ethane.to_smiles(mapped=True),
            coordinates=ethane.conformers[0].m_as(unit.angstrom),
        ),
    )


def test_plugin_not_needed_to_use_mainline_force_field(monkeypatch, ethane):
    pytest.importorskip("deforcefields.deforcefields")
    from deforcefields import deforcefields

    assert len(deforcefields.get_forcefield_paths()) > 0

    def return_no_paths():
        return list()

    monkeypatch.setattr(deforcefields, "get_forcefield_paths", return_no_paths)

    assert len(deforcefields.get_forcefield_paths()) == 0

    _run_openmm(
        MinimizationInput(
            inchi_key=ethane.to_inchikey(),
            qcarchive_id=5,
            force_field="openff-1.0.0",
            mapped_smiles=ethane.to_smiles(mapped=True),
            coordinates=ethane.conformers[0].m_as(unit.angstrom),
        ),
    )


@pytest.mark.timeout(100 if platform.system() == "Darwin" else 60)
def test_partially_minimized(tiny_cache, tmp_path):
    """Test that minimizing with one force field produces expected results.

    See https://github.com/mattwthompson/ib/pull/21#discussion_r1511804909
    """
    import yammbs

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
    tinier_cache = QCArchiveDataset()

    # ... so slice out some really small molecules (< 7 heavy atoms)
    # which should be 4 molecules for this dataset
    for result in tiny_cache.qm_molecules:
        molecule = Molecule.from_mapped_smiles(result.mapped_smiles)
        if len([atom for atom in molecule.atoms if atom.atomic_number > 1]) < 7:
            tinier_cache.qm_molecules.append(result)

    tinier_store = MoleculeStore.from_qcarchive_dataset(
        tinier_cache,
        database_name=(tmp_path / "tiny.sqlite").as_posix(),
    )

    assert get_n_results(tinier_store) == (0, 0)

    tinier_store.optimize_mm(force_field="openff-1.0.0", n_processes=1)

    assert get_n_results(tinier_store) == (4, 0)

    tinier_store.optimize_mm(force_field="openff-2.0.0", n_processes=1)

    assert get_n_results(tinier_store) == (4, 4)

    assert tinier_store.software_provenance["yammbs"] == yammbs.__version__
    assert tinier_store.software_provenance["openff.toolkit"] == __toolkit_version__
    assert tinier_store.software_provenance["qcfractal"] is None
