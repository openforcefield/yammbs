import json
import random
import tempfile

import numpy
import pytest
from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit import Molecule
from openff.utilities import get_data_file_path, has_executable, temporary_cd

from yammbs import MoleculeStore
from yammbs.checkmol import ChemicalEnvironment
from yammbs.exceptions import DatabaseExistsError
from yammbs.inputs import QCArchiveDataset, QCArchiveMolecule
from yammbs.models import MMConformerRecord, QMConformerRecord


def test_from_qcsubmit(small_qcsubmit_collection):
    db = "foo.sqlite"
    with temporary_cd():
        store = MoleculeStore.from_qcsubmit_collection(small_qcsubmit_collection, db)

        # Sanity check molecule deduplication
        assert len(store.get_smiles()) == len({*store.get_smiles()})

        # Ensure a new object can be created from the same database
        assert len(MoleculeStore(db)) == len(store)


def test_from_cached_collection(tiny_cache):
    db = "foo.sqlite"
    with temporary_cd():
        store = MoleculeStore.from_qcarchive_dataset(tiny_cache, db)

        # Sanity check molecule deduplication
        assert len(store.get_smiles()) == len({*store.get_smiles()})

        # Ensure a new object can be created from the same database
        assert len(MoleculeStore(db)) == len(store)

        # check output type for #67
        assert isinstance(store.get_qm_conformer_by_qcarchive_id(18433006), numpy.ndarray)


def test_from_qcarchive_dataset(small_qcsubmit_collection):
    """Test loading from YAMMBS's QCArchive model"""
    db = "foo.sqlite"
    with temporary_cd():
        store = MoleculeStore.from_qcarchive_dataset(
            QCArchiveDataset.from_qcsubmit_collection(small_qcsubmit_collection),
            db,
        )

        # Sanity check molecule deduplication
        assert len(store.get_smiles()) == len({*store.get_smiles()})

        # Ensure a new object can be created from the same database
        assert len(MoleculeStore(db)) == len(store)

        assert len(store.get_smiles()) == small_qcsubmit_collection.n_molecules


def test_from_qcarchive_dataset_undefined_stereo():
    """Test loading from YAMMBS's QCArchive model with undefined stereochemistry"""
    db = "foo.sqlite"

    ds = QCArchiveDataset(
        qm_molecules=[
            QCArchiveMolecule.model_validate_json(
                open(get_data_file_path("_tests/data/qcsubmit/undefined_stereo.json", "yammbs")).read(),
            ),
        ],
    )
    with temporary_cd():
        store = MoleculeStore.from_qcarchive_dataset(ds, db)

        # Sanity check molecule deduplication
        assert len(store.get_smiles()) == len({*store.get_smiles()})

        # Ensure a new object can be created from the same database
        assert len(MoleculeStore(db)) == len(store)


def test_do_not_overwrite(small_qcsubmit_collection):
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as file:
        with pytest.raises(DatabaseExistsError, match="already exists."):
            MoleculeStore.from_qcsubmit_collection(
                small_qcsubmit_collection,
                file.name,
            )


def test_load_existing_database(small_store):
    assert len(small_store) == 40


def test_get_molecule_ids(small_store):
    molecule_ids = small_store.get_molecule_ids()

    assert len(molecule_ids) == len({*molecule_ids}) == 40

    assert min(molecule_ids) == 1
    assert max(molecule_ids) == 40


def test_get_molecule_id_by_qcarchive_id(small_store):
    molecule_id = 40
    qcarchive_id = small_store.get_qcarchive_ids_by_molecule_id(molecule_id)[-1]

    assert small_store.get_molecule_id_by_qcarchive_id(qcarchive_id) == molecule_id


def test_get_inchi_by_molecule_id(small_store):
    Molecule.from_inchi(small_store.get_inchi_key_by_molecule_id(40))


def test_molecules_sorted_by_qcarchive_id():
    raw_ch = json.load(
        open(
            get_data_file_path(
                "_tests/data/qcsubmit/01-processed-qm-ch.json",
                "yammbs",
            ),
        ),
    )

    random.shuffle(raw_ch["entries"]["https://api.qcarchive.molssi.org:443/"])

    with tempfile.NamedTemporaryFile(mode="w+") as file:
        json.dump(raw_ch, file)
        file.flush()

        store = MoleculeStore.from_qcsubmit_collection(
            OptimizationResultCollection.parse_file(file.name),
            database_name=tempfile.NamedTemporaryFile(suffix=".sqlite").name,
        )

        qcarchive_ids = store.get_qcarchive_ids_by_molecule_id(40)

    for index, id in enumerate(qcarchive_ids[:-1]):
        assert id < qcarchive_ids[index + 1]


def test_get_conformers(small_store):
    force_field = "openff-2.0.0"
    molecule_id = 40
    qcarchive_id = small_store.get_qcarchive_ids_by_molecule_id(molecule_id)[-1]

    numpy.testing.assert_allclose(
        small_store.get_qm_conformer_by_qcarchive_id(
            qcarchive_id,
        ),
        small_store.get_qm_conformers_by_molecule_id(molecule_id)[-1],
    )

    numpy.testing.assert_allclose(
        small_store.get_mm_conformer_by_qcarchive_id(
            qcarchive_id,
            force_field=force_field,
        ),
        small_store.get_mm_conformers_by_molecule_id(
            molecule_id,
            force_field=force_field,
        )[-1],
    )


def test_get_molecules(small_store):
    force_field = "openff-2.0.0"
    molecule_id = 40
    qcarchive_id = small_store.get_qcarchive_ids_by_molecule_id(molecule_id)[-1]

    numpy.testing.assert_allclose(
        small_store.get_qm_conformer_by_qcarchive_id(
            qcarchive_id,
        ),
        small_store.get_qm_molecule_by_qcarchive_id(qcarchive_id).conformers[0].m_as("angstroms"),
    )

    numpy.testing.assert_allclose(
        small_store.get_mm_conformer_by_qcarchive_id(
            qcarchive_id,
            force_field=force_field,
        ),
        small_store.get_mm_molecule_by_qcarchive_id(
            qcarchive_id,
            force_field=force_field,
        )
        .conformers[0]
        .m_as("angstroms"),
    )


def test_get_force_fields(small_store):
    force_fields = small_store.get_force_fields()

    assert len(force_fields) == 9

    assert "openff-2.1.0" in force_fields
    assert "gaff-2.11" in force_fields
    assert "openff-3.0.0" not in force_fields


def test_get_mm_conformer_records_by_molecule_id(small_store, diphenylvinylbenzene):
    records = small_store.get_mm_conformer_records_by_molecule_id(
        1,
        force_field="openff-2.1.0",
    )

    for record in records:
        assert isinstance(record, MMConformerRecord)
        assert record.molecule_id == 1
        assert record.force_field == "openff-2.1.0"
        assert record.coordinates.shape == (36, 3)
        assert record.energy is not None

        assert Molecule.from_mapped_smiles(record.mapped_smiles).is_isomorphic_with(
            diphenylvinylbenzene,
        )


def test_get_qm_conformer_records_by_molecule_id(small_store, diphenylvinylbenzene):
    records = small_store.get_qm_conformer_records_by_molecule_id(1)

    for record in records:
        assert isinstance(record, QMConformerRecord)
        assert record.molecule_id == 1
        assert record.coordinates.shape == (36, 3)
        assert record.energy is not None

        assert Molecule.from_mapped_smiles(record.mapped_smiles).is_isomorphic_with(
            diphenylvinylbenzene,
        )


@pytest.mark.parametrize(("molecule_id", "expected_len"), [(28, 1), (40, 9)])
def test_get_mm_energies_by_molecule_id(
    small_store,
    molecule_id,
    expected_len,
):
    """Trigger issue #16."""
    energies = small_store.get_mm_energies_by_molecule_id(
        molecule_id,
        force_field="openff-2.0.0",
    )

    for energy in energies:
        assert isinstance(energy, float)

    assert len(energies) == expected_len


@pytest.mark.parametrize(("molecule_id", "expected_len"), [(28, 1), (40, 9)])
def test_get_qm_energies_by_molecule_id(
    small_store,
    molecule_id,
    expected_len,
):
    energies = small_store.get_qm_energies_by_molecule_id(molecule_id)

    for energy in energies:
        assert isinstance(energy, float)

    assert len(energies) == expected_len


@pytest.mark.skipif(not has_executable("checkmol"), reason="checkmol not installed")
@pytest.mark.parametrize(
    "func",
    [
        ("get_dde"),
        ("get_rmsd"),
        # ("get_internal_coordinate_rmsd"),
        ("get_tfd"),
    ],
)
@pytest.mark.parametrize(
    ("environment", "expected_len"),
    [
        (ChemicalEnvironment.Alkane, 9),
        (ChemicalEnvironment.Alkene, 8),
        (ChemicalEnvironment.Aromatic, 24),
        (ChemicalEnvironment.Alcohol, 0),  # no O in dataset
        (ChemicalEnvironment.Nitrile, 0),  # no N in dataset
    ],
)
def test_filter_by_checkmol(small_store, environment, expected_len, func):
    all_values = getattr(small_store, func)(force_field="openff-2.1.0")

    filtered_ids = small_store.filter_by_checkmol(environment)
    assert len(filtered_ids) == expected_len

    filtered_values = getattr(small_store, func)(
        force_field="openff-2.1.0",
        molecule_ids=filtered_ids,
    )

    for value in filtered_values:
        assert value in all_values


@pytest.mark.parametrize(
    "func",
    [
        ("get_dde"),
        ("get_rmsd"),
        # ("get_internal_coordinate_rmsd"),
        ("get_tfd"),
    ],
)
@pytest.mark.parametrize(
    ("smirks", "expected_len"),
    [
        ("[#6:1]=[#6:2]", 8),
        ("[#6:1]:[#6:2]", 24),
        ("[#6:1]~[#7:2]", 0),
        ("[#6:1]~[#8:2]", 0),
    ],
)
def test_filter_by_smirks(small_store, smirks, expected_len, func):
    all_values = getattr(small_store, func)(force_field="openff-2.1.0")

    filtered_ids = small_store.filter_by_smirks(smirks)
    assert len(filtered_ids) == expected_len

    filtered_values = getattr(small_store, func)(
        force_field="openff-2.1.0",
        molecule_ids=filtered_ids,
    )

    for value in filtered_values:
        assert value in all_values


def test_get_metrics(small_store):
    metrics = small_store.get_metrics()

    sage_metrics = metrics.metrics["openff-2.1.0"]

    assert len(sage_metrics) > 0

    for other_force_field in [
        "openff-1.0.0",
        "openff-1.3.0",
        "openff-2.1.0",
        "gaff-2.11",
    ]:
        assert len(sage_metrics) == len(metrics.metrics[other_force_field])

    this_metric = sage_metrics[37016887]

    # semi hard-coded ranges, which shouldn't change with source code anyway
    assert abs(this_metric.dde) < 0.01
    assert this_metric.rmsd < 0.2
    assert this_metric.tfd < 0.2
    assert this_metric.icrmsd["Bond"] < 0.1
    assert this_metric.icrmsd["Angle"] < 2
    assert this_metric.icrmsd["Dihedral"] < 15
    assert this_metric.icrmsd["Improper"] < 2
