import tempfile

import numpy
import pytest
from openff.utilities import get_data_file_path, temporary_cd

from ibstore._store import MoleculeStore
from ibstore.exceptions import DatabaseExistsError


def test_from_qcsubmit(small_collection):
    db = "foo.sqlite"
    with temporary_cd():
        store = MoleculeStore.from_qcsubmit_collection(small_collection, db)

        # Sanity check molecule deduplication
        assert len(store.get_smiles()) == len({*store.get_smiles()})

        # Ensure a new object can be created from the same database
        assert len(MoleculeStore(db)) == len(store)


def test_do_not_overwrite(small_collection):
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as file:
        with pytest.raises(DatabaseExistsError, match="already exists."):
            MoleculeStore.from_qcsubmit_collection(
                small_collection,
                file.name,
            )


def test_load_existing_databse():
    # This file manually generated from data/01-processed-qm-ch.json
    store = MoleculeStore(
        get_data_file_path(
            "_tests/data/01-processed-qm-ch.sqlite",
            package_name="ibstore",
        ),
    )

    assert len(store) == 40


def test_get_molecule_ids(small_store):
    molecule_ids = small_store.get_molecule_ids()

    assert len(molecule_ids) == len({*molecule_ids}) == 40

    assert min(molecule_ids) == 1
    assert max(molecule_ids) == 40


def test_get_molecule_id_by_qcarchive_id(small_store):
    molecule_id = 40
    qcarchive_id = small_store.get_qcarchive_ids_by_molecule_id(molecule_id)[-1]

    assert small_store.get_molecule_id_by_qcarchive_id(qcarchive_id) == molecule_id


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
