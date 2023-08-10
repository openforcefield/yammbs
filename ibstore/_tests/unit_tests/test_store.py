import tempfile

import pytest
from openff.utilities import get_data_file_path

from ibstore._store import MoleculeStore
from ibstore.exceptions import DatabaseExistsError


def test_from_qcsubmit(small_collection):
    store = MoleculeStore.from_qcsubmit_collection(
        small_collection,
        "foo.sqlite",
    )

    # Sanity check molecule deduplication
    assert len(store.get_smiles()) == len({*store.get_smiles()})

    # Ensure a new object can be created from the same database
    assert len(MoleculeStore("foo.sqlite")) == len(store)


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
