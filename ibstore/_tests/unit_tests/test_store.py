import tempfile

from openff.utilities import get_data_file_path

from ibstore._store import MoleculeStore


def test_from_qcsubmit(small_collection):
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as file:
        store = MoleculeStore.from_qcsubmit_collection(
            small_collection,
            file.name,
        )

        # Sanity check molecule deduplication
        assert len(store.get_smiles()) == len({*store.get_smiles()})

        assert len(MoleculeStore(file.name)) == len(store)


def test_load_existing_databse():
    # This file manually generated from data/01-processed-qm-ch.json
    store = MoleculeStore(
        get_data_file_path(
            "_tests/data/01-processed-qm-ch.sqlite",
            package_name="ibstore",
        ),
    )

    assert len(store) == 40
