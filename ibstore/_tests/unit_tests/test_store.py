import tempfile

from ibstore._store import MoleculeStore


def test_from_qcsubmit(small_collection):
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as file:
        store = MoleculeStore.from_qcsubmit_collection(
            small_collection,
            file.name,
        )

        # Sanity check molecule deduplication
        assert len(store.get_smiles()) == len({*store.get_smiles()})
