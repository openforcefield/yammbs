import pytest
from openff.qcsubmit.results import OptimizationResultCollection
from openff.utilities.utilities import get_data_file_path

from ibstore.models import MoleculeRecord


@pytest.fixture()
def small_collection() -> OptimizationResultCollection:
    return OptimizationResultCollection.parse_file(
        get_data_file_path("_tests/data/01-processed-qm-ch.json", "ibstore")
    )


def test_load_from_qcsubmit(small_collection):
    for record_and_molecule in small_collection.to_records():
        record = record_and_molecule[0]
        molecule = record_and_molecule[1]

        mapped_smiles = molecule.to_smiles(
            mapped=True,
            isomeric=True,
        )
        ichi_key = molecule.to_inchikey(
            fixed_hydrogens=True,
        )

        record = MoleculeRecord.from_record_and_molecule(
            record,
            molecule,
        )

        assert record.mapped_smiles == mapped_smiles
        assert record.inchi_key == ichi_key
