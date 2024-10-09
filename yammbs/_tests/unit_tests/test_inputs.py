import numpy
from openff.toolkit import Molecule
from openff.utilities.utilities import get_data_file_path

from yammbs.inputs import QCArchiveDataset


class TestQCArchiveDataset:
    def test_from_qcsubmit_collection(
        self,
        small_qcsubmit_collection,
    ):
        collection = QCArchiveDataset.from_qcsubmit_collection(small_qcsubmit_collection)

        for qm_molecule in collection.qm_molecules:
            assert isinstance(qm_molecule.final_energy, float)

            assert isinstance(qm_molecule.qcarchive_id, int)
            assert not isinstance(qm_molecule.qcarchive_id, (bool, str))

            molecule = Molecule.from_mapped_smiles(qm_molecule.mapped_smiles)

            assert (molecule.n_atoms, 3) == qm_molecule.coordinates.shape


class TestSerialization:
    def test_json_roundtrip(self):
        dataset = QCArchiveDataset.model_validate_json(
            open(get_data_file_path("_tests/data/yammbs/01-processed-qm-ch.json", "yammbs")).read(),
        )

        roundtripped = QCArchiveDataset.model_validate_json(dataset.model_dump_json())

        assert len(dataset.qm_molecules) == len(roundtripped.qm_molecules)

        assert numpy.all(dataset.qm_molecules[-1].coordinates == roundtripped.qm_molecules[-1].coordinates)
