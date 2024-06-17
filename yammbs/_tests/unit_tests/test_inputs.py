from openff.toolkit import Molecule

from yammbs.inputs import QCArchiveDataset


class TestQCArchiveDataset:
    def test_from_qcsubmit_collection(
        self,
        small_collection,
    ):
        collection = QCArchiveDataset.from_qcsubmit_collection(small_collection)

        for qm_molecule in collection.qm_molecules:
            assert isinstance(qm_molecule.final_energy, float)

            assert isinstance(qm_molecule.qcarchive_id, int)
            assert not isinstance(qm_molecule.qcarchive_id, (bool, str))

            molecule = Molecule.from_mapped_smiles(qm_molecule.mapped_smiles)

            assert (molecule.n_atoms, 3) == qm_molecule.coordinates.shape
