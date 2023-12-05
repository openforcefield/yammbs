import numpy
import qcelemental
from openff.units import unit

from ibstore.models import MoleculeRecord, QMConformerRecord

hartree2kcalmol = qcelemental.constants.hartree2kcalmol


def test_load_from_qcsubmit(small_collection):
    for qc_record, molecule in small_collection.to_records():
        mapped_smiles = molecule.to_smiles(mapped=True, isomeric=True)
        ichi_key = molecule.to_inchi(fixed_hydrogens=True)

        molecule_record = MoleculeRecord.from_molecule(molecule)

        qm_conformer = QMConformerRecord.from_qcarchive_record(
            molecule_id="1",
            mapped_smiles=mapped_smiles,
            qc_record=qc_record,
            coordinates=molecule.conformers[0],
        )

        assert isinstance(molecule_record, MoleculeRecord)
        assert molecule_record.mapped_smiles == mapped_smiles
        assert molecule_record.inchi_key == ichi_key

        assert isinstance(qm_conformer, QMConformerRecord)
        assert qm_conformer.energy == qc_record.energies[-1] * hartree2kcalmol
        assert qm_conformer.qcarchive_id == qc_record.id
        assert numpy.allclose(
            qm_conformer.coordinates,
            molecule.conformers[0].m_as(unit.angstrom),
        )
