import numpy
import qcelemental
from openff.units import unit

from ibstore.models import MoleculeRecord, QMConformerRecord

hartree2kcalmol = qcelemental.constants.hartree2kcalmol


def test_load_from_qcsubmit(small_collection):
    for record_and_molecule in small_collection.to_records():
        qc_record = record_and_molecule[0]
        molecule = record_and_molecule[1]

        mapped_smiles = molecule.to_smiles(
            mapped=True,
            isomeric=True,
        )
        ichi_key = molecule.to_inchikey(
            fixed_hydrogens=True,
        )

        molecule_record, qm_conformer = MoleculeRecord.from_record_and_molecule(
            qc_record,
            molecule,
        )

        assert isinstance(molecule_record, MoleculeRecord)
        assert molecule_record.mapped_smiles == mapped_smiles
        assert molecule_record.inchi_key == ichi_key

        assert isinstance(qm_conformer, QMConformerRecord)
        assert qm_conformer.energy == qc_record.get_final_energy() * hartree2kcalmol
        assert qm_conformer.qcarchive_id == qc_record.id
        assert numpy.allclose(
            qm_conformer.coordinates,
            molecule.conformers[0].m_as(unit.angstrom),
        )
