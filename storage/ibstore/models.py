from openff.nagl._base.array import Array
from openff.nagl._base.base import ImmutableModel
from openff.toolkit import Molecule
from pydantic import Field
from qcportal.models.records import OptimizationRecord


class Record(ImmutableModel):
    class Config(ImmutableModel.Config):
        orm_mode = True


class QMConformerRecord(Record):
    coordinates: Array = Field(
        ...,
        description=(
            "The coordinates [Angstrom] of this conformer with shape=(n_atoms, 3)."
        ),
    )
    energy: float


class MMConformerRecord(Record):
    coordinates: Array = Field(
        ...,
        description=(
            "The coordinates [Angstrom] of this conformer with shape=(n_atoms, 3)."
        ),
    )
    energy: float


class MoleculeRecord(Record):
    """A record which contains information for a labelled molecule. This may include the
    coordinates of the molecule in different conformers, and partial charges / WBOs
    computed for those conformers."""

    qcarchive_id: str = Field(
        ...,
        description="The ID of the molecule in the QCArchive database",
    )
    qcarchive_energy: float = Field(
        ...,
        description="The final energy (kcal/mol) of the molecule as optimized and computed by QCArchive",
    )
    mapped_smiles: str = Field(
        ...,
        description="The mapped SMILES string for the molecule with hydrogens specified",
    )
    inchi_key: str = Field(
        ...,
        description="The InChI key for the molecule",
    )
    conformer: QMConformerRecord = Field(
        ...,
        description="Conformers associated with the molecule. ",
    )
    minimized_conformer: MMConformerRecord = Field(
        ...,
        description="Minimized conformers associated with the molecule. ",
    )
    minimized_energy: float = Field(
        ...,
        description="The final energy of the molecule as minimized and computed by OpenMM",
    )

    @property
    def smiles(self):
        return self.mapped_smiles

    @classmethod
    def from_qc_and_mm(
        cls,
        qc_record: OptimizationRecord,
        qc_molecule: Molecule,
        mm_molecule: Molecule,
        minimized_energy: float,
    ):
        import qcelemental
        from openff.units import unit

        hartree2kcalmol = qcelemental.constants.hartree2kcalmol

        assert qc_molecule.n_conformers == mm_molecule.n_conformers == 1
        assert qc_molecule.to_smiles(
            mapped=True,
            isomeric=True,
        ) == mm_molecule.to_smiles(
            mapped=True,
            isomeric=True,
        )

        return cls(
            qcarchive_id=qc_record.id,
            qcarchive_energy=qc_record.get_final_energy() * hartree2kcalmol,
            mapped_smiles=qc_molecule.to_smiles(
                mapped=True,
                isomeric=True,
            ),
            conformer=QMConformerRecord(
                coordinates=qc_molecule.conformers[0].m_as(unit.angstrom)
            ),
            minimized_conformer=MMConformerRecord(
                coordinates=mm_molecule.conformers[0].m_as(unit.angstrom)
            ),
            minimized_energy=minimized_energy,
        )
