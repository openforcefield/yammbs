from openff.nagl._base.array import Array
from openff.nagl._base.base import ImmutableModel
from openff.toolkit import Molecule
from pydantic import Field
from qcportal.models.records import OptimizationRecord


class Record(ImmutableModel):
    class Config(ImmutableModel.Config):
        orm_mode = True


class QMConformerRecord(Record):
    """A record for storing coordinates computed from QC. Assumes use with QCArchive"""

    qcarchive_id: str = Field(
        ...,
        description="The ID of the molecule in the QCArchive database",
    )
    coordinates: Array = Field(
        ...,
        description=(
            "The coordinates [Angstrom] of this conformer with shape=(n_atoms, 3)."
        ),
    )
    energy: float = Field(
        ...,
        description="The final energy (kcal/mol) of the molecule as optimized and computed by QCArchive",
    )


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

    mapped_smiles: str = Field(
        ...,
        description="The mapped SMILES string for the molecule with hydrogens specified",
    )
    inchi_key: str = Field(
        ...,
        description="The InChI key for the molecule",
    )

    @property
    def smiles(self):
        return self.mapped_smiles

    @classmethod
    def from_record_and_molecule(
        cls,
        record: OptimizationRecord,
        molecule: Molecule,
    ):
        # import qcelemental
        # from openff.units import unit

        # hartree2kcalmol = qcelemental.constants.hartree2kcalmol

        assert molecule.n_conformers == 1

        return cls(
            mapped_smiles=molecule.to_smiles(
                mapped=True,
                isomeric=True,
            ),
            inchi_key=molecule.to_inchikey(
                fixed_hydrogens=True,
            ),
        )

        # TODO: Also add QC data (id, energy, coordinates) as a QCConformerRecord,
        #       but don't store it here

        # qcarchive_id=record.id,
        # qcarchive_energy=record.get_final_energy() * hartree2kcalmol,
        # conformer=QMConformerRecord(
        #     coordinates=molecule.conformers[0].m_as(unit.angstrom),
        #     energy=record.get_final_energy() * hartree2kcalmol,
        # ),
