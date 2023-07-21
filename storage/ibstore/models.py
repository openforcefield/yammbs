from typing import TypeVar

import qcelemental
from openff.toolkit import Molecule
from pydantic import Field
from qcportal.models.records import OptimizationRecord

from ibstore._base.array import Array
from ibstore._base.base import ImmutableModel

hartree2kcalmol = qcelemental.constants.hartree2kcalmol
bohr2angstroms = qcelemental.constants.bohr2angstroms

MR = TypeVar("MR", bound="MoleculeRecord")


class Record(ImmutableModel):
    class Config(ImmutableModel.Config):
        orm_mode = True


class QMConformerRecord(Record):
    """A record for storing coordinates computed from QC. Assumes use with QCArchive"""

    molecule_id: int = Field(
        ...,
        description="The ID of the molecule in the database",
    )
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

    @classmethod
    def from_qcarchive_record(
        cls,
        molecule_id: int,
        qc_record: OptimizationRecord,
    ):
        return cls(
            molecule_id=molecule_id,
            qcarchive_id=qc_record.id,
            coordinates=qc_record.get_final_molecule().geometry * bohr2angstroms,
            energy=qc_record.get_final_energy() * hartree2kcalmol,
        )


class MMConformerRecord(Record):
    molecule_id: int = Field(
        ...,
        description="The ID of the molecule in the database",
    )
    qcarchive_id: str = Field(
        ...,
        description="The ID of the molecule in the QCArchive database that this conformer corresponds to",
    )
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
    def from_molecule(
        cls,
        molecule: Molecule,
    ) -> tuple[MR, QMConformerRecord]:
        assert molecule.n_conformers == 1

        return cls(
            mapped_smiles=molecule.to_smiles(
                mapped=True,
                isomeric=True,
            ),
            inchi_key=molecule.to_inchi(
                fixed_hydrogens=True,
            ),
        )
