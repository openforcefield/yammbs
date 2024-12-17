from typing import Any, TypeVar

import qcelemental
from openff.toolkit import Molecule
from pydantic import ConfigDict, Field

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel

hartree2kcalmol = qcelemental.constants.hartree2kcalmol
bohr2angstroms = qcelemental.constants.bohr2angstroms

MR = TypeVar("MR", bound="MoleculeRecord")


class Record(ImmutableModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        frozen=False,
    )


class QMConformerRecord(Record):
    """A record for storing coordinates computed from QC. Assumes use with QCArchive"""

    molecule_id: int = Field(
        ...,
        description="The ID of the molecule in the database",
    )
    qcarchive_id: int = Field(
        ...,
        description="The ID of the molecule in the QCArchive database",
    )
    mapped_smiles: str = Field(
        ...,
        description="The mapped SMILES string for the molecule, stored to track atom maps",
    )
    coordinates: Array = Field(
        ...,
        description="The coordinates [Angstrom] of this conformer with shape=(n_atoms, 3).",
    )
    energy: float = Field(
        ...,
        description="The final energy (kcal/mol) of the molecule as optimized and computed by QCArchive",
    )

    @classmethod
    def from_qcarchive_record(
        cls,
        molecule_id: int,
        mapped_smiles: str,
        qc_record: Any,  # qcportal.optimization.OptimizationRecord ?
        coordinates,
    ):
        return cls(
            molecule_id=molecule_id,
            qcarchive_id=qc_record.id,
            mapped_smiles=mapped_smiles,
            coordinates=coordinates,
            energy=qc_record.energies[-1] * hartree2kcalmol,
        )


class MMConformerRecord(Record):
    molecule_id: int = Field(
        ...,
        description="The ID of the molecule in the database",
    )
    qcarchive_id: int = Field(
        ...,
        description="The ID of the molecule in the QCArchive database that this conformer corresponds to",
    )
    force_field: str = Field(
        ...,
        description="The identifier of the force field used to generate this conformer",
    )
    mapped_smiles: str = Field(
        ...,
        description="The mapped SMILES string for the molecule, stored to track atom maps",
    )
    coordinates: Array = Field(
        ...,
        description="The coordinates [Angstrom] of this conformer with shape=(n_atoms, 3).",
    )
    energy: float = Field(
        ...,
        description="The energy (kcal/mol) of this conformer as optimized by the force field.",
    )


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
        cls: type[MR],
        molecule: Molecule,
    ) -> MR:
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
