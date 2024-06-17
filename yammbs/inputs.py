from typing import TypeVar

import numpy
import qcelemental
from openff.qcsubmit.results import OptimizationResultCollection
from pydantic import Field

from yammbs._base.base import ImmutableModel

hartree2kcalmol = qcelemental.constants.hartree2kcalmol

QMD = TypeVar("QMD", bound="QMDataset")


class QMMolecule(ImmutableModel):
    id: int
    mapped_smiles: str
    coordinates: numpy.ndarray = Field(
        "Coordinates, stored with implicit Angstrom units",
    )

    class Config:
        artbitrary_types_allowed = True


class QCArchiveMolecule(QMMolecule):
    qcarchive_id: int
    final_energy: float


class QMDataset(ImmutableModel):

    name: str

    qm_molecules: list[QMMolecule] = Field(
        list(),
        description="A list of QM molecules in the dataset",
    )

    class Config:
        artbitrary_types_allowed = True


class QCArchiveDataset(QMDataset):

    qm_molecules: list[QCArchiveMolecule] = Field(
        list(),
        description="A list of QM molecules in the dataset",
    )

    @classmethod
    def from_qcsubmit_collection(
        cls,
        collection: OptimizationResultCollection,
    ) -> QMD:
        return cls(
            name="foobar",
            qm_molecules=[
                QCArchiveMolecule(
                    id=id,
                    qcarchive_id=qcarchive_record.id,
                    mapped_smiles=molecule.to_smiles(
                        mapped=True,
                        isomeric=True,
                        explicit_hydrogens=True,
                    ),
                    coordinates=molecule.conformers[0].m_as("angstrom"),
                    final_energy=qcarchive_record.energies[-1] * hartree2kcalmol,
                )
                for id, (qcarchive_record, molecule) in enumerate(
                    collection.to_records(),
                )
            ],
        )
