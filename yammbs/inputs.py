"""Input models for minimizations."""

from collections.abc import Sequence
from typing import TypeVar

import qcelemental
from openff.qcsubmit.results import OptimizationResultCollection
from pydantic import Field

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel

hartree2kcalmol = qcelemental.constants.hartree2kcalmol

QMD = TypeVar("QMD", bound="QMDataset")


class QMMolecule(ImmutableModel):
    """A base model for a QM molecule."""

    id: int
    mapped_smiles: str
    coordinates: Array = Field(
        description="Coordinates, stored with implicit Angstrom units",
    )


class QCArchiveMolecule(QMMolecule):
    """A model for a QM molecule from QCArchive."""

    qcarchive_id: int
    final_energy: float


class QMDataset(ImmutableModel):
    """Base model for a dataset of QM molecules."""

    tag: str

    qm_molecules: Sequence[QMMolecule] = Field(
        list(),
        description="A list of QM molecules in the dataset",
    )


class QCArchiveDataset(QMDataset):
    """A dataset of QCArchive molecules."""

    tag: str = Field("QCArchive dataset", description="A tag for the dataset")

    version: int = Field(1, description="The version of this model")

    qm_molecules: Sequence[QCArchiveMolecule] = Field(
        list(),
        description="A list of QM molecules in the dataset",
    )

    @classmethod
    def from_qcsubmit_collection(
        cls,
        collection: OptimizationResultCollection,
    ) -> "QCArchiveDataset":
        """Create a QCArchiveDataset from a QCSubmit collection."""
        return cls(
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
