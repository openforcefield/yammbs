from typing import Sequence

import qcelemental
from openff.qcsubmit.results import TorsionDriveResultCollection
from pydantic import Field

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel

hartree2kcalmol = qcelemental.constants.hartree2kcalmol
bohr2angstroms = qcelemental.constants.bohr2angstroms


class TorsionDataset(ImmutableModel):
    tag: str


class TorsionProfile(ImmutableModel):
    mapped_smiles: str
    dihedral_indices: list[int]

    # TODO: Should this store more information than just the grid points and
    #       final geometries? i.e. each point is tagged with an ID in QCArchive
    coordinates: dict[float, Array]

    energies: dict[float, float]


class QCArchiveTorsionProfile(TorsionProfile):
    id: int = Field(..., description="The attribute TorsiondriveRecord.id")


class QCArchiveTorsionDataset(TorsionDataset):
    tag: str = Field("QCArchive torsiondrive dataset", description="A tag for the dataset")

    version: int = Field(1, description="The version of this model")

    qm_torsions: Sequence[TorsionProfile] = Field(
        list(),
        description="A list of QM-drived torsion profiles in the dataset",
    )

    @classmethod
    def from_qcsubmit_collection(
        cls,
        collection: TorsionDriveResultCollection,
    ) -> "QCArchiveTorsionDataset":
        return cls(
            qm_torsions=[
                QCArchiveTorsionProfile(
                    id=record.id,
                    mapped_smiles=molecule.to_smiles(
                        mapped=True,
                        isomeric=True,
                        explicit_hydrogens=True,
                    ),
                    dihedral_indices=record.specification.keywords.dihedrals[0],  # assuming this is only ever 1-len?
                    coordinates={
                        grid_id[0]: optimization.final_molecule.geometry
                        for grid_id, optimization in record.minimum_optimizations.items()
                    },
                    energies={
                        grid_id[0]: optimization.energies[-1] * hartree2kcalmol
                        for grid_id, optimization in record.minimum_optimizations.items()
                    },
                )
                for record, molecule in collection.to_records()
            ],
        )
