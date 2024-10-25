from typing import Sequence

from openff.qcsubmit.results import TorsionDriveResultCollection
from pydantic import Field

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel


class TorsionDataset(ImmutableModel):
    tag: str


class TorsionProfile(ImmutableModel):
    mapped_smiles: str

    # TODO: Should this store more information than just the grid points and
    #       final geometries? i.e. each point is tagged with an ID in QCArchive
    points: dict[float, Array]


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
                TorsionProfile(
                    mapped_smiles=molecule.to_smiles(
                        mapped=True,
                        isomeric=True,
                        explicit_hydrogens=True,
                    ),
                    points={
                        grid_id[0]: optimization.final_molecule.geometry
                        for grid_id, optimization in record.minimum_optimizations.items()
                    },
                )
                for record, molecule in collection.to_records()
            ],
        )
