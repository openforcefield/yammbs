from pydantic.v1 import Field

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel


class MinimizedMolecule(ImmutableModel):

    final_energy: float
    mapped_smiles: str
    coordinates: Array = Field(
        "Coordinates, stored with implicit Angstrom units",
    )

    class Config:
        artbitrary_types_allowed = True


class MinimizedQCArchiveMolecule(MinimizedMolecule):
    qcarchive_id: int


class MinimizedQMDataset(ImmutableModel):

    tag: str = Field("QCArchive dataset", description="A tag for the dataset")

    version: int = Field(1, description="The version of this model")

    mm_molecules: dict[str, list[MinimizedMolecule]] = Field(
        list(),
        description="Molecules minimized with MM, keyed by the force field.",
    )


class MinimizedQCArchiveDataset(MinimizedQMDataset):

    qm_molecules: dict[str, list[MinimizedQCArchiveMolecule]] = Field(
        list(),
        description="Molecules minimized with QM",
    )
