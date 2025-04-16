"""Output models for minimizations."""

from __future__ import annotations

from pydantic import Field

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel


class MinimizedMolecule(ImmutableModel):
    """Base model for information about a minimized molecule."""

    final_energy: float
    mapped_smiles: str
    coordinates: Array = Field(
        description="Coordinates, stored with implicit Angstrom units",
    )


class MinimizedQCArchiveMolecule(MinimizedMolecule):
    """A model for a QCArchive molecule which was minimized."""

    qcarchive_id: int


class MinimizedQMDataset(ImmutableModel):
    """Base model for a dataset of minimized molecules."""

    tag: str = Field("QCArchive dataset", description="A tag for the dataset")

    version: int = Field(1, description="The version of this model")

    mm_molecules: dict[str, list[MinimizedMolecule]] = Field(
        dict(),
        description="Molecules minimized with MM, keyed by the force field.",
    )


class MinimizedQCArchiveDataset(MinimizedQMDataset):
    """A model for a dataset of minimized molecules from QCArchive."""

    qm_molecules: dict[str, list[MinimizedQCArchiveMolecule]] = Field(
        dict(),
        description="Molecules minimized with QM",
    )


class Metric(ImmutableModel):
    """A model for small molecule metrics."""

    dde: float | None
    rmsd: float
    tfd: float
    icrmsd: dict[str, float | None]  # This could maybe be NamedTuple for cleaner typing


class MetricCollection(ImmutableModel):
    """A collection of metric models."""

    metrics: dict[str, dict[int, Metric]] = Field(
        dict(),
        description="The metrics, keyed by the QM reference ID, then keyed by force field.",
    )
