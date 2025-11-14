"""Torsion-specific outputs."""

from __future__ import annotations

from pydantic import Field

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel


class MinimizedTorsionProfile(ImmutableModel):
    """The profile of a single torsion scan, including the coordinates and energies at each point."""

    mapped_smiles: str
    dihedral_indices: tuple[int, int, int, int] = Field(
        ...,
        description="The indices, 0-indexed, of the atoms which define the driven dihedral angle",
    )

    # TODO: Should this store more information than just the grid points and
    #       final geometries? i.e. each point is tagged with an ID in QCArchive
    coordinates: dict[float, Array] = Field(
        ...,
        description="A mapping between the grid angle and atomic coordinates, in Angstroms, of the molecule "
        "at that point in the torsion scan.",
    )

    energies: dict[float, float] = Field(
        ...,
        description="A mapping between the grid angle and (QM) energies, in kcal/mol, of the molecule "
        "at that point in the torsion scan.",
    )


class MinimizedTorsionDataset(ImmutableModel):
    """Dataset containing many torsion profiles, minimized with a given force field(s)."""

    tag: str = Field("QCArchive dataset", description="A tag for the dataset")

    version: int = Field(1, description="The version of this model")

    mm_torsions: dict[str, list[MinimizedTorsionProfile]] = Field(
        dict(),
        description="Torsion profiles minimized with MM, keyed by the force field.",
    )


class Metric(ImmutableModel):
    """Summary metrics for a given force field and QM reference."""

    rms_rmsd: float
    mean_rmsd: float
    rmse: float
    mean_error: float
    js_distance: tuple[float, float]


class MetricCollection(ImmutableModel):
    """Class storing metrics of potentially multiple force fields."""

    metrics: dict[str, dict[int, Metric]] = Field(
        dict(),
        description="The metrics, keyed by the QM reference ID, then keyed by force field.",
    )
