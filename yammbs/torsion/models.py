"""Torsion-specific models."""

from pydantic import Field

from yammbs._base.array import Array
from yammbs.models import MoleculeRecord, Record


class TorsionRecord(MoleculeRecord):
    """Base class for torsion records."""

    torsion_id: int = Field(
        ...,
        description="The ID of the torsion profile in the database",
    )
    dihedral_indices: tuple[int, int, int, int] = Field(
        ...,
        description="The indices of the atoms which define the driven dihedral angle",
    )


class QMTorsionPointRecord(Record):
    """A record for a specific 'point' in a torsion scan."""

    torsion_id: int = Field(
        ...,
        description="The ID of the torsion profile in the database",
    )

    # TODO: This needs to be a tuple[float] for 2D grids?
    grid_id: float = Field(
        ...,
        description="The grid identifier of the torsion scan point.",
    )

    coordinates: Array = Field(
        ...,
        description="The coordinates [Angstrom] of this conformer with shape=(n_atoms, 3).",
    )
    energy: float = Field(
        ...,
        description="The final energy (kcal/mol) of the molecule as optimized and computed by QCArchive",
    )


class MMTorsionPointRecord(QMTorsionPointRecord):
    """A record for a specific 'point' in a torsion scan after MM minimization."""

    force_field: str = Field(
        ...,
        description="The force field used to generate this conformer.",
    )
