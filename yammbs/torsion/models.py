from pydantic import ConfigDict, Field

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel


class Record(ImmutableModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        frozen=False,
    )


class TorsionRecord(Record):
    """A record which contains information for a labelled molecule. This may include the
    coordinates of the molecule in different conformers, and partial charges / WBOs
    computed for those conformers."""

    mapped_smiles: str = Field(
        ...,
        description="The mapped SMILES string of the initial molecule of this torsion drive.",
    )
    inchi_key: str = Field(..., description="The InChI string of the initial molecule of this torsion drive.")


class QMTorsionPointRecord(Record):
    """A record for a specific 'point' in a torsion scan."""

    grid_identifier: tuple[int, ...] = Field(
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
    """A record for a specific 'point' in a torsion scan after MM 'minimization.'"""

    force_field: str = Field(
        ...,
        description="The force field used to generate this conformer.",
    )
