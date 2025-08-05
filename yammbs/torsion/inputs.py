"""Input models for torsion datasets."""

import logging
from collections.abc import Sequence

import qcelemental
from openff.qcsubmit.results import TorsionDriveResultCollection
from pydantic import Field

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel

hartree2kcalmol = qcelemental.constants.hartree2kcalmol
bohr2angstroms = qcelemental.constants.bohr2angstroms


LOGGER = logging.getLogger(__name__)


class TorsionDataset(ImmutableModel):
    """Base class for a torsion dataset."""

    tag: str


class TorsionProfile(ImmutableModel):
    """Base class for a torsion profile."""

    mapped_smiles: str
    dihedral_indices: tuple[int, int, int, int] = Field(
        ...,
        description="The indices, 0-indexed, of the atoms which define the driven dihedral angle",
    )
    qcarchive_id: int = Field(
        ...,
        description="The ID of the torsion profile in QCArchive, probably the same as the TorsiondriveRecord.id",
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


class QCArchiveTorsionProfile(TorsionProfile):
    """A single QCArchive torsion profile."""

    id: int = Field(..., description="The attribute TorsiondriveRecord.id")


class QCArchiveTorsionDataset(TorsionDataset):
    """Store a collection of torsion profiles from QCArchive."""

    tag: str = Field("QCArchive torsiondrive dataset", description="A tag for the dataset")

    version: int = Field(1, description="The version of this model")

    qm_torsions: Sequence[QCArchiveTorsionProfile] = Field(
        list(),
        description="A list of QM-drived torsion profiles in the dataset",
    )

    @classmethod
    def from_qcsubmit_collection(
        cls,
        collection: TorsionDriveResultCollection,
    ) -> "QCArchiveTorsionDataset":
        """Create a QCArchiveTorsionDataset from a TorsionDriveResultCollection."""
        LOGGER.info(
            "Converting a TorsionDriveResultCollection (a QCSubmit model) "
            "to a QCArchiveTorsionDataset (a YAMMBS model)",
        )

        return cls(
            qm_torsions=[
                QCArchiveTorsionProfile(
                    id=record.id,
                    mapped_smiles=molecule.to_smiles(
                        mapped=True,
                        isomeric=True,
                        explicit_hydrogens=True,
                    ),
                    dihedral_indices=record.specification.keywords.dihedrals[
                        0
                    ],  # might be 2-D in the future, 1-D for now
                    qcarchive_id=record.id,
                    coordinates={
                        grid_id[0]: optimization.final_molecule.geometry * bohr2angstroms
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
