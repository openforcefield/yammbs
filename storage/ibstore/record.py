"""This module defines the data models used to store the data in the database."""
from typing import TYPE_CHECKING, Dict, List

import numpy as np
from openff.nagl._base.array import Array
from openff.nagl._base.base import ImmutableModel
from openff.nagl.toolkits.openff import map_indexed_smiles
from openff.toolkit import Molecule
from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY
from openff.units import unit
from pydantic import Field, validator

if TYPE_CHECKING:
    import openff.toolkit

__all__ = [
    "ConformerRecord",
    "MoleculeRecord",
]


class Record(ImmutableModel):
    class Config(ImmutableModel.Config):
        orm_mode = True


class ConformerRecord(Record):
    """A record which stores the coordinates of a molecule in a particular conformer,
    as well as sets of partial charges and WBOs computed using this conformer and
    for different methods."""

    coordinates: Array = Field(
        ...,
        description=(
            "The coordinates [Angstrom] of this conformer with shape=(n_atoms, 3)."
        ),
    )

    @validator("coordinates")
    def _validate_coordinate_type(cls, v):
        if isinstance(v, unit.Quantity):
            v = v.m_as(unit.angstrom)
        else:
            raise ValueError("coordinates must be a Quantity")
        return v

    @validator("coordinates")
    def _validate_coordinates(cls, v):
        try:
            v = v.reshape((-1, 3))
        except ValueError:
            raise ValueError("coordinates must be re-shapable to `(n_atoms, 3)`")
        v.flags.writeable = False
        return v

    def map_to(self, mapping: Dict[int, int]) -> "ConformerRecord":
        """Map the conformer to a new set of atom indices"""
        index_array = np.array(list(mapping.values()))
        inverse_map = {v: k for k, v in mapping.items()}
        return type(self)(
            coordinates=self.coordinates[index_array],
            partial_charges={
                method: charges[index_array]
                for method, charges in self.partial_charges.items()
            },
            bond_orders={
                method: bond_orders.map_to(inverse_map)
                for method, bond_orders in self.bond_orders.items()
            },
        )


class MoleculeRecord(Record):
    """A record which contains information for a labelled molecule. This may include the
    coordinates of the molecule in different conformers, and partial charges / WBOs
    computed for those conformers."""

    qcarchive_id: str = Field(
        ...,
        description="The ID of the molecule in the QCArchive database",
    )
    mapped_smiles: str = Field(
        ...,
        description="The mapped SMILES string for the molecule with hydrogens specified",
    )
    conformers: List[ConformerRecord] = Field(
        ...,
        description=("Conformers associated with the molecule. "),
    )

    @property
    def smiles(self):
        return self.mapped_smiles

    @classmethod
    def from_qcsubmit_record(
        cls,
        qcarchive_id: str,
        molecule: Molecule,
    ):
        return cls(
            qcarchive_id=qcarchive_id,
            mapped_smiles=molecule.to_smiles(mapped=True, isomeric=True),
            conformers=[
                ConformerRecord(coordinates=conformer.m_as(unit.angstrom))
                for conformer in molecule.conformers
            ],
        )

    def to_openff(
        self,
        toolkit_registry=GLOBAL_TOOLKIT_REGISTRY,
    ) -> "openff.toolkit.topology.Molecule":
        """Convert the record to an OpenFF molecule with averaged properties"""

        from openff.toolkit.topology.molecule import Molecule
        from openff.toolkit.topology.molecule import unit as off_unit

        offmol = Molecule.from_mapped_smiles(
            self.mapped_smiles,
            allow_undefined_stereo=True,
            toolkit_registry=toolkit_registry,
        )
        offmol._conformers = [
            conformer.coordinates * off_unit.angstrom for conformer in self.conformers
        ]

        return offmol

    def reorder(self, target_mapped_smiles: str) -> "MoleculeRecord":
        """Reorder the molecule to match the target mapped SMILES"""
        if self.mapped_smiles == target_mapped_smiles:
            return self

        atom_map = map_indexed_smiles(target_mapped_smiles, self.mapped_smiles)
        self_to_target = {k: atom_map[k] for k in sorted(atom_map)}

        return type(self)(
            mapped_smiles=target_mapped_smiles,
            conformers=[
                conformer.map_to(self_to_target) for conformer in self.conformers
            ],
        )
