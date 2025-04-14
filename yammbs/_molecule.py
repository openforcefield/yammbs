"""Molecule conversion utilities"""

from functools import lru_cache

from openff.toolkit import Molecule, Quantity

from yammbs._base.array import Array


def _molecule_with_conformer_from_smiles(
    mapped_smiles: str,
    conformer: Array,
) -> Molecule:
    """Create a molecule from mapped SMILES and attach a single conformer."""
    molecule = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)
    molecule.add_conformer(Quantity(conformer, "angstrom"))

    return molecule


@lru_cache
def _smiles_to_inchi_key(smiles: str) -> str:
    from openff.toolkit import Molecule

    return Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True).to_inchi(
        fixed_hydrogens=True,
    )
