"""Conformer generation configuration and functions."""
from typing import List

from openff.models.models import DefaultModel
from openff.toolkit.topology.molecule import Molecule
from openff.units import Quantity, unit
from rdkit.Chem import rdMolAlign


class ConformerGenerationConfig(DefaultModel):
    n_conformers: int = 10
    rms_cutoff: Quantity = Quantity(1.0, unit.angstrom)
    min_rmsd: Quantity = Quantity(2.0, unit.angstrom)
    rmsd_step: Quantity = Quantity(0.1, unit.angstrom)


def _generate_conformers(
    molecule: Molecule,
    config: ConformerGenerationConfig,
) -> List[Molecule]:
    # TODO: Probably yoink the alignment logic from
    #       https://github.com/openforcefield/openff-benchmark/blob/3b9ae9de3d19b744840c702155e8982a7df3e92e/openff/benchmark/utils/generate_conformers.py#L172

    molecule_copy = Molecule(molecule)
    del molecule

    molecule_copy.generate_conformers(
        n_conformers=config.n_conformers,
        rms_cutoff=config.rms_cutoff,
    )

    print(f"Generated {molecule_copy.n_conformers} conformers")

    return [
        _copy_molecule_with_conformer(molecule_copy, conformer)
        for conformer in molecule_copy.conformers
    ]


def _copy_molecule_with_conformer(
    molecule: Molecule,
    conformer: Quantity,
) -> Molecule:
    molecule_copy = Molecule(molecule)
    molecule_copy._conformers = [conformer]
    return molecule_copy


def _calc_rmsd(reference: Molecule, result: Molecule):
    # https://github.com/openforcefield/openff-benchmark/blob/44c8babc52aa3158f68f86687807db6759fd30c5/openff/benchmark/analysis/analysis.py#L86
    return unit.Quantity(
        rdMolAlign.GetBestRMS(reference.to_rdkit(), result.to_rdkit()),
        unit.angstrom,
    )
