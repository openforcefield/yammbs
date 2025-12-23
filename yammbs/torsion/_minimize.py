import logging
from collections.abc import Callable, Generator
from multiprocessing import Pool
from typing import Literal

import numpy
import openmm
from numpy.typing import NDArray
from openff.toolkit import Molecule
from pydantic import Field
from tqdm import tqdm

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel
from yammbs._forcefields import build_omm_system
from yammbs._minimize import _minimize_geometric, _minimize_openmm

logger = logging.getLogger(__name__)

_ConstrainedMinimizationFn = Callable[
    [
        Molecule,
        openmm.System,
        numpy.ndarray,
        tuple[int, int, int, int],
        float,
    ],
    tuple[numpy.ndarray, float],
]


class ConstrainedMinimizationInput(ImmutableModel):
    torsion_id: int = Field(
        ...,
        description="The identifier of the torsion unit",
    )
    mapped_smiles: str = Field(
        ...,
        description="The SMILES of the molecule",
    )
    dihedral_indices: tuple[int, int, int, int] = Field(
        ...,
        description="The indices of the atoms which define the driven dihedral angle",
    )
    force_field: str = Field(
        ...,
        description="And identifier of the force field to use for the minimization",
    )
    coordinates: Array = Field(
        ...,
        description="The coordinates [Angstrom] of this conformer with shape=(n_atoms, 3).",
    )
    grid_id: float = Field(
        ...,
        description="The grid identifier of the torsion scan point.",
    )
    restraint_k: float = Field(
        default=0.0,
        description="Restraint force constant in kcal/(mol*Angstrom^2) for atoms not in dihedral.",
    )

    method: Literal["openmm", "geometric"] = Field(
        "openmm",
        description="The minimization method to use.",
    )

    @property
    def constrained_minimization_function(self) -> _ConstrainedMinimizationFn:
        return _minimize_openmm_constrained if self.method == "openmm" else _minimize_geometric_constrained


class ConstrainedMinimizationResult(ConstrainedMinimizationInput):
    energy: float = Field(
        ...,
        description="Minimized energy in kcal/mol",
    )


def _minimize_torsions(
    data: Generator[
        tuple[
            int,
            str,
            tuple[int, int, int, int],
            float,
            NDArray,
            float,
        ],
        None,
        None,
    ],
    force_field: str,
    method: Literal["openmm", "geometric"] = "openmm",
    n_processes: int = 2,
    chunksize=32,
    restraint_k: float = 0.0,
) -> Generator[ConstrainedMinimizationResult, None, None]:
    logger.info("Mapping `data` generator into `inputs` generator")

    # It'd be smoother to skip this tranformation - just pass this generator
    # from inside of TorsionStore
    inputs: Generator[ConstrainedMinimizationInput, None, None] = (
        ConstrainedMinimizationInput(
            torsion_id=torsion_id,
            mapped_smiles=mapped_smiles,
            dihedral_indices=dihedral_indices,
            force_field=force_field,
            coordinates=coordinates,
            grid_id=grid_id,
            method=method,
            restraint_k=restraint_k,
        )
        for (
            torsion_id,
            mapped_smiles,
            dihedral_indices,
            grid_id,
            coordinates,
            _,
        ) in data
    )

    logger.info("Setting up multiprocessing pool with generator (of unknown length)")

    # TODO: It'd be nice to have the `total` argument passed through, but that would require using
    #       a list-like iterable instead of a generator, which might cause problems at scale
    with Pool(processes=n_processes) as pool:
        for val in tqdm(
            pool.imap(
                _run_minimization_constrained,
                inputs,
                chunksize=chunksize,
            ),
            desc=f"Building and minimizing systems with {force_field}",
        ):
            if val is not None:
                yield val


class ConstrainedMinimizationError(Exception):
    """The constrained minimization failed."""

    pass


def _restrain_omm_system(
    mol: Molecule,
    system: openmm.System,
    positions: numpy.ndarray,
    dihedral_indices: tuple[int, int, int, int],
    restraint_k: float,
) -> None:
    """Add a restraint to all atoms except those in the dihedral."""
    restraint_force = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint_force.addGlobalParameter(
        "k",
        restraint_k * openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**2,
    )
    for parameter in ("x0", "y0", "z0"):
        restraint_force.addPerParticleParameter(parameter)

    logger.debug(f"Adding restraint to particles not in {dihedral_indices=}")
    for atom_index in range(mol.n_atoms):
        if atom_index in dihedral_indices:
            continue

        particle_index = restraint_force.addParticle(atom_index)
        restraint_force.setParticleParameters(
            particle_index,
            atom_index,
            [(x * openmm.unit.angstrom).in_units_of(openmm.unit.nanometer) for x in positions[atom_index]],
        )

    system.addForce(restraint_force)


def _zero_masses_of_dihedral_atoms(
    system: openmm.System,
    dihedral_indices: tuple[int, int, int, int],
) -> None:
    """Set the masses of the dihedral atoms to zero to 'constrain' them minimization."""
    logger.debug(f"Adding restraint to particles not in {dihedral_indices=}")
    for index in dihedral_indices:
        system.setParticleMass(index, 0.0)


def _minimize_openmm_constrained(
    mol: Molecule,
    system: openmm.System,
    positions: numpy.ndarray,
    dihedral_indices: tuple[int, int, int, int],
    angle: float,
) -> tuple[numpy.ndarray, float]:
    """Minimize a molecule with OpenMM with 'constraints' on a dihedral."""
    # Add the "dihedral constraint" by zeroing the masses of the dihedral atoms
    _zero_masses_of_dihedral_atoms(system=system, dihedral_indices=dihedral_indices)

    return _minimize_openmm(
        mol=mol,
        system=system,
        positions=positions,
    )


def _minimize_geometric_constrained(
    mol: Molecule,
    system: openmm.System,
    positions: numpy.ndarray,
    dihedral_indices: tuple[int, int, int, int],
    angle: float,
) -> tuple[numpy.ndarray, float]:
    """Minimize a molecule with Geometric with the specified dihedral constrained."""
    constraints = {"set": [{"indices": dihedral_indices, "type": "dihedral", "value": angle}]}

    return _minimize_geometric(
        mol=mol,
        system=system,
        positions=positions,
        constraints=constraints,
    )


def _run_minimization_constrained(
    input: ConstrainedMinimizationInput,
) -> ConstrainedMinimizationResult:
    """Taken from openff-strike-team 10/31/24.

    https://github.com/lilyminium/openff-strike-team/blob/a6ccd2821ed627064529f5c4a22b47c1fa36efe2/torsions/datasets/mm/minimize-torsion-constrained.py#L35-L106
    """
    from openff.toolkit import Molecule

    logger.info(f"############ Method: {input.method} ############")

    logger.debug(f"Setting up constrained minimization for {input.model_dump()=}")

    logger.debug(f"Creating molecule from {input.mapped_smiles=}")
    molecule = Molecule.from_mapped_smiles(input.mapped_smiles, allow_undefined_stereo=True)
    # molecule.add_conformer(Quantity(input.coordinates, "angstrom"))

    logger.debug(f"Creating OpenMM system with force field {input.force_field=}")
    # TODO: Add same error handling as in _run_minimization?
    system = build_omm_system(
        force_field=input.force_field,
        molecule=molecule,
    )

    atom_indices = list(range(len(molecule.atoms)))
    atom_indices = sorted(set(atom_indices))  # - set([index - 0 for index in input.dihedral_indices]))

    # Add the restraint force to the system
    _restrain_omm_system(
        mol=molecule,
        system=system,
        positions=input.coordinates,
        dihedral_indices=input.dihedral_indices,
        restraint_k=input.restraint_k,
    )

    logger.debug("Trying to minimize energy")
    try:
        final_positions, final_energy = input.constrained_minimization_function(
            molecule,
            system,
            input.coordinates,
            input.dihedral_indices,
            input.grid_id,
        )
    except Exception as e:
        raise ConstrainedMinimizationError(f"Minimization failed for {input=} : {e}") from e

    logger.debug("Returning result")
    return ConstrainedMinimizationResult(
        torsion_id=input.torsion_id,
        mapped_smiles=input.mapped_smiles,
        dihedral_indices=input.dihedral_indices,
        force_field=input.force_field,
        coordinates=final_positions,
        energy=final_energy,
        grid_id=input.grid_id,
        method=input.method,
    )
