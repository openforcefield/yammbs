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
from yammbs._minimize import (
    _DEFAULT_ENERGY_MINIMIZATION_MAX_ITERATIONS,
    _DEFAULT_ENERGY_MINIMIZATION_TOLERANCE,
    _minimize_openmm,
)

LOGGER = logging.getLogger(__name__)

_DEFAULT_TORSION_RESTRAINT_K: float = 100_000
"""Default torsion restraint force constant in kcal/(mol·rad²)."""

_DEFAULT_ANGLE_TOLERANCE: float = 0.1
"""Default tolerance in degrees for the post-minimization dihedral sanity check."""

_ConstrainedMinimizationFn = Callable[
    [
        Molecule,
        openmm.System,
        numpy.ndarray,
        tuple[int, int, int, int],
        float,
        int,
    ],
    tuple[numpy.ndarray, float],
]

# Might be overkill to use a registry at present with only two methods,
# but makes it more extensible
_CONSTRAINED_MINIMIZATION_REGISTRY: dict[str, _ConstrainedMinimizationFn] = {}


def _register_minimization_method(
    name: str,
) -> Callable[[_ConstrainedMinimizationFn], _ConstrainedMinimizationFn]:
    """Register a constrained minimization function under the given method name."""

    def decorator(fn: _ConstrainedMinimizationFn) -> _ConstrainedMinimizationFn:
        _CONSTRAINED_MINIMIZATION_REGISTRY[name] = fn
        return fn

    return decorator


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

    method: Literal["openmm_torsion_atoms_frozen", "openmm_torsion_restrained"] = Field(
        "openmm_torsion_restrained",
        description="The minimization method to use.",
    )

    @property
    def constrained_minimization_function(self) -> _ConstrainedMinimizationFn:
        """Get the minimization function from the registry."""
        return _CONSTRAINED_MINIMIZATION_REGISTRY[self.method]


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
    method: Literal["openmm_torsion_atoms_frozen", "openmm_torsion_restrained"] = "openmm_torsion_restrained",
    n_processes: int = 2,
    chunksize=32,
    restraint_k: float = 0.0,
) -> Generator[ConstrainedMinimizationResult, None, None]:
    LOGGER.info("Mapping `data` generator into `inputs` generator")

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

    LOGGER.info("Setting up multiprocessing pool with generator (of unknown length)")

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


def _find_unused_force_group(system: openmm.System) -> int:
    """Return the highest-numbered force group not currently used by any force in the system.

    OpenMM supports force groups 0–31.

    Raises:
        RuntimeError: If all 32 force groups are already in use.

    """
    used = {system.getForce(i).getForceGroup() for i in range(system.getNumForces())}
    for group in range(31, -1, -1):
        if group not in used:
            return group
    raise RuntimeError("All 32 force groups are already in use; cannot add restraint forces.")


def _restrain_omm_system(
    mol: Molecule,
    system: openmm.System,
    positions: numpy.ndarray,
    dihedral_indices: tuple[int, int, int, int],
    restraint_k: float,
    force_group: int = 31,
) -> None:
    """Add a restraint to all atoms except those in the dihedral."""
    restraint_force = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint_force.addGlobalParameter(
        "k",
        restraint_k * openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**2,
    )
    for parameter in ("x0", "y0", "z0"):
        restraint_force.addPerParticleParameter(parameter)

    LOGGER.debug(f"Adding restraint to particles not in {dihedral_indices=}")
    for atom_index in range(mol.n_atoms):
        if atom_index in dihedral_indices:
            continue

        particle_index = restraint_force.addParticle(atom_index)
        restraint_force.setParticleParameters(
            particle_index,
            atom_index,
            [(x * openmm.unit.angstrom).in_units_of(openmm.unit.nanometer) for x in positions[atom_index]],
        )

    restraint_force.setForceGroup(force_group)
    system.addForce(restraint_force)


def _add_torsion_restraint_to_omm_system(
    system: openmm.System,
    dihedral_indices: tuple[int, int, int, int],
    target_angle: float,
    force_group: int = 1,
    torsion_restraint_k: float = _DEFAULT_TORSION_RESTRAINT_K,
) -> None:
    """Add a harmonic torsion restraint to maintain the target dihedral angle.

    The restraint is added to a separate force group so it guides minimization
    but doesn't contribute to the final reported energy.

    Uses a periodic-aware potential to properly handle angle wrapping
    (e.g., -180° and 180° are treated as the same angle).

    Args:
        system: The OpenMM system to modify
        dihedral_indices: The four atom indices defining the dihedral
        target_angle: Target dihedral angle in degrees
        force_group: Force group to assign the restraint to (default: 1)
        torsion_restraint_k: Force constant for the torsion restraint in kcal/(mol*rad^2)

    """
    # Create CustomTorsionForce with periodic harmonic potential
    # dtheta is the shortest angular distance between theta and theta0
    # This ensures -180° and 180° are treated as equivalent
    torsion_restraint = openmm.CustomTorsionForce(
        "0.5*k_torsion*dtheta^2; dtheta = atan2(sin(theta-theta0), cos(theta-theta0))",
    )

    # Add force constant in kcal/(mol*rad^2) to ensure that the angle is
    # ~ constrained during minimisation.
    torsion_restraint.addGlobalParameter(
        "k_torsion",
        torsion_restraint_k * openmm.unit.kilocalorie_per_mole / openmm.unit.radian**2,
    )

    # Add per-torsion parameter for target angle
    torsion_restraint.addPerTorsionParameter("theta0")

    # Convert target angle from degrees to radians
    target_angle_rad = target_angle * numpy.pi / 180.0

    # Add the torsion with its target angle
    torsion_restraint.addTorsion(
        int(dihedral_indices[0]),
        int(dihedral_indices[1]),
        int(dihedral_indices[2]),
        int(dihedral_indices[3]),
        [target_angle_rad],
    )

    # Assign to separate force group so it doesn't contribute to final energy
    torsion_restraint.setForceGroup(force_group)

    LOGGER.debug(
        f"Adding torsion restraint to {dihedral_indices=} with {target_angle=} degrees to force group {force_group}",
    )
    system.addForce(torsion_restraint)


def _angular_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees, accounting for periodicity."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _zero_masses_of_dihedral_atoms(
    system: openmm.System,
    dihedral_indices: tuple[int, int, int, int],
) -> None:
    """Set the masses of the dihedral atoms to zero to 'constrain' them minimization."""
    LOGGER.debug(f"Adding restraint to particles not in {dihedral_indices=}")
    for index in dihedral_indices:
        system.setParticleMass(index, 0.0)


@_register_minimization_method("openmm_torsion_atoms_frozen")
def _minimize_openmm_atoms_frozen(
    mol: Molecule,
    system: openmm.System,
    positions: numpy.ndarray,
    dihedral_indices: tuple[int, int, int, int],
    angle: float,
    restraint_force_group: int,
) -> tuple[numpy.ndarray, float]:
    """Minimize a molecule with OpenMM with 'constraints' on a dihedral."""
    # Add the "dihedral constraint" by zeroing the masses of the dihedral atoms
    _zero_masses_of_dihedral_atoms(system=system, dihedral_indices=dihedral_indices)

    return _minimize_openmm(
        mol=mol,
        system=system,
        positions=positions,
    )


@_register_minimization_method("openmm_torsion_restrained")
def _minimize_openmm_torsion_restrained(
    mol: Molecule,
    system: openmm.System,
    positions: numpy.ndarray,
    dihedral_indices: tuple[int, int, int, int],
    angle: float,
    restraint_force_group: int,
) -> tuple[numpy.ndarray, float]:
    """Minimize a molecule with OpenMM with a strong harmonic restraint on a dihedral.

    Unlike _minimize_openmm_constrained which zeros masses (preventing movement),
    this method uses a CustomTorsionForce to apply a strong harmonic restraint
    on the dihedral angle, allowing atoms to move while maintaining the target angle.

    The restraint is added to a separate force group and excluded from the final
    energy calculation, so the reported energy is the actual molecular mechanics
    energy without the artificial restraint contribution.
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis.dihedrals import Dihedral

    # Add the torsion restraint to maintain the target angle
    _add_torsion_restraint_to_omm_system(
        system=system,
        dihedral_indices=dihedral_indices,
        target_angle=angle,
        force_group=restraint_force_group,
        torsion_restraint_k=_DEFAULT_TORSION_RESTRAINT_K,
    )

    # Perform minimization with custom energy evaluation
    # that excludes the restraint force group
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.1 * openmm.unit.femtoseconds),
        openmm.Platform.getPlatformByName("Reference"),
    )

    context.setPositions(
        (positions * openmm.unit.angstrom).in_units_of(openmm.unit.nanometer),
    )
    context.computeVirtualSites()

    # Sanity check: verify the initial dihedral angle
    u_initial = mda.Universe.empty(n_atoms=len(positions), trajectory=True)
    u_initial.load_new(positions, order="fac")

    dihedral_calc_initial = Dihedral([u_initial.atoms[list(dihedral_indices)]])
    dihedral_calc_initial.run()
    initial_angle = dihedral_calc_initial.results.angles[0][0]

    # Calculate initial angle difference accounting for periodicity
    initial_angle_diff = _angular_diff(initial_angle, angle)

    LOGGER.info(
        f"Initial dihedral angle: {initial_angle:.2f}° (target: {angle:.2f}°, diff: {initial_angle_diff:.2f}°)",
    )

    # Log initial energy (excluding restraint)
    groups_mask = sum(1 << group for group in range(32) if group != restraint_force_group)
    initial_energy = (
        context.getState(getEnergy=True, groups=groups_mask)
        .getPotentialEnergy()
        .value_in_unit(openmm.unit.kilocalorie_per_mole)
    )
    LOGGER.info(f"Initial energy (excluding restraint): {initial_energy} kcal/mol")

    # Minimize (restraint is active during minimization)
    openmm.LocalEnergyMinimizer.minimize(
        context=context,
        tolerance=_DEFAULT_ENERGY_MINIMIZATION_TOLERANCE.to_openmm(),
        maxIterations=_DEFAULT_ENERGY_MINIMIZATION_MAX_ITERATIONS,
    )

    # Get final state excluding restraint force group
    final_state = context.getState(getPositions=True, getEnergy=True, groups=groups_mask)

    final_positions = final_state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)

    final_energy = final_state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole)

    LOGGER.info(f"Final energy (excluding restraint): {final_energy} kcal/mol")

    # Sanity check: verify the final dihedral angle matches the target
    u_final = mda.Universe.empty(n_atoms=len(final_positions), trajectory=True)
    u_final.load_new(final_positions, order="fac")

    dihedral_calc_final = Dihedral([u_final.atoms[list(dihedral_indices)]])
    dihedral_calc_final.run()
    final_angle = dihedral_calc_final.results.angles[0][0]

    # Calculate angle difference accounting for periodicity
    final_angle_diff = _angular_diff(final_angle, angle)

    # Raise if restraint didn't maintain the angle within tolerance
    if final_angle_diff > _DEFAULT_ANGLE_TOLERANCE:
        raise ConstrainedMinimizationError(
            f"Torsion restraint sanity check failed: "
            f"initial={initial_angle:.2f}°, target={angle:.2f}°, final={final_angle:.2f}°, "
            f"diff={final_angle_diff:.2f}° exceeds tolerance of {_DEFAULT_ANGLE_TOLERANCE:.2f}°",
        )
    LOGGER.info(f"Final dihedral angle: {final_angle:.2f}° (target: {angle:.2f}°, diff: {final_angle_diff:.2f}°)")

    return final_positions, final_energy


def _run_minimization_constrained(
    input: ConstrainedMinimizationInput,
) -> ConstrainedMinimizationResult | None:
    """Taken from openff-strike-team 10/31/24.

    https://github.com/lilyminium/openff-strike-team/blob/a6ccd2821ed627064529f5c4a22b47c1fa36efe2/torsions/datasets/mm/minimize-torsion-constrained.py#L35-L106
    """
    from openff.interchange.exceptions import UnassignedValenceError
    from openff.toolkit import Molecule

    LOGGER.info(f"############ Method: {input.method} ############")

    LOGGER.debug(f"Setting up constrained minimization for {input.model_dump()=}")

    LOGGER.debug(f"Creating molecule from {input.mapped_smiles=}")
    molecule = Molecule.from_mapped_smiles(input.mapped_smiles, allow_undefined_stereo=True)
    # molecule.add_conformer(Quantity(input.coordinates, "angstrom"))

    LOGGER.debug(f"Creating OpenMM system with force field {input.force_field=}")
    try:
        system = build_omm_system(
            force_field=input.force_field,
            molecule=molecule,
        )
    except UnassignedValenceError:
        LOGGER.warning(f"Skipping record {input.torsion_id} with unassigned valence terms")
        return None
    except (RuntimeError, ValueError) as e:  # charging error
        LOGGER.warning(f"Skipping record {input.torsion_id} with a value error (probably a charge failure): {e}")
        return None

    atom_indices = list(range(len(molecule.atoms)))
    atom_indices = sorted(set(atom_indices))  # - set([index - 0 for index in input.dihedral_indices]))

    restraint_group = _find_unused_force_group(system)

    # Add the restraint force to the system
    _restrain_omm_system(
        mol=molecule,
        system=system,
        positions=input.coordinates,
        dihedral_indices=input.dihedral_indices,
        restraint_k=input.restraint_k,
        force_group=restraint_group,
    )

    LOGGER.debug("Trying to minimize energy")
    try:
        final_positions, final_energy = input.constrained_minimization_function(
            molecule,
            system,
            input.coordinates,
            input.dihedral_indices,
            input.grid_id,
            restraint_group,
        )
    except Exception as e:
        raise ConstrainedMinimizationError(f"Minimization failed for {input=} : {e}") from e

    LOGGER.debug("Returning result")
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
