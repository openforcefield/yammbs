import logging
from collections.abc import Generator
from multiprocessing import Pool

from numpy.typing import NDArray
from pydantic import Field
from tqdm import tqdm

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel
from yammbs._minimize import _lazy_load_force_field

LOGGER = logging.getLogger(__name__)


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
    n_processes: int = 2,
    chunksize=32,
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
        )
        for (torsion_id, mapped_smiles, dihedral_indices, grid_id, coordinates, _) in data
    )

    LOGGER.info("Setting up multiprocessing pool with generator (of unknown length)")

    # TODO: It'd be nice to have the `total` argument passed through, but that would require using
    #       a list-like iterable instead of a generator, which might cause problems at scale
    with Pool(processes=n_processes) as pool:
        for val in tqdm(
            pool.imap(
                _minimize_constrained,
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


def _minimize_constrained(
    input: ConstrainedMinimizationInput,
) -> ConstrainedMinimizationResult:
    """Taken from openff-strike-team 10/31/24.

    https://github.com/lilyminium/openff-strike-team/blob/a6ccd2821ed627064529f5c4a22b47c1fa36efe2/torsions/datasets/mm/minimize-torsion-constrained.py#L35-L106
    """
    import openmm
    import openmm.unit
    from openff.interchange.operations.minimize import _DEFAULT_ENERGY_MINIMIZATION_TOLERANCE
    from openff.toolkit import Molecule, Quantity

    LOGGER.debug(f"Setting up constrained minimization for {input.dict()=}")

    # TODO: Pass this through
    restrain_k = 1.0

    # TODO: GAFF/Espaloma/local file/plugin force fields

    LOGGER.debug(f"Loading force field {input.force_field=}")
    force_field = _lazy_load_force_field(input.force_field)

    # if this force field is constrained, this will be the H-* constraint ...
    try:
        assert "tip3p" not in force_field["Constraints"].parameters[0].id
    except (KeyError, AssertionError):
        pass

    LOGGER.debug(f"Creating molecule, with conformer, from {input.mapped_smiles=}")
    molecule = Molecule.from_mapped_smiles(input.mapped_smiles, allow_undefined_stereo=True)
    molecule.add_conformer(Quantity(input.coordinates, "angstrom"))

    LOGGER.debug("Creating interchange object")
    interchange = force_field.create_interchange(molecule.to_topology())

    restraint_force = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint_force.addGlobalParameter(
        "k",
        restrain_k * openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**2,
    )
    for parameter in ("x0", "y0", "z0"):
        restraint_force.addPerParticleParameter(parameter)

    atom_indices = list(range(len(molecule.atoms)))
    atom_indices = sorted(set(atom_indices))  # - set([index - 0 for index in input.dihedral_indices]))

    # switch to nm now... just in case
    positions = interchange.positions.to("nanometer")

    LOGGER.debug(f"Adding restraint to particles not in {input.dihedral_indices=}")
    for atom_index in range(molecule.n_atoms):
        if atom_index in input.dihedral_indices:
            continue

        particle_index = restraint_force.addParticle(atom_index)
        restraint_force.setParticleParameters(
            particle_index,
            atom_index,
            [x.to_openmm() for x in positions[atom_index]],
        )

    LOGGER.debug("Creating openmm.app.Simulation object")
    simulation = interchange.to_openmm_simulation(
        openmm.LangevinMiddleIntegrator(
            293.15 * openmm.unit.kelvin,
            1.0 / openmm.unit.picosecond,
            2.0 * openmm.unit.femtosecond,
        ),
        combine_nonbonded_forces=True,
        additional_forces=[restraint_force],
    )

    simulation.context.computeVirtualSites()

    for index in input.dihedral_indices:
        simulation.system.setParticleMass(index, 0.0)

    LOGGER.debug("Trying to minimize energy")
    try:
        simulation.minimizeEnergy(
            tolerance=_DEFAULT_ENERGY_MINIMIZATION_TOLERANCE.to_openmm(),
            maxIterations=10_000,
        )
    except Exception as e:
        LOGGER.error(
            {
                index: simulation.system.getParticleMass(index)._value
                for index in range(simulation.system.getNumParticles())
            },
        )
        LOGGER.error(input.dihedral_indices, input.mapped_smiles)

        raise ConstrainedMinimizationError("Minimization failed, see logger") from e

    LOGGER.debug("Returning result")
    return ConstrainedMinimizationResult(
        torsion_id=input.torsion_id,
        mapped_smiles=input.mapped_smiles,
        dihedral_indices=input.dihedral_indices,
        force_field=input.force_field,
        coordinates=simulation.context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(openmm.unit.angstrom)[: interchange.positions.shape[0], :],
        energy=simulation.context.getState(
            getEnergy=True,
        )
        .getPotentialEnergy()
        .value_in_unit(
            openmm.unit.kilocalorie_per_mole,
        ),
        grid_id=input.grid_id,
    )
