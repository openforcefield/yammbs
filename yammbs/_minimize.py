import logging
from collections.abc import Callable, Iterator
from multiprocessing import Pool
from typing import Literal

import numpy
import openmm
import openmm.unit
from openff.interchange.exceptions import UnassignedValenceError
from openff.interchange.operations.minimize import (
    _DEFAULT_ENERGY_MINIMIZATION_TOLERANCE,
)
from openff.toolkit import Molecule
from openff.toolkit.typing.engines.smirnoff import get_available_force_fields
from pydantic import Field
from tqdm import tqdm

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel
from yammbs._forcefields import build_omm_system

_AVAILABLE_FORCE_FIELDS = get_available_force_fields()

logger = logging.getLogger(__name__)
logging.basicConfig()

_MinimizationFn = Callable[
    [Molecule, openmm.System, numpy.ndarray],
    tuple[numpy.ndarray, float],
]


def _minimize_blob(
    input: dict[str, list],
    force_field: str,
    method: Literal["openmm"] = "openmm",
    n_processes: int = 2,
    chunksize=32,
) -> Iterator["MinimizationResult"]:
    inputs = list()

    inputs = [
        MinimizationInput(
            inchi_key=inchi_key,
            qcarchive_id=row["qcarchive_id"],
            force_field=force_field,
            mapped_smiles=row["mapped_smiles"],
            coordinates=row["coordinates"],
            method=method,
        )
        for inchi_key in input
        for row in input[inchi_key]
    ]

    with Pool(processes=n_processes) as pool:
        for val in tqdm(
            pool.imap(
                _run_minimization,
                inputs,
                chunksize=chunksize,
            ),
            desc=f"Building and minimizing systems with {force_field}",
            total=len(inputs),
        ):
            if val is not None:
                yield val


class MinimizationInput(ImmutableModel):
    inchi_key: str = Field(..., description="The InChI key of the molecule")
    qcarchive_id: int = Field(
        ...,
        description="The ID of the molecule in the QCArchive",
    )
    force_field: str = Field(
        ...,
        description="And identifier of the force field to use for the minimization",
    )
    mapped_smiles: str = Field(
        ...,
        description="The mapped SMILES string for the molecule, stored to track atom maps",
    )
    coordinates: Array = Field(
        ...,
        description="The coordinates [Angstrom] of this conformer with shape=(n_atoms, 3).",
    )

    method: Literal["openmm"] = Field(
        "openmm",
        description="The minimization method to use",
    )

    @property
    def minimization_function(self) -> _MinimizationFn:
        return _minimize_openmm


class MinimizationResult(MinimizationInput):
    energy: float = Field(..., description="Minimized energy in kcal/mol")


def _minimize_openmm(
    mol: Molecule,
    system: openmm.System,
    positions: numpy.ndarray,
) -> tuple[numpy.ndarray, float]:
    """Minimize a system using OpenMM's LocalEnergyMinimizer."""
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.1 * openmm.unit.femtoseconds),
        openmm.Platform.getPlatformByName("Reference"),
    )

    context.setPositions(
        (positions * openmm.unit.angstrom).in_units_of(openmm.unit.nanometer),
    )

    # TODO: Remove this?
    context.computeVirtualSites()

    # Log the initial energy
    initial_energy = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(
            openmm.unit.kilocalorie_per_mole,
        )
    )
    logger.info(f"Initial energy: {initial_energy} kcal/mol")

    # Log initial positions
    initial_positions = (
        context.getState(getPositions=True)
        .getPositions(
            asNumpy=True,
        )
        .value_in_unit(openmm.unit.angstrom)
    )
    logger.debug(f"Initial positions (Angstrom): {initial_positions}")

    openmm.LocalEnergyMinimizer.minimize(
        context=context,
        tolerance=_DEFAULT_ENERGY_MINIMIZATION_TOLERANCE.to_openmm(),
        maxIterations=10_000,
    )

    final_positions = (
        context.getState(getPositions=True)
        .getPositions(
            asNumpy=True,
        )
        .value_in_unit(openmm.unit.angstrom)
    )

    final_energy = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(
            openmm.unit.kilocalorie_per_mole,
        )
    )

    return final_positions, final_energy


def _run_minimization(
    input: MinimizationInput,
) -> MinimizationResult | None:
    inchi_key: str = input.inchi_key
    qcarchive_id: int = input.qcarchive_id
    positions: numpy.ndarray = input.coordinates

    molecule = Molecule.from_mapped_smiles(
        input.mapped_smiles,
        allow_undefined_stereo=True,
    )

    try:
        system = build_omm_system(
            force_field=input.force_field,
            molecule=molecule,
        )
    except UnassignedValenceError:
        logger.warning(f"Skipping record {qcarchive_id} with unassigned valence terms")
        return None
    except ValueError as e:  # charging error
        logger.warning(f"Skipping record {qcarchive_id} with a value error (probably a charge failure): {e}")
        return None

    final_positions, final_energy = input.minimization_function(
        molecule,
        system,
        positions,
    )

    return MinimizationResult(
        inchi_key=inchi_key,
        qcarchive_id=qcarchive_id,
        force_field=input.force_field,
        mapped_smiles=input.mapped_smiles,
        coordinates=final_positions,
        energy=final_energy,
        method=input.method,
    )
