from collections import defaultdict
from multiprocessing import Pool
from typing import Union

import numpy
import openmm
import openmm.app
import openmm.unit
from openff.toolkit import ForceField, Molecule
from openff.toolkit.utils import OpenEyeToolkitWrapper
from openff.toolkit.utils.toolkit_registry import _toolkit_registry_manager
from pydantic import Field
from tqdm import tqdm

from ibstore._base.array import Array
from ibstore._base.base import ImmutableModel

N_PROCESSES = 10

FORCE_FIELD = ForceField("openff_unconstrained-2.0.0.offxml")


def _minimize_blob(input: dict[str, dict[str, Union[str, numpy.ndarray]]]):
    returned = defaultdict(list)
    tmp = list()

    with _toolkit_registry_manager(OpenEyeToolkitWrapper()):
        for inchi_key in tqdm(input, desc="Creating systems"):
            molecule = Molecule.from_inchi(inchi_key)

            system = FORCE_FIELD.create_openmm_system(molecule.to_topology())

            for row in input[inchi_key]:
                tmp.append(
                    tuple(
                        (
                            inchi_key,
                            row["qcarchive_id"],
                            row["coordinates"],
                            system,
                        )
                    )
                )

        with Pool(processes=N_PROCESSES) as pool:
            for inchi_key, result in pool.starmap(
                _run_openmm, tqdm(tmp, desc="Minimizing systems")
            ):
                returned[inchi_key].append(result)

    return returned


class MinimizedResult(ImmutableModel):
    qcarchive_id: str
    coordinates: Array
    energy: float = Field(..., description="Minimized energy in kcal/mol")


def _run_openmm(
    inchi_key: str, qcarchive_id: str, positions: numpy.ndarray, system: openmm.System
):
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.1 * openmm.unit.femtoseconds),
        openmm.Platform.getPlatformByName("Reference"),
    )

    context.setPositions(positions * openmm.unit.angstrom)
    openmm.LocalEnergyMinimizer.minimize(context, 5.0e-9, 1500)

    return inchi_key, MinimizedResult(
        qcarchive_id=qcarchive_id,
        coordinates=context.getState(getPositions=True).getPositions(asNumpy=True),
        energy=context.getState(
            getEnergy=True,
        )
        .getPotentialEnergy()
        .value_in_unit(
            openmm.unit.kilocalorie_per_mole,
        ),
    )
