import pickle
import sqlite3

import openmm
import openmm.app
import openmm.unit
from openff.toolkit import Molecule
from openff.toolkit.utils import GLOBAL_TOOLKIT_REGISTRY, OpenEyeToolkitWrapper
from openff.units import unit
from steps._forcefields import _get_openmm_system

N_PROCESSES = 16


def _run_openmm(
    molecule: Molecule, system: openmm.System
) -> tuple[openmm.unit.Quantity, float]:
    """
    Minimize molecule with specified system and return the positions of the optimized
    molecule.
    """

    # Make sure we consistently only use OE in this script
    for toolkit in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits:
        if isinstance(toolkit, OpenEyeToolkitWrapper):
            continue
        GLOBAL_TOOLKIT_REGISTRY.deregister_toolkit(toolkit)

    integrator = openmm.VerletIntegrator(0.1 * openmm.unit.femtoseconds)

    platform = openmm.Platform.getPlatformByName("Reference")

    openmm_context = openmm.Context(system, integrator, platform)
    openmm_context.setPositions(molecule.conformers[0].to(unit.nanometer).to_openmm())
    print(
        f"made objects, starting minimization on a molecule with {molecule.n_atoms} atoms"
    )
    openmm.LocalEnergyMinimizer.minimize(openmm_context, 5.0e-9, 1500)
    print("minimization complete, getting results")

    conformer = openmm_context.getState(getPositions=True).getPositions(asNumpy=True)
    energy = openmm_context.getState(getEnergy=True).getPotentialEnergy()
    print("got results, returning")
    return conformer, energy.value_in_unit(openmm.unit.kilocalories_per_mole)


def minimize_from_database(
    database: str,
    force_field_paths: list[str],
):
    connection = sqlite3.connect(database)

    cursor = connection.cursor()

    cursor.execute("SELECT * from molecules;")

    for (
        id,
        qcarchive_id,
        qcarchive_energy,
        inchi_key,
        mapped_smiles,
    ) in cursor.fetchall():
        cursor.execute(f"SELECT * from conformers where parent_id = {id};")

        for id, parent_id, coordinates in cursor.fetchall():
            molecule = Molecule.from_mapped_smiles(mapped_smiles)

            conformer = unit.Quantity(
                pickle.loads(coordinates),
                unit.angstrom,
            )

            molecule.add_conformer(conformer)

            for force_field_path in force_field_paths:
                system = _get_openmm_system(molecule, force_field_path)

                mm_conformer, mm_energy = _run_openmm(molecule, system)

                print(f"{force_field_path} minimized {mapped_smiles} to {mm_energy}")

                cursor.execute(
                    "INSERT INTO conformers (mm_energy, mm_conformer) VALUES "
                    f"({mm_energy}, {pickle.dumps(mm_conformer)});"
                )

                connection.commit()
