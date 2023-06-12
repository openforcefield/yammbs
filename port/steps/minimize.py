import logging
from tqdm import tqdm
import os
from multiprocessing import get_context
from pathlib import Path
import click
import numpy
from openeye import oechem
from openff.toolkit import Molecule, ForceField
from openff.toolkit.utils import GLOBAL_TOOLKIT_REGISTRY, OpenEyeToolkitWrapper
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.units import unit
from openff.units.openmm import ensure_quantity
import openmm
import openmm.app
import openmm.unit


N_PROCESSES = 16

def _run_openmm(molecule: Molecule, system: openmm.System):
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
    print(f"made objects, starting minimization on a molecule with {molecule.n_atoms} atoms")
    openmm.LocalEnergyMinimizer.minimize(openmm_context, 5.0e-9, 1500)
    print("minimization complete, getting results")

    conformer = openmm_context.getState(getPositions=True).getPositions(asNumpy=True)
    energy = openmm_context.getState(getEnergy=True).getPotentialEnergy()
    print("got results, returning")
    return conformer, energy.value_in_unit(openmm.unit.kilocalories_per_mole)


def minimize(
    input_path,
    input_name,
    force_field_path,
    force_field_type: str,
    output_directory,
    n_chunks: int,
):
    os.makedirs(output_directory, exist_ok=True)

    with get_context("spawn").Pool(processes=N_PROCESSES) as pool:
        TASKS = [
                    (
                        os.path.join(input_path, input_name) + f"-{index + 1}.sdf",
                        force_field_path,
                        force_field_type,
                        os.path.join(
                            output_directory, f"{Path(force_field_path).stem}-{index + 1}.sdf"
                        ),
                    ) for index in range(n_chunks)]

        tqdm(
            iterable=pool.starmap(
                _minimize,
                TASKS,
            ),
            total=len(TASKS),
        )


def _minimize(input_path, force_field_path, force_field_type: str, output_path):
    print(f"trying to minimize {input_path} with {force_field_path} ... ")
    input_stream = oechem.oemolistream(input_path)
    output_stream = oechem.oemolostream(output_path)

    failed = False

    try:
        for oe_molecule in input_stream.GetOEGraphMols():
            oe_molecule = oechem.OEGraphMol(oe_molecule)
            oechem.OE3DToInternalStereo(oe_molecule)

            oe_data = {
                pair.GetTag(): pair.GetValue()
                for pair in oechem.OEGetSDDataPairs(oe_molecule)
            }

            off_molecule = Molecule.from_openeye(
                oe_molecule, allow_undefined_stereo=True
            )

            off_molecule._conformers = [
                numpy.array(
                    [oe_molecule.GetCoords()[i] for i in range(off_molecule.n_atoms)]
                )
                * unit.angstrom
            ]

            if force_field_type.lower() == "smirnoff":
                smirnoff_force_field = ForceField(
                    force_field_path, load_plugins=True, allow_cosmetic_attributes=True
                )

                if "Constraints" in smirnoff_force_field.registered_parameter_handlers:
                    smirnoff_force_field.deregister_parameter_handler("Constraints")

                omm_system = smirnoff_force_field.create_openmm_system(
                    off_molecule.to_topology()
                )

            elif force_field_type.lower() == "gaff":
                force_field = openmm.app.ForceField()

                generator = GAFFTemplateGenerator(
                    molecules=off_molecule, forcefield=force_field_path
                )

                force_field.registerTemplateGenerator(generator.generator)

                omm_system = force_field.createSystem(
                    off_molecule.to_topology().to_openmm(),
                    nonbondedCutoff=0.9 * openmm.unit.nanometer,
                    constraints=None,
                )

            else:
                raise NotImplementedError()

            new_conformer, energy = _run_openmm(off_molecule, omm_system)

            off_molecule._conformers = [ensure_quantity(new_conformer, "openff")]
            oe_molecule = off_molecule.to_openeye()

            oechem.OESetSDData(oe_molecule, "Energy FFXML", str(energy))

            for key, value in oe_data.items():
                oechem.OESetSDData(oe_molecule, key, value)

            oechem.OEWriteMolecule(output_stream, oe_molecule)

    except BaseException:
        logging.exception(f"failed to minimize {input_path} with {force_field_path}")
        failed = True

    input_stream.close()
    output_stream.close()

    if failed and os.path.isfile(output_path):
        os.unlink(output_path)
