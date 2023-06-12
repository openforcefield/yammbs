"""Energy evaluation functions."""
import openmm
import openmm.app
from openff.toolkit import ForceField, Molecule
from openff.units import Quantity
from openff.units.openmm import ensure_quantity
from openmm import unit as openmm_unit


def _get_energy(molecule: Molecule, force_field: ForceField) -> Quantity:
    from openff.interchange.drivers import get_openmm_energies

    interchange = force_field.create_interchange(molecule.to_topology())

    return sum(
        get_openmm_energies(
            interchange, combine_nonbonded_forces=True
        ).energies.values()
    )


def _minimize(system_provider) -> Quantity:
    system = system_provider.to_system()
    integrator = openmm.VerletIntegrator(1.0 * openmm_unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("CPU")

    simulation = openmm.app.Simulation(
        topology=system_provider.to_openmm_topology(),
        system=system,
        integrator=integrator,
        platform=platform,
    )

    simulation.context.setPositions(
        ensure_quantity(system_provider.positions, "openmm")
    )

    simulation.minimizeEnergy()

    state = simulation.context.getState(getPositions=True)

    return ensure_quantity(state.getPositions(), "openff")
