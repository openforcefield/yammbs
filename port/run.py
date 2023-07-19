import openmm
import openmm.app
import openmm.unit
import qcelemental
from ibstore._store import MoleculeStore, smiles_to_inchi_key
from ibstore.models import MMConformerRecord, MoleculeRecord, QMConformerRecord
from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit import ForceField, Molecule
from openff.toolkit.utils.openeye_wrapper import OpenEyeToolkitWrapper
from openff.toolkit.utils.toolkit_registry import _toolkit_registry_manager
from openff.units import unit
from openff.units.openmm import ensure_quantity

hartree2kcalmol = qcelemental.constants.hartree2kcalmol
FORCE_FIELD = "openff_unconstrained-2.1.0.offxml"

force_field = ForceField(FORCE_FIELD)


def _minimize(molecule: Molecule) -> tuple[openmm.unit.Quantity, float]:
    """
    Minimize molecule with specified system and return the positions of the optimized
    molecule.
    """
    integrator = openmm.VerletIntegrator(0.1 * openmm.unit.femtoseconds)

    system = force_field.create_interchange(molecule.to_topology()).to_openmm()

    platform = openmm.Platform.getPlatformByName("Reference")

    openmm_context = openmm.Context(system, integrator, platform)
    openmm_context.setPositions(molecule.conformers[0].to(unit.nanometer).to_openmm())
    print(
        f"made objects, starting minimization on a molecule with {molecule.n_atoms} atoms"
    )
    openmm.LocalEnergyMinimizer.minimize(openmm_context, 5.0e-9, 1500)
    print("minimization complete, getting results")

    conformer = openmm_context.getState(getPositions=True).getPositions()
    energy = openmm_context.getState(getEnergy=True).getPotentialEnergy()
    print("got results, returning")
    return ensure_quantity(conformer, "openff"), energy.value_in_unit(
        openmm.unit.kilocalories_per_mole
    )


collection = OptimizationResultCollection.parse_file("01-processed-qm-ch.json")

store = MoleculeStore("test.sqlite")

for record_and_molecule in collection.to_records():
    record = record_and_molecule[0]
    molecule = record_and_molecule[1]

    with _toolkit_registry_manager(OpenEyeToolkitWrapper()):
        minimized_conformer, minimized_energy = _minimize(molecule)

    record = MoleculeRecord(
        qcarchive_id=record.id,
        qcarchive_energy=record.get_final_energy() * hartree2kcalmol,
        mapped_smiles=molecule.to_smiles(mapped=True, isomeric=True),
        inchi_key=smiles_to_inchi_key(molecule.to_smiles(mapped=True, isomeric=True)),
        conformer=QMConformerRecord(
            coordinates=molecule.conformers[0].m_as(unit.angstrom),
            energy=record.get_final_energy() * hartree2kcalmol,
        ),
        minimized_conformer=MMConformerRecord(
            coordinates=minimized_conformer.m_as(unit.angstrom),
            energy=minimized_energy,
        ),
        minimized_energy=minimized_energy,
    )

    store.store(record)

assert len(store.get_smiles()) == 40
