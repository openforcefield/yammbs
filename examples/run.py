from openff.toolkit import ForceField as OpenFFForceField
from openff.toolkit import Molecule
from openmm import XmlSerializer
from openmm.app import ForceField as OpenMMForceField
from openmmforcefields.generators import GAFFTemplateGenerator
from packaging.version import Version

from ib.forcefields import GAFFForceFieldProvider, SMIRNOFFForceFieldProvider
from ib.molecules import OpenFFMolecule, OpenFFSingleMoleculeTopologyProvider
from ib.systems import GAFFSystemProvider, SMIRNOFFSystemProvider

GAFF_VERSION = Version("2.11")
OPENFF_VERSION = Version("2.0.0")

molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)
sage = OpenFFForceField(f"openff_unconstrained-{OPENFF_VERSION}.offxml")

topology_object = OpenFFSingleMoleculeTopologyProvider(
    identifier="openff",
    components=[OpenFFMolecule(molecule=molecule)],
)

gaff_generator = GAFFTemplateGenerator(
    molecules=molecule,
    forcefield=f"gaff-{str(GAFF_VERSION)}",
)

gaff_forcefield = OpenMMForceField()
gaff_forcefield.registerTemplateGenerator(gaff_generator.generator)

smirnoff_force_field = SMIRNOFFForceFieldProvider(
    identifier="openff-2.0.0",
    force_field=sage,
)

gaff_force_field = GAFFForceFieldProvider(
    identifier=f"gaff-{str(GAFF_VERSION)}",
    force_field=gaff_forcefield,
)

sage_system = SMIRNOFFSystemProvider(
    topology=topology_object,
    force_field=smirnoff_force_field,
    positions=molecule.conformers[0],
).to_system()

gaff_system = GAFFSystemProvider(
    topology=topology_object,
    force_field=gaff_force_field,
    positions=molecule.conformers[0],
).to_system()

with open("sage.xml", "w") as sage_file:
    sage_file.write(XmlSerializer.serialize(sage_system))

with open("gaff.xml", "w") as gaff_file:
    gaff_file.write(XmlSerializer.serialize(gaff_system))
