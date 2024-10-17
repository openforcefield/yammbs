import openmm
from openff.toolkit import Molecule


def _smirnoff(molecule: Molecule, force_field_path: str) -> openmm.System:
    from openff.toolkit import ForceField

    smirnoff_force_field = ForceField(
        force_field_path,
        load_plugins=True,
        allow_cosmetic_attributes=True,
    )

    if "Constraints" in smirnoff_force_field.registered_parameter_handlers:
        smirnoff_force_field.deregister_parameter_handler("Constraints")

    return smirnoff_force_field.create_openmm_system(molecule.to_topology())


def _gaff(molecule: Molecule, force_field_name: str) -> openmm.System:
    import openmm.app
    import openmm.unit
    from openmmforcefields.generators import GAFFTemplateGenerator

    if not force_field_name.startswith("gaff"):
        raise NotImplementedError(f"Force field {force_field_name} not implemented.")

    force_field = openmm.app.ForceField()

    generator = GAFFTemplateGenerator(molecules=molecule, forcefield=force_field_name)

    force_field.registerTemplateGenerator(generator.generator)

    return force_field.createSystem(
        molecule.to_topology().to_openmm(),
        nonbondedCutoff=0.9 * openmm.unit.nanometer,
        constraints=None,
    )


def _espaloma(molecule: Molecule, force_field_name: str) -> openmm.System:
    """Generate an OpenMM System for a molecule and force field name. The force
    field name should be of the form espaloma-force-field-name, such as
    espaloma-openff_unconstrained-2.1.0. Everything after the first dash is
    passed as the forcefield argument to
    espaloma.graphs.deploy.openmm_system_from_graph, where it will be appended
    with .offxml. Raises a ValueError if there is no dash in force_field_name.
    """
    import espaloma

    if not force_field_name.startswith("espaloma"):
        raise NotImplementedError(f"Force field {force_field_name} not implemented.")

    ff = force_field_name.split("-", 1)[1:2]

    if len(ff) == 0:
        raise ValueError("espaloma force field must have an OpenFF force field too")

    mol_graph = espaloma.Graph(molecule)
    model = espaloma.get_model("latest")
    model(mol_graph.heterograph)

    return espaloma.graphs.deploy.openmm_system_from_graph(mol_graph, forcefield=ff[0])
