import openmm
from openff.toolkit import Molecule


def _get_force_field_type(
    force_field_path: str,
) -> str:
    if force_field_path.endswith(".offxml"):
        return "SMIRNOFF"
    elif force_field_path.endswith(".xml"):
        return "OPENMM"
    else:
        raise NotImplementedError(f"Force field type {force_field_path} not supported.")


def _get_openmm_system(
    molecule: Molecule,
    force_field_path: str,
) -> openmm.System:
    force_field_type = _get_force_field_type(force_field_path)

    if force_field_type == "SMIRNOFF":
        return _smirnoff(molecule, force_field_path)

    elif force_field_type == "OPENMM":
        return _gaff(molecule, force_field_path)

    else:
        raise NotImplementedError(f"force field type {force_field_type} not supported.")


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
    import espaloma as esp

    if not force_field_name.startswith("espaloma"):
        raise NotImplementedError(f"Force field {force_field_name} not implemented.")

    ff = force_field_name.split("-", 1)[1:2]

    if ff == []:
        raise ValueError("espaloma force field must have an OpenFF force field too")
    else:
        ff = ff[0]

    mol_graph = esp.Graph(molecule)
    model = esp.get_model("latest")
    model(mol_graph.heterograph)

    return esp.graphs.deploy.openmm_system_from_graph(mol_graph, forcefield=ff)
