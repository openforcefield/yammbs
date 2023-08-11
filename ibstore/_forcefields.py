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
