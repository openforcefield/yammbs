import functools
import logging
import re

import openmm
from openff.toolkit import ForceField, Molecule

logger = logging.getLogger(__name__)


def _shorthand_to_full_force_field_name(
    shorthand: str,
    make_unconstrained: bool = True,
) -> str:
    """Make i.e. `openff-2.1.0` into `openff_unconstrained-2.1.0.offxml`."""
    if make_unconstrained:
        # Split on '-' immediately followed by a number;
        # cannot split on '-' because of i.e. 'de-force-1.0.0'
        prefix, version, _ = re.split(r"-([\d.]+)", shorthand, maxsplit=1)

        return f"{prefix}_unconstrained-{version}.offxml"
    else:
        return shorthand + ".offxml"


@functools.lru_cache(maxsize=1)
def _lazy_load_force_field(force_field_name: str) -> ForceField:
    """Attempt to load a force field from a shorthand string or a file path.

    Caching is used to speed up loading; a single force field takes O(100 ms) to
    load, but the cache takes O(10 ns) to access. The cache key is simply the
    argument passed to this function; a hash collision should only occur when
    two identical strings are expected to return different force fields, which
    seems like an assumption that the toolkit has always made anyway.
    """
    if not force_field_name.endswith(".offxml"):
        force_field_name = _shorthand_to_full_force_field_name(
            force_field_name,
            make_unconstrained=True,
        )

    return ForceField(
        force_field_name,
        allow_cosmetic_attributes=True,
        load_plugins=True,
    )


def _smirnoff(molecule: Molecule, force_field_path: str) -> openmm.System:
    from openff.toolkit import ForceField

    try:
        force_field = _lazy_load_force_field(force_field_path)
    except KeyError:
        # Attempt to load from local path
        try:
            force_field = ForceField(
                force_field_path,
                allow_cosmetic_attributes=True,
                load_plugins=True,
            )
        except Exception as error:
            # The toolkit does a poor job of distinguishing between a string
            # argument being a file that does not exist and a file that it should
            # try to parse (polymorphic input), so just have to clobber whatever
            raise NotImplementedError(
                f"Could not find or parse force field {force_field_path}",
            ) from error

    if "Constraints" in force_field.registered_parameter_handlers:
        logger.info("Deregistering Constraints handler from SMIRNOFF force field")
        force_field.deregister_parameter_handler("Constraints")

    return force_field.create_openmm_system(molecule.to_topology())


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
    """Generate an OpenMM System for a molecule and force field name.

    The force field name should be of the form espaloma-force-field-name, such as
    espaloma-openff_unconstrained-2.1.0. Everything after the first dash is passed as
    the forcefield argument to espaloma.graphs.deploy.openmm_system_from_graph, where
    it will be appended with .offxml. Raises a ValueError if there is no dash in
    force_field_name.
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


NON_SMIRNOFF_SYSTEM_BUILDERS = {
    "gaff": _gaff,
    "espaloma": _espaloma,
}


def build_omm_system(force_field: str, molecule: Molecule) -> openmm.System:
    """Get an OpenMM System for a given force field and molecule."""
    for prefix, builder in NON_SMIRNOFF_SYSTEM_BUILDERS.items():
        if force_field.startswith(prefix):
            return builder(molecule, force_field)

    return _smirnoff(molecule, force_field)
