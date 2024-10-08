import functools
import re
from multiprocessing import Pool
from typing import Iterator, Union

import numpy
import openmm
import openmm.app
import openmm.unit
from openff.toolkit import ForceField, Molecule
from openff.toolkit.typing.engines.smirnoff import get_available_force_fields
from pydantic.v1 import Field
from tqdm import tqdm

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel

_AVAILABLE_FORCE_FIELDS = get_available_force_fields()


def _shorthand_to_full_force_field_name(
    shorthand: str,
    make_unconstrained: bool = True,
) -> str:
    """Make i.e. `openff-2.1.0` into `openff_unconstrained-2.1.0.offxml`"""
    if make_unconstrained:
        # Split on '-' immediately followed by a number;
        # cannot split on '-' because of i.e. 'de-force-1.0.0'
        prefix, version = re.split(r"-[0-9]", shorthand, maxsplit=1)
        return f"{prefix}_unconstrained-{version}.offxml"
    else:
        return shorthand + ".offxml"


@functools.lru_cache(maxsize=1)
def _lazy_load_force_field(force_field_name: str) -> ForceField:
    """
    Attempt to load a force field from a shorthand string or a file path.

    Caching is used to speed up loading; a single force field takes O(100 ms) to
    load, but the cache takes O(10 ns) to access. The cache key is simply the
    argument passed to this function; a hash collision should only occur when
    two identical strings are expected to return different force fields, which
    seems like an assumption that the toolkit has always made anyway.
    """
    if not force_field_name.endswith(".offxml"):
        force_field_name = _shorthand_to_full_force_field_name(
            force_field_name,
            make_unconstrained=False,
        )

    return ForceField(
        force_field_name,
        allow_cosmetic_attributes=True,
        load_plugins=True,
    )


def _minimize_blob(
    input: dict[str, dict[str, Union[str, numpy.ndarray]]],
    force_field: str,
    n_processes: int = 2,
    chunksize=32,
) -> Iterator["MinimizationResult"]:
    inputs = list()

    inputs = [
        MinimizationInput(
            inchi_key=inchi_key,
            qcarchive_id=row["qcarchive_id"],
            force_field=force_field,
            mapped_smiles=row["mapped_smiles"],
            coordinates=row["coordinates"],
        )
        for inchi_key in input
        for row in input[inchi_key]
    ]

    with Pool(processes=n_processes) as pool:
        for val in tqdm(
            pool.imap(
                _run_openmm,
                inputs,
                chunksize=chunksize,
            ),
            desc=f"Building and minimizing systems with {force_field}",
            total=len(inputs),
        ):
            if val is not None:
                yield val


class MinimizationInput(ImmutableModel):
    inchi_key: str = Field(..., description="The InChI key of the molecule")
    qcarchive_id: str = Field(
        ...,
        description="The ID of the molecule in the QCArchive",
    )
    force_field: str = Field(
        ...,
        description="And identifier of the force field to use for the minimization",
    )
    mapped_smiles: str = Field(
        ...,
        description="The mapped SMILES string for the molecule, stored to track atom maps",
    )
    coordinates: Array = Field(
        ...,
        description="The coordinates [Angstrom] of this conformer with shape=(n_atoms, 3).",
    )


class MinimizationResult(ImmutableModel):
    # This could probably just subclass and add on the energy field?
    inchi_key: str = Field(..., description="The InChI key of the molecule")
    qcarchive_id: str
    force_field: str
    mapped_smiles: str
    coordinates: Array
    energy: float = Field(..., description="Minimized energy in kcal/mol")


def _run_openmm(
    input: MinimizationInput,
) -> MinimizationResult | None:
    from openff.interchange.exceptions import UnassignedValenceError

    inchi_key: str = input.inchi_key
    qcarchive_id: str = input.qcarchive_id
    positions: numpy.ndarray = input.coordinates

    molecule = Molecule.from_mapped_smiles(
        input.mapped_smiles,
        allow_undefined_stereo=True,
    )

    if input.force_field.startswith("gaff"):
        from yammbs._forcefields import _gaff

        system = _gaff(
            molecule=molecule,
            force_field_name=input.force_field,
        )

    elif input.force_field.startswith("espaloma"):
        from yammbs._forcefields import _espaloma

        system = _espaloma(
            molecule=molecule,
            force_field_name=input.force_field,
        )

    else:
        try:
            force_field = _lazy_load_force_field(input.force_field)
        except KeyError:
            # Attempt to load from local path
            try:
                force_field = ForceField(
                    input.force_field,
                    allow_cosmetic_attributes=True,
                    load_plugins=True,
                )
            except Exception as error:
                # The toolkit does a poor job of distinguishing between a string
                # argument being a file that does not exist and a file that it should
                # try to parse (polymorphic input), so just have to clobber whatever
                raise NotImplementedError(
                    f"Could not find or parse force field {input.force_field}",
                ) from error

        try:
            system = force_field.create_interchange(molecule.to_topology()).to_openmm(
                combine_nonbonded_forces=False,
            )
        except UnassignedValenceError:
            return None

    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.1 * openmm.unit.femtoseconds),
        openmm.Platform.getPlatformByName("Reference"),
    )

    context.setPositions(
        (positions * openmm.unit.angstrom).in_units_of(openmm.unit.nanometer),
    )
    openmm.LocalEnergyMinimizer.minimize(
        context=context,
        tolerance=10,
        maxIterations=0,
    )

    return MinimizationResult(
        inchi_key=inchi_key,
        qcarchive_id=qcarchive_id,
        force_field=input.force_field,
        mapped_smiles=input.mapped_smiles,
        coordinates=context.getState(getPositions=True).getPositions().value_in_unit(openmm.unit.angstrom),
        energy=context.getState(
            getEnergy=True,
        )
        .getPotentialEnergy()
        .value_in_unit(
            openmm.unit.kilocalorie_per_mole,
        ),
    )
