from multiprocessing import Pool
from typing import Union

import numpy
import openmm
import openmm.app
import openmm.unit
from openff.toolkit import ForceField, Molecule
from pydantic import Field
from tqdm import tqdm

from ibstore._base.array import Array
from ibstore._base.base import ImmutableModel

# TODO: This causes all of these to be loaded at import time, which is not ideal
FORCE_FIELDS: dict[str, ForceField] = {
    "openff-1.0.0": ForceField("openff_unconstrained-1.0.0.offxml"),
    "openff-1.1.0": ForceField("openff_unconstrained-1.1.0.offxml"),
    "openff-1.2.0": ForceField("openff_unconstrained-1.2.0.offxml"),
    "openff-1.3.0": ForceField("openff_unconstrained-1.3.0.offxml"),
    "openff-2.0.0": ForceField("openff_unconstrained-2.0.0.offxml"),
    "openff-2.1.0": ForceField("openff_unconstrained-2.1.0.offxml"),
    "de-force-1.0.1": ForceField(
        "de-force_unconstrained-1.0.1.offxml",
        load_plugins=True,
    ),
}


def _minimize_blob(
    input: dict[str, dict[str, Union[str, numpy.ndarray]]],
    force_field: str,
    prune_isomorphs: bool,
    n_processes: int = 2,
    chunksize=32,
) -> dict[str, list["MinimizationResult"]]:
    inputs = list()

    for inchi_key in input:
        for row in input[inchi_key]:
            if prune_isomorphs:
                # This behavior is always useless and probably bad as there is
                # no reason to use InCHI when mapped SMILES is known. See #7
                are_isomorphic, _ = Molecule.are_isomorphic(
                    Molecule.from_inchi(inchi_key, allow_undefined_stereo=True),
                    Molecule.from_mapped_smiles(
                        row["mapped_smiles"],
                        allow_undefined_stereo=True,
                    ),
                )

                if not are_isomorphic:
                    continue

            inputs.append(
                MinimizationInput(
                    inchi_key=inchi_key,
                    qcarchive_id=row["qcarchive_id"],
                    force_field=force_field,
                    mapped_smiles=row["mapped_smiles"],
                    coordinates=row["coordinates"],
                ),
            )

    with Pool(processes=n_processes) as pool:
        yield from tqdm(
            pool.imap(
                _run_openmm,
                inputs,
                chunksize=chunksize,
            ),
            desc=f"Building and minimizing systems with {force_field}",
            total=len(inputs),
        )


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
    inchi_key: str = Field(..., description="The InChI key of the molecule")
    qcarchive_id: str
    force_field: str
    mapped_smiles: str
    coordinates: Array
    energy: float = Field(..., description="Minimized energy in kcal/mol")


def _run_openmm(
    input: MinimizationInput,
) -> MinimizationResult:
    inchi_key: str = input.inchi_key
    qcarchive_id: str = input.qcarchive_id
    positions: numpy.ndarray = input.coordinates

    molecule = Molecule.from_mapped_smiles(
        input.mapped_smiles,
        allow_undefined_stereo=True,
    )

    if input.force_field.startswith("gaff"):
        from ibstore._forcefields import _gaff

        system = _gaff(
            molecule=molecule,
            force_field_name=input.force_field,
        )

    elif input.force_field.startswith("espaloma"):
        from ibstore._forcefields import _espaloma

        system = _espaloma(
            molecule=molecule,
            force_field_name=input.force_field,
        )

    else:
        try:
            force_field = FORCE_FIELDS[input.force_field]
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

        system = force_field.create_interchange(molecule.to_topology()).to_openmm(
            combine_nonbonded_forces=False,
        )

    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.1 * openmm.unit.femtoseconds),
        openmm.Platform.getPlatformByName("Reference"),
    )

    context.setPositions(
        (positions * openmm.unit.angstrom).in_units_of(openmm.unit.nanometer),
    )
    openmm.LocalEnergyMinimizer.minimize(context, 5.0e-9, 1500)

    return MinimizationResult(
        inchi_key=inchi_key,
        qcarchive_id=qcarchive_id,
        force_field=input.force_field,
        mapped_smiles=input.mapped_smiles,
        coordinates=context.getState(getPositions=True)
        .getPositions()
        .value_in_unit(openmm.unit.angstrom),
        energy=context.getState(
            getEnergy=True,
        )
        .getPotentialEnergy()
        .value_in_unit(
            openmm.unit.kilocalorie_per_mole,
        ),
    )
