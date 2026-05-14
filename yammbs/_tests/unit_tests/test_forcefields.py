import math

import openmm
import openmm.unit
import pytest
from openff.interchange._tests import MoleculeWithConformer

from yammbs._forcefields import _espaloma, _gaff, _openmm_ml, _smirnoff, build_omm_system


@pytest.fixture
def molecule():
    return MoleculeWithConformer.from_smiles("CCO")


def assert_energy_is_finite(system, molecule):
    context = openmm.Context(system, openmm.VerletIntegrator(0.001))
    context.setPositions(molecule.conformers[0].to_openmm())

    energy = context.getState(energy=True).getPotentialEnergy()

    assert math.isfinite(energy.value_in_unit(openmm.unit.kilojoule_per_mole))


@pytest.mark.parametrize(
    "force_field_name",
    [
        "openff-2.1.0.offxml",
        "gaff-2.11",
        "mlp:aimnet2",
    ],
)
def test_build_omm_system_dispactch(molecule, force_field_name):
    system = build_omm_system(force_field=force_field_name, molecule=molecule)

    assert isinstance(system, openmm.System)
    assert system.getNumParticles() == molecule.n_atoms


def test_smirnoff_basic(molecule):
    system = _smirnoff(molecule, "openff-2.1.0.offxml")

    assert isinstance(system, openmm.System)
    assert system.getNumParticles() == molecule.n_atoms


def test_gaff_basic(molecule):
    system = _gaff(molecule, "gaff-2.11")

    assert isinstance(system, openmm.System)
    assert system.getNumParticles() == molecule.n_atoms


def test_gaff_unsupported(molecule):
    with pytest.raises(NotImplementedError):
        _gaff(molecule, "foo")


def test_espaloma_basic(molecule):
    pytest.importorskip("espaloma")

    system = _espaloma(molecule, "espaloma-openff_unconstrained-2.1.0")

    assert isinstance(system, openmm.System)
    assert system.getNumParticles() == molecule.n_atoms


def test_espaloma_unsupported(molecule):
    pytest.importorskip("espaloma")

    with pytest.raises(NotImplementedError):
        _espaloma(molecule, "foo")


@pytest.mark.parametrize("force_field_name", ["mlp:aimnet2", "mlp:orb-v3-conservative-omol"])
def test_openmm_ml_mlps_have_energies(molecule, force_field_name):
    system = _openmm_ml(molecule, force_field_name=force_field_name)

    assert isinstance(system, openmm.System)
    assert system.getNumParticles() == molecule.n_atoms

    # minimizing with MLPs can be quite slow on generic hardware, so just make sure we can get an energy
    assert_energy_is_finite(system, molecule)


@pytest.mark.parametrize(
    "force_field_name,error",
    [
        ("mlp:magic", NotImplementedError),
        ("aimnet2", NotImplementedError),
    ],
)
def test_unsupported_mlp(molecule, force_field_name, error):
    with pytest.raises(error):
        _openmm_ml(molecule, force_field_name=force_field_name)
