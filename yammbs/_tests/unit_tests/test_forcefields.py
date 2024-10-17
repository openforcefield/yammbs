import openmm
import pytest
from openff.toolkit import Molecule

from yammbs._forcefields import _espaloma, _gaff, _smirnoff


@pytest.fixture
def molecule():
    return Molecule.from_smiles("CCO")


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
