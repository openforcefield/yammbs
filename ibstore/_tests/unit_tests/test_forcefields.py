import openmm
import pytest
from openff.toolkit import Molecule

from ibstore._forcefields import _gaff, _smirnoff


@pytest.fixture()
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
