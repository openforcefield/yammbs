import numpy
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from ibstore._minimize import MinimizationInput, _run_openmm


@pytest.fixture()
def ethane() -> Molecule:
    """Return an ethane with a bad conformer"""
    ethane = Molecule.from_smiles("CC")
    ethane.generate_conformers(n_conformers=1)

    return ethane


@pytest.fixture()
def perturbed_ethane(ethane) -> Molecule:
    """Return an ethane with a bad conformer"""
    ethane._conformers[0] *= 1.2

    return ethane


def basic_input(force_field="openff-1.0.0") -> MinimizationInput:
    ethane = Molecule.from_smiles("CC")
    ethane.generate_conformers(n_conformers=1)

    return MinimizationInput(
        inchi_key=ethane.to_inchikey(),
        qcarchive_id="test",
        force_field=force_field,
        mapped_smiles=ethane.to_smiles(mapped=True),
        coordinates=ethane.conformers[0].m_as(unit.angstrom),
    )


@pytest.fixture()
def perturbed_input(perturbed_ethane) -> MinimizationInput:
    return MinimizationInput(
        inchi_key=perturbed_ethane.to_inchikey(),
        qcarchive_id="test",
        force_field="openff-1.0.0",
        mapped_smiles=perturbed_ethane.to_smiles(mapped=True),
        coordinates=perturbed_ethane.conformers[0].m_as(unit.angstrom),
    )


def test_minimization_basic(perturbed_input):
    # This C-C bond distance should be ~1.5 * 1.2 ~= 1.8
    initial = (perturbed_input.coordinates[0] - perturbed_input.coordinates[1],)
    assert numpy.linalg.norm(initial) > 1.6

    result = _run_openmm(perturbed_input)

    for attr in (
        "inchi_key",
        "qcarchive_id",
        "mapped_smiles",
    ):
        assert getattr(perturbed_input, attr) == getattr(result, attr)

    # Check that the C-C bond distance has settled to ~1.5 A
    final = result.coordinates[0] - result.coordinates[1]
    assert 1.5 < numpy.linalg.norm(final) < 1.6


def test_same_force_field_same_results():
    energy1 = _run_openmm(basic_input("openff-1.0.0")).energy
    energy2 = _run_openmm(basic_input("openff-1.0.0")).energy

    assert energy1 == energy2


def test_different_force_fields_different_results():
    energy1 = _run_openmm(basic_input("openff-1.0.0")).energy
    energy2 = _run_openmm(basic_input("openff-2.0.0")).energy

    assert energy1 != energy2


def test_plugin_loadable(ethane):
    _run_openmm(
        MinimizationInput(
            inchi_key=ethane.to_inchikey(),
            qcarchive_id="test",
            force_field="de-force-1.0.1",
            mapped_smiles=ethane.to_smiles(mapped=True),
            coordinates=ethane.conformers[0].m_as(unit.angstrom),
        ),
    )
