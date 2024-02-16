# Ensure QCPortal is imported before any OpenEye modules, see
# https://github.com/conda-forge/qcfractal-feedstock/issues/43
try:
    import qcportal
except ImportError:
    qcportal = None

import shutil

import pytest
from openff.interchange._tests import MoleculeWithConformer
from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit import Molecule
from openff.utilities.utilities import get_data_file_path

from ibstore._store import MoleculeStore


@pytest.fixture()
def water():
    return MoleculeWithConformer.from_smiles("O")


@pytest.fixture()
def hydrogen_peroxide():
    return MoleculeWithConformer.from_smiles("OO")


@pytest.fixture()
def formaldehyde():
    return MoleculeWithConformer.from_smiles("C=O")


@pytest.fixture()
def ligand():
    """Return a ligand that can have many viable conformers."""
    molecule = Molecule.from_smiles("C[C@@H](C(c1ccccc1)c2ccccc2)Nc3c4cc(c(cc4ncn3)F)F")
    molecule.generate_conformers(n_conformers=100)

    return molecule


@pytest.fixture()
def small_collection() -> OptimizationResultCollection:
    return OptimizationResultCollection.parse_file(
        get_data_file_path("_tests/data/01-processed-qm-ch.json", "ibstore"),
    )


@pytest.fixture()
def small_store(tmp_path) -> MoleculeStore:
    """Return a small molecule store, copied from a single source and provided as a temporary file."""
    # This file manually generated from data/01-processed-qm-ch.json
    source_path = get_data_file_path(
        "_tests/data/ch.sqlite",
        package_name="ibstore",
    )

    dest_path = (tmp_path / "ch.sqlite").as_posix()

    shutil.copy(source_path, dest_path)

    return MoleculeStore(dest_path)


@pytest.fixture()
def diphenylvinylbenzene():
    """Return 1,2-diphenylvinylbenzene"""
    return Molecule.from_smiles("c1ccc(cc1)C=C(c2ccccc2)c3ccccc3")
