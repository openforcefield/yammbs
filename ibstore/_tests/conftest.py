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
def small_store() -> MoleculeStore:
    return MoleculeStore(
        get_data_file_path(
            "_tests/data/ch.sqlite",
            package_name="ibstore",
        ),
    )
