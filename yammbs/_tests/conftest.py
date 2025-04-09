# Ensure QCPortal is imported before any OpenEye modules, see
# https://github.com/conda-forge/qcfractal-feedstock/issues/43
try:
    import qcportal
except ImportError:
    qcportal = None

import multiprocessing
import os
import shutil

import pytest
from openff.interchange._tests import MoleculeWithConformer
from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit import Molecule
from openff.utilities.utilities import get_data_file_path

from yammbs import MoleculeStore
from yammbs.inputs import QCArchiveDataset
from yammbs.torsion.inputs import QCArchiveTorsionDataset, QCArchiveTorsionProfile


@pytest.fixture
def guess_n_processes() -> int:
    if os.getenv("GITHUB_ACTIONS"):
        return 4
    else:
        return multiprocessing.cpu_count()


@pytest.fixture
def water():
    return MoleculeWithConformer.from_smiles("O")


@pytest.fixture
def hydrogen_peroxide():
    return MoleculeWithConformer.from_smiles("OO")


@pytest.fixture
def formaldehyde():
    return MoleculeWithConformer.from_smiles("C=O")


@pytest.fixture
def ligand():
    """Return a ligand that can have many viable conformers."""
    molecule = Molecule.from_smiles("C[C@@H](C(c1ccccc1)c2ccccc2)Nc3c4cc(c(cc4ncn3)F)F")
    molecule.generate_conformers(n_conformers=100)

    return molecule


@pytest.fixture
def small_qcsubmit_collection() -> OptimizationResultCollection:
    return OptimizationResultCollection.parse_file(
        get_data_file_path("_tests/data/qcsubmit/01-processed-qm-ch.json", "yammbs"),
    )


@pytest.fixture
def small_dataset() -> QCArchiveDataset:
    return QCArchiveDataset.from_json(
        get_data_file_path("_tests/data/yammbs/01-processed-qm-ch.json", "yammbs"),
    )


@pytest.fixture
def small_store(tmp_path) -> MoleculeStore:
    """Return a small molecule store, copied from a single source and provided as a temporary file."""
    # This file manually generated from data/01-processed-qm-ch.json
    source_path = get_data_file_path(
        "_tests/data/ch.sqlite",
        package_name="yammbs",
    )

    dest_path = (tmp_path / "ch.sqlite").as_posix()

    shutil.copy(source_path, dest_path)

    return MoleculeStore(dest_path)


@pytest.fixture
def tiny_cache() -> QCArchiveDataset:
    """Return the "tiny" molecule store, copied from a single source and provided as a temporary file."""
    with open(
        get_data_file_path(
            "_tests/data/tiny-opt.json",
            package_name="yammbs",
        ),
    ) as inp:
        return QCArchiveDataset.model_validate_json(inp.read())


@pytest.fixture
def diphenylvinylbenzene():
    """Return 1,2-diphenylvinylbenzene."""
    return Molecule.from_smiles("c1ccc(cc1)C=C(c2ccccc2)c3ccccc3")


@pytest.fixture
def allicin():
    """Return allicin, inspired by PQR."""
    return Molecule.from_smiles(
        "C=CCSS(=O)CC=C",
        allow_undefined_stereo=True,
    )


@pytest.fixture
def conformers(allicin):
    other_allicin = Molecule(allicin)

    other_allicin.generate_conformers(n_conformers=10)

    return other_allicin.conformers


@pytest.fixture
def torsion_dataset():
    return QCArchiveTorsionDataset.model_validate_json(
        open(
            get_data_file_path(
                "_tests/data/yammbs/torsiondrive-data.json",
                "yammbs",
            ),
        ).read(),
    )


@pytest.fixture
def single_torsion_dataset(torsion_dataset):
    """`torsion_dataset` with only one (QM) torsion profile."""
    # TODO: The dict round-trip should not be necessary
    return QCArchiveTorsionDataset(
        tag=torsion_dataset.tag,
        version=torsion_dataset.version,
        qm_torsions=[QCArchiveTorsionProfile(**torsion_dataset.qm_torsions[-1].dict())],
    )
