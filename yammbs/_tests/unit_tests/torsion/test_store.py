import numpy
import os

from openff.qcsubmit.results import TorsionDriveResultCollection
from openff.utilities import get_data_file_path

from yammbs.torsion._store import TorsionStore


class TestTorsionStore:
    def test_from_qcsubmit_collection(self, tmp_path):
        store = TorsionStore.from_qcsubmit_collection(
            TorsionDriveResultCollection.parse_file(
                get_data_file_path(
                    "_tests/data/qcsubmit/filtered-supp-td.json",
                    "yammbs",
                ),
            ),
            database_name=tmp_path / "tmp.sqlite",
        )

        assert len(store) == 20

    def test_from_torsion_dataset(self, torsion_dataset, tmp_path):
        store = TorsionStore.from_torsion_dataset(
            torsion_dataset,
            database_name=tmp_path / "tmp.sqlite",
        )

        assert len(store) == 20

    def test_torsions_with_same_smiles_and_indices(self, tmp_path):
        """Reproduce Issue #131"""
        store = TorsionStore.from_qcsubmit_collection(
            TorsionDriveResultCollection.parse_file(
                get_data_file_path(
                    "_tests/data/qcsubmit/duplicate-smiles-atom-indices.json",
                    "yammbs",
                ),
            ),
            database_name=tmp_path / "tmp.sqlite",
        )

        # these ints are torsion IDs, same as the record IDs in the source data
        assert not numpy.allclose(
             [*store.get_qm_points_by_torsion_id(21272423).values()],
             [*store.get_qm_points_by_torsion_id(120098113).values()],
        )

        assert store.get_dihedral_indices_by_torsion_id(21272423) == store.get_dihedral_indices_by_torsion_id(120098113)

        assert store.get_smiles_by_torsion_id(21272423) == store.get_smiles_by_torsion_id(120098113)

def test_minimize_basic(single_torsion_dataset, tmp_path):
    store = TorsionStore.from_torsion_dataset(
        single_torsion_dataset,
        database_name=tmp_path / "test.sqlite",
    )

    store.optimize_mm(force_field="openff-2.2.0", n_processes=os.cpu_count())

    torsion_id = store.get_torsion_ids()[0]

    assert store.get_force_fields() == ["openff-2.2.0"]

    # stored outputs include MM profiles for 1 molecule with 1 force field
    output = store.get_outputs().model_dump()

    assert len(output["mm_torsions"]) == 1
    assert len(output["mm_torsions"]["openff-2.2.0"]) == 1

    # metrics are similarly-shaped, since there's only 1 profile
    metrics = store.get_metrics().model_dump()

    assert len(metrics["metrics"]) == 1
    assert len(metrics["metrics"]["openff-2.2.0"]) == 1
