import pathlib

from yammbs.torsion import TorsionStore
from yammbs.torsion.inputs import QCArchiveTorsionDataset


def main():
    force_fields = [
        "openff-1.0.0",
        "openff-1.1.0",
        "openff-1.2.0",
        "openff-1.3.0",
        "openff-2.0.0",
        "openff-2.1.0",
        "gaff-1.8",
        "gaff-2.11",
        "de-force-1.0.1.offxml",
    ]

    if not pathlib.Path("torsiondrive-data.json").exists():
        from openff.qcsubmit.results import TorsionDriveResultCollection

        collection = TorsionDriveResultCollection.parse_file("filtered-supp-td.json")

        QCArchiveTorsionDataset.from_qcsubmit_collection(collection).model_dump_json("torsiondrive-data.json")

    dataset = QCArchiveTorsionDataset.model_validate_json("torsiondrive-data.json")

    store = TorsionStore.from_qm_dataset(
        dataset,
        database_name="torsion-example.sqlite",
    )

    for force_field in force_fields:
        store.optimize_mm(force_field=force_field, n_processes=10)
