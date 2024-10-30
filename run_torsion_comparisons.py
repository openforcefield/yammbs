import pathlib
from multiprocessing import freeze_support

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

        with open("torsiondrive-data.json", "w") as f:
            f.write(QCArchiveTorsionDataset.from_qcsubmit_collection(collection).model_dump_json())

    dataset = QCArchiveTorsionDataset.model_validate_json(open("torsiondrive-data.json").read())

    store = TorsionStore.from_torsion_dataset(
        dataset,
        database_name="torsion-example.sqlite",
    )

    for force_field in force_fields:
        store.optimize_mm(force_field=force_field, n_processes=10)


if __name__ == "__main__":
    # This setup is necessary for reasons that confused me - both setting it up in the __main__ block and calling
    # freeze_support(). This is probably not necessary after MoleculeStore.optimize_mm() is called, so you can load up
    # the same database for later analysis once the MM conformers are stored
    freeze_support()
    main()
