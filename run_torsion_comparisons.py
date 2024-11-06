import pathlib
from multiprocessing import freeze_support

from matplotlib import pyplot

from yammbs.torsion import TorsionStore
from yammbs.torsion.inputs import QCArchiveTorsionDataset


def main():
    force_fields = [
        "openff-1.0.0",
        # "openff-1.1.0",
        "openff-1.3.0",
        "openff-2.0.0",
        # "openff-2.1.0",
        "openff-2.2.1",
    ]

    if not pathlib.Path("torsiondrive-data.json").exists():
        from openff.qcsubmit.results import TorsionDriveResultCollection

        collection = TorsionDriveResultCollection.parse_file("filtered-supp-td.json")

        with open("torsiondrive-data.json", "w") as f:
            f.write(QCArchiveTorsionDataset.from_qcsubmit_collection(collection).model_dump_json())

    dataset = QCArchiveTorsionDataset.model_validate_json(open("torsiondrive-data.json").read())

    if pathlib.Path("torsion-example.sqlite").exists():
        store = TorsionStore("torsion-example.sqlite")
    else:
        store = TorsionStore.from_torsion_dataset(
            dataset,
            database_name="torsion-example.sqlite",
        )

    for force_field in force_fields:
        store.optimize_mm(force_field=force_field, n_processes=24)

    fig, axes = pyplot.subplots(5, 4, figsize=(20, 20))

    for molecule_id, axis in zip(store.get_molecule_ids(), axes.flatten()):
        qm = store.get_qm_energies_by_molecule_id(molecule_id)
        qm_min = min(qm.values())

        for key in qm:
            qm[key] -= qm_min

        sorted_qm = dict(sorted(qm.items()))
        axis.plot(sorted_qm.keys(), sorted_qm.values(), "k.-", label=f"QM {molecule_id}")

        for force_field in force_fields:
            mm = dict(sorted(store.get_mm_energies_by_molecule_id(molecule_id, force_field=force_field).items()))
            if len(mm) == 0:
                continue
            mm_min = min(mm.values())

            axis.plot(mm.keys(), [val - mm_min for val in mm.values()], "o--", label=force_field)

        axis.legend(loc=0)

    fig.savefig("random.png")


if __name__ == "__main__":
    # This setup is necessary for reasons that confused me - both setting it up in the __main__ block and calling
    # freeze_support(). This is probably not necessary after MoleculeStore.optimize_mm() is called, so you can load up
    # the same database for later analysis once the MM conformers are stored
    freeze_support()
    main()
