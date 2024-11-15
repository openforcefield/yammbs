import pathlib
from multiprocessing import freeze_support

from matplotlib import pyplot

from yammbs.torsion import TorsionStore
from yammbs.torsion.inputs import QCArchiveTorsionDataset


def main():
    force_fields = [
        "openff-1.0.0",
        # "openff-1.1.0",
        # "openff-1.3.0",
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
        _qm = store.get_qm_energies_by_molecule_id(molecule_id)

        _qm = dict(sorted(_qm.items()))

        qm_minimum_index = min(_qm, key=_qm.get)

        # Make a new dict to avoid in-place modification while iterating
        qm = {key: _qm[key] - _qm[qm_minimum_index] for key in _qm}

        axis.plot(
            qm.keys(),
            qm.values(),
            "k.-",
            label=f"QM {molecule_id}",
        )

        for force_field in force_fields:
            mm = dict(sorted(store.get_mm_energies_by_molecule_id(molecule_id, force_field=force_field).items()))
            if len(mm) == 0:
                continue

            axis.plot(
                mm.keys(),
                [val - mm[qm_minimum_index] for val in mm.values()],
                "o--",
                label=force_field,
            )

        axis.legend(loc=0)
        axis.grid(axis="both")

    fig.savefig("random.png")


if __name__ == "__main__":
    # This setup is necessary for reasons that confused me - both setting it up in the __main__ block and calling
    # freeze_support(). This is probably not necessary after MoleculeStore.optimize_mm() is called, so you can load up
    # the same database for later analysis once the MM conformers are stored
    freeze_support()
    main()
